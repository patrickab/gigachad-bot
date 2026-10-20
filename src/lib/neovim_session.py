"""Neovim in a PTY, editing a private copy of one document.

- The browser renders the PTY through xterm.js, the backend only relays bytes
- `:w` never writes the copy: a buffer-local `BufWriteCmd` sends the text over
  msgpack-RPC to the caller's `save`, which persists it and returns what was stored
- Closing a session discards unsaved edits
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import contextlib
import fcntl
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import termios

import msgpack

SaveDocument = Callable[[str], Awaitable[str]]

_STARTUP_SECONDS = 15
_SHUTDOWN_SECONDS = 2
_READ_CHUNK = 65536

# Receives (channel, working path, optional baseline path). Mirrors Neovim's write rules:
# empty, otherwise a final newline unless both 'eol' and 'fixeol' are off.
_SETUP_LUA = r"""
local channel, path, base_path = ...
if base_path == vim.NIL then base_path = nil end
local buffer = vim.fn.bufadd(path)
vim.fn.bufload(buffer)
vim.api.nvim_win_set_buf(0, buffer)

local function buffer_text()
  local lines = vim.api.nvim_buf_get_lines(buffer, 0, -1, false)
  if #lines == 1 and lines[1] == "" then return "" end
  local text = table.concat(lines, "\n")
  if vim.bo[buffer].eol or vim.bo[buffer].fixeol then text = text .. "\n" end
  return text
end

local function show_stored(text)
  local lines = vim.split(text, "\n", { plain = true })
  if #lines > 1 and lines[#lines] == "" then table.remove(lines) else vim.bo[buffer].eol = false end
  vim.api.nvim_buf_set_lines(buffer, 0, -1, false, lines)
end

vim.api.nvim_create_autocmd("BufWriteCmd", {
  buffer = buffer,
  callback = function()
    local text = buffer_text()
    local ok, stored = pcall(vim.rpcrequest, channel, "save", text)
    if not ok then error("Save failed: " .. tostring(stored), 0) end
    if stored ~= text then show_stored(stored) end
    vim.bo[buffer].modified = false
    vim.api.nvim_echo({ { "Saved " .. vim.fn.fnamemodify(path, ":t") } }, false, {})
  end,
})

if base_path then
  local work_window = vim.api.nvim_get_current_win()
  vim.cmd("leftabove vertical split " .. vim.fn.fnameescape(base_path))
  local base_buffer = vim.api.nvim_get_current_buf()
  vim.bo[base_buffer].readonly = true
  vim.bo[base_buffer].modifiable = false
  vim.api.nvim_create_autocmd("BufWriteCmd", {
    buffer = base_buffer,
    callback = function() error("Baseline is read-only", 0) end,
  })
  vim.cmd("diffthis")
  vim.api.nvim_set_current_win(work_window)
  vim.cmd("diffthis")
end
"""


def neovim_config_dir() -> Path:
    """Directory Neovim loads its config from, honoring `XDG_CONFIG_HOME`."""
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "nvim"


def is_neovim_installed() -> bool:
    """`nvim` on PATH and a config to load."""
    return shutil.which("nvim") is not None and neovim_config_dir().is_dir()


class NeovimRpcError(RuntimeError):
    """Neovim answered a request with an error."""


class _NeovimRpc:
    """Minimal msgpack-RPC client over Neovim's `--listen` socket."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        handle_request: Callable[[str, list], Awaitable[object]],
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._handle_request = handle_request
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._answers: set[asyncio.Task] = set()
        self._read_task = asyncio.create_task(self._read_messages())

    async def request(self, method: str, *params: object) -> object:
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[self._next_id] = future
        self._send([0, self._next_id, method, list(params)])
        return await future

    async def close(self) -> None:
        self._read_task.cancel()
        for task in (self._read_task, *self._answers):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._writer.close()

    def _send(self, message: list) -> None:
        self._writer.write(msgpack.packb(message))

    async def _read_messages(self) -> None:
        unpacker = msgpack.Unpacker(raw=False, unicode_errors="replace")
        try:
            while data := await self._reader.read(_READ_CHUNK):
                unpacker.feed(data)
                for message in unpacker:
                    self._dispatch(message)
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ConnectionError("Neovim RPC connection closed"))

    def _dispatch(self, message: list) -> None:
        """Resolve responses, answer requests, ignore notifications."""
        if message[0] == 1:
            _, message_id, error, result = message
            future = self._pending.pop(message_id, None)
            if future is None or future.done():
                return
            if error:
                future.set_exception(NeovimRpcError(str(error)))
            else:
                future.set_result(result)
        elif message[0] == 0:
            _, message_id, method, params = message
            task = asyncio.create_task(self._answer(message_id, method, params))
            self._answers.add(task)
            task.add_done_callback(self._answers.discard)

    async def _answer(self, message_id: int, method: str, params: list) -> None:
        try:
            result = await self._handle_request(method, params)
        except Exception as exc:
            self._send([1, message_id, str(exc), None])
        else:
            self._send([1, message_id, None, result])


def _document_name(filename: str) -> str:
    """Basename for the private copy; keeps the extension so filetype detection works."""
    name = Path(filename).name
    return name if name not in {"", ".", ".."} else "document.md"


def _set_terminal_size(fd: int, rows: int, cols: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _claim_controlling_terminal() -> None:
    """Make the PTY the child's controlling terminal so resizes deliver SIGWINCH."""
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


def _environment() -> dict[str, str]:
    environment = {**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"}
    environment.pop("NVIM", None)  # set when the backend itself runs inside a Neovim terminal
    return environment


class NeovimSession:
    """One `nvim` process in a PTY, editing a temporary copy of one document.

    Examples
    --------
    >>> session = NeovimSession.spawn("notes.md", "# Notes\\n", rows=40, cols=120)
    >>> await session.connect(save)  # `:w` now calls `save(text)`
    >>> while (output := await session.read_output()) is not None: ...
    >>> await session.close()
    """

    def __init__(self, process: subprocess.Popen, terminal_fd: int, workdir: Path, document: Path, base: Path | None = None) -> None:
        self._base = base
        self._process = process
        self._terminal_fd = terminal_fd
        self._workdir = workdir
        self._document = document
        self._output: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._rpc: _NeovimRpc | None = None
        self._loop = asyncio.get_running_loop()
        self._loop.add_reader(terminal_fd, self._read_terminal)

    @classmethod
    def spawn(cls, filename: str, content: str, *, rows: int, cols: int, cwd: Path, base_content: str | None = None) -> NeovimSession:
        """Start `nvim` in *cwd* on a private copy of *content*, diffed against a read-only *base_content* when given. Call from a running event loop."""
        workdir = Path(tempfile.mkdtemp(prefix="gigachad-nvim-"))
        document = workdir / _document_name(filename)
        document.write_text(content, encoding="utf-8", newline="")
        base: Path | None = None
        if base_content is not None:
            (workdir / "base").mkdir()
            base = workdir / "base" / _document_name(filename)
            base.write_text(base_content, encoding="utf-8", newline="")
        terminal_fd, child_fd = os.openpty()
        _set_terminal_size(terminal_fd, rows, cols)
        try:
            process = subprocess.Popen(
                ["nvim", "-n", "--listen", str(workdir / "rpc.sock"), str(document)],
                stdin=child_fd,
                stdout=child_fd,
                stderr=child_fd,
                cwd=cwd,
                env=_environment(),
                start_new_session=True,
                preexec_fn=_claim_controlling_terminal,
            )
        except BaseException:
            os.close(terminal_fd)
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        finally:
            os.close(child_fd)
        return cls(process, terminal_fd, workdir, document, base)

    async def connect(self, save: SaveDocument) -> None:
        """Route `:w` of the document buffer to *save*, whose return value replaces the buffer if it differs."""

        async def handle_request(method: str, params: list) -> object:
            if method != "save":
                raise NeovimRpcError(f"Unknown request {method!r}")
            return await save(params[0])

        socket_path = self._workdir / "rpc.sock"
        deadline = self._loop.time() + _STARTUP_SECONDS
        while not socket_path.exists():
            if self._process.poll() is not None:
                raise RuntimeError(f"Neovim exited during startup with code {self._process.returncode}")
            if self._loop.time() > deadline:
                raise TimeoutError("Neovim did not open its RPC socket")
            await asyncio.sleep(0.02)
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        self._rpc = _NeovimRpc(reader, writer, handle_request)
        channel, _ = await asyncio.wait_for(self._rpc.request("nvim_get_api_info"), _STARTUP_SECONDS)
        base = str(self._base) if self._base is not None else None
        await asyncio.wait_for(self._rpc.request("nvim_exec_lua", _SETUP_LUA, [channel, str(self._document), base]), _STARTUP_SECONDS)

    async def read_output(self) -> bytes | None:
        """Next terminal output chunk, None once the terminal has closed."""
        return await self._output.get()

    def write_input(self, data: bytes) -> None:
        """Forward keystrokes, dropped once the terminal has closed."""
        view = memoryview(data)
        while view and self._terminal_fd >= 0:
            view = view[os.write(self._terminal_fd, view) :]

    def resize(self, rows: int, cols: int) -> None:
        if self._terminal_fd >= 0:
            _set_terminal_size(self._terminal_fd, rows, cols)

    async def wait(self) -> int:
        """Exit code of `nvim`."""
        return await asyncio.to_thread(self._process.wait)

    async def close(self) -> None:
        """Stop `nvim` and delete the private copy, discarding unsaved edits."""
        if self._rpc is not None:
            await self._rpc.close()
        if self._process.poll() is None:
            self._process.terminate()
            try:
                await asyncio.to_thread(self._process.wait, _SHUTDOWN_SECONDS)
            except subprocess.TimeoutExpired:
                self._process.kill()
                await asyncio.to_thread(self._process.wait)
        self._stop_reading()
        shutil.rmtree(self._workdir, ignore_errors=True)

    def _read_terminal(self) -> None:
        try:
            data = os.read(self._terminal_fd, _READ_CHUNK)
        except OSError:  # EIO once the child side of the PTY is closed
            data = b""
        if data:
            self._output.put_nowait(data)
        else:
            self._stop_reading()

    def _stop_reading(self) -> None:
        if self._terminal_fd < 0:
            return
        self._loop.remove_reader(self._terminal_fd)
        os.close(self._terminal_fd)
        self._terminal_fd = -1
        self._output.put_nowait(None)
