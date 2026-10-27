"""Neovim in a PTY, editing a private copy of one document.

- The browser renders the PTY through xterm.js, the backend only relays bytes
- `:w` never writes the copy: a buffer-local `BufWriteCmd` sends the text over
  msgpack-RPC to the caller's `save`, which persists it and returns what was stored
- A `TextYankPost` autocmd notifies the caller's `on_yank` with the yanked text, for the browser clipboard
- Closing a session discards unsaved edits
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import fcntl
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import termios

from lib.neovim_rpc import SETUP_LUA, NeovimRpc, NeovimRpcError

SaveDocument = Callable[[str], Awaitable[str]]
YankText = Callable[[str], None]

_STARTUP_SECONDS = 15
_SHUTDOWN_SECONDS = 2
_READ_CHUNK = 65536


def is_neovim_installed() -> bool:
    """`nvim` on PATH and a config (under `XDG_CONFIG_HOME`) to load."""
    config_dir = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "nvim"
    return shutil.which("nvim") is not None and config_dir.is_dir()


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
    >>> await session.connect(save, on_yank)  # `:w` calls `save(text)`, yanks call `on_yank(text)`
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
        self._rpc: NeovimRpc | None = None
        self._loop = asyncio.get_running_loop()
        self._loop.add_reader(terminal_fd, self._read_terminal)

    @classmethod
    def spawn(cls, filename: str, content: str, *, rows: int, cols: int, cwd: Path, base_content: str | None = None) -> NeovimSession:
        """Start `nvim` in *cwd* on a private copy of *content*, diffed against a read-only *base_content* when given.

        Call from a running event loop.
        """
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

    async def connect(self, save: SaveDocument, on_yank: YankText) -> None:
        """Route `:w` of the document buffer to *save* (its return value replaces the buffer if it differs) and yanks to *on_yank*."""

        async def handle_request(method: str, params: list) -> object:
            if method != "save":
                raise NeovimRpcError(f"Unknown request {method!r}")
            return await save(params[0])

        def handle_notification(method: str, params: list) -> None:
            if method == "yank":
                on_yank(params[0])

        socket_path = self._workdir / "rpc.sock"
        deadline = self._loop.time() + _STARTUP_SECONDS
        while not socket_path.exists():
            if self._process.poll() is not None:
                raise RuntimeError(f"Neovim exited during startup with code {self._process.returncode}")
            if self._loop.time() > deadline:
                raise TimeoutError("Neovim did not open its RPC socket")
            await asyncio.sleep(0.02)
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        self._rpc = NeovimRpc(reader, writer, handle_request, handle_notification)
        channel, _ = await asyncio.wait_for(self._rpc.request("nvim_get_api_info"), _STARTUP_SECONDS)
        while not await asyncio.wait_for(self._rpc.request("nvim_eval", "v:vim_did_enter"), _STARTUP_SECONDS):
            if self._loop.time() > deadline + _STARTUP_SECONDS:
                raise TimeoutError("Neovim did not finish starting")
            await asyncio.sleep(0.02)  # config and plugins still loading, FileType handlers not registered yet
        base = str(self._base) if self._base is not None else None
        setup = self._rpc.request("nvim_exec_lua", SETUP_LUA, [channel, str(self._document), base])
        await asyncio.wait_for(setup, _STARTUP_SECONDS)

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
