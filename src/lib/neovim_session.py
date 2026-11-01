"""Neovim in a PTY, editing a private copy of one document.

- The browser renders the PTY through xterm.js, the backend only relays bytes
- `:w` never writes the copy: a buffer-local `BufWriteCmd` sends the text over
  msgpack-RPC to the caller's `save`, which persists it and returns what was stored
- `base` is the text Neovim and the stored file last agreed on (open, save, `set_text`). Edits made elsewhere
  merge into unsaved ones (`lib.text_merge.merge_theirs`, a conflict takes the other side's text)
- A `TextYankPost` autocmd notifies the caller's `on_yank` with the yanked text, for the browser clipboard
- With an `on_selection` callback, visual-mode `<leader>a` reports the selected lines
- `set_text` takes edits made to the file elsewhere without moving the cursor: an unmodified buffer shows them,
  a modified one merges them and stays modified
- `:w` of a session with a `disk_path` whose text left `base` saves the buffer merged with the disk text, and fails
  without writing when that text is not UTF-8 or the merge fails. `set_text` waits for a running `:w`
- `set_background` sets 'background' to the browser theme's, so the colorscheme picks its light or dark palette
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
from lib.text_merge import TextMergeError, merge_theirs

SaveDocument = Callable[[str], Awaitable[str]]
YankText = Callable[[str], None]
Selection = dict[str, int | str]
SelectLines = Callable[[Selection], None]

_STARTUP_SECONDS = 15
_RPC_SECONDS = 10
_SHUTDOWN_SECONDS = 2
_READ_CHUNK = 65536
# `set_text` retries while keystrokes change the buffer between reading and merging it.
_SET_TEXT_ATTEMPTS = 5


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

    def __init__(
        self,
        process: subprocess.Popen,
        terminal_fd: int,
        workdir: Path,
        document: Path,
        *,
        base: str,
        baseline: Path | None = None,
        disk_path: Path | None = None,
    ) -> None:
        self.base = base
        # Held by `set_text` and by `:w` until `save` answered, so neither runs in the middle of the other
        self._sync = asyncio.Lock()
        self._saves = 0
        self._baseline = baseline
        self._disk_path = disk_path
        self._process = process
        self._terminal_fd = terminal_fd
        self._workdir = workdir
        self._document = document
        self._output: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._rpc: NeovimRpc | None = None
        # 'background' the browser asked for, applied once setup finished, so the user's config cannot override it
        self._background: str | None = None
        self._ready = False
        self._loop = asyncio.get_running_loop()
        self._loop.add_reader(terminal_fd, self._read_terminal)

    @classmethod
    def spawn(
        cls,
        filename: str,
        content: str,
        *,
        rows: int,
        cols: int,
        cwd: Path,
        base_content: str | None = None,
        disk_path: Path | None = None,
    ) -> NeovimSession:
        """Start `nvim` in *cwd* on a private copy of *content*, diffed against a read-only *base_content* when given.

        *disk_path* is the file *content* was read from, `:w` merges its later changes. Call from a running event loop.
        """
        workdir = Path(tempfile.mkdtemp(prefix="gigachad-nvim-"))
        document = workdir / _document_name(filename)
        document.write_text(content, encoding="utf-8", newline="")
        baseline: Path | None = None
        if base_content is not None:
            (workdir / "base").mkdir()
            baseline = workdir / "base" / _document_name(filename)
            baseline.write_text(base_content, encoding="utf-8", newline="")
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
        return cls(process, terminal_fd, workdir, document, base=content, baseline=baseline, disk_path=disk_path)

    async def connect(self, save: SaveDocument, on_yank: YankText, on_selection: SelectLines | None = None) -> None:
        """Route `:w` of the document buffer to *save* (its return value replaces the buffer if it differs) and yanks to *on_yank*.

        `:w` sends the buffer merged with the disk text when the file at `disk_path` changed since `base` (or with the text
        of a `set_text` that ran while the save waited), and holds off `set_text` until *save* returned.

        Giving *on_selection* also installs the visual-mode `<leader>a` mapping, it receives `{"start_line", "end_line", "text"}`.
        """

        async def handle_request(method: str, params: list) -> object:
            if method != "save":
                raise NeovimRpcError(f"Unknown request {method!r}")
            return await self._save(save, params[0])

        def handle_notification(method: str, params: list) -> None:
            if method == "yank":
                on_yank(params[0])
            elif method == "selection" and on_selection is not None:
                on_selection(params[0])

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
        base = str(self._baseline) if self._baseline is not None else None
        setup = self._rpc.request("nvim_exec_lua", SETUP_LUA, [channel, str(self._document), base, on_selection is not None])
        await asyncio.wait_for(setup, _STARTUP_SECONDS)
        self._ready = True
        if self._background is not None:
            await self._apply_background()

    async def set_text(self, content: str) -> None:
        """Show *content* (now on disk) keeping cursor and view, merged into unsaved edits of a modified buffer.

        Waits for a running `:w`. When one stored the buffer meanwhile, *content* and the stored text both grew from
        the old `base`: the buffer takes their merge, stays modified when that differs from the stored text, and
        `base` stays the stored text.
        """
        base, saves = self.base, self._saves
        async with self._sync:
            for _ in range(_SET_TEXT_ATTEMPTS):
                state = await self._lua("return _G.gigachad_buffer_state()")
                if self._saves == saves:
                    merge = state["modified"]
                    text = await asyncio.to_thread(merge_theirs, self.base, state["text"], content) if merge else content
                    dirty, agreed = False, content
                else:
                    text = await asyncio.to_thread(merge_theirs, base, state["text"], content)
                    dirty, agreed = text != self.base, self.base
                if await self._lua("return _G.gigachad_set_text(...)", text, state["tick"], dirty):
                    self.base = agreed
                    return
        raise RuntimeError("Neovim buffer kept changing, the changed file was not applied")

    async def _save(self, save: SaveDocument, text: str) -> dict[str, str | bool]:
        """Store *text* merged with the edits made elsewhere since `base`, `merged` tells whether they changed it."""
        base = self.base  # *text* grew from it, a `set_text` holding the lock may still move `self.base` on
        async with self._sync:
            theirs = await asyncio.to_thread(self._disk_text)
            if theirs is None:
                theirs = self.base  # no disk file: the last `set_text` is the newest text from elsewhere
            merged = text
            if theirs != base:
                try:
                    merged = await asyncio.to_thread(merge_theirs, base, text, theirs)
                except TextMergeError as exc:
                    raise TextMergeError(f"could not merge with the changed file, nothing written ({exc})") from exc
            stored = await save(merged)
            self.base = stored
            self._saves += 1
        return {"text": stored, "merged": merged != text}

    def _disk_text(self) -> str | None:
        """Current text at `disk_path`, None without one or once it was deleted. Raises when it cannot be read."""
        if self._disk_path is None:
            return None
        try:
            return self._disk_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"{self._disk_path.name} on disk is not UTF-8 text, nothing written") from exc
        except OSError as exc:
            raise RuntimeError(f"cannot read {self._disk_path.name} on disk to merge with it, nothing written: {exc}") from exc

    async def _lua(self, code: str, *args: object) -> object:
        if self._rpc is None:
            raise RuntimeError("Neovim is not connected")
        return await asyncio.wait_for(self._rpc.request("nvim_exec_lua", code, list(args)), _RPC_SECONDS)

    async def set_background(self, light: bool) -> None:
        """Set 'background' to light or dark now, or once connected. Neovim reloads the colorscheme on a change."""
        self._background = "light" if light else "dark"
        if self._ready:
            await self._apply_background()

    async def _apply_background(self) -> None:
        assert self._rpc is not None
        await asyncio.wait_for(self._rpc.request("nvim_set_option_value", "background", self._background, {}), _RPC_SECONDS)

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
