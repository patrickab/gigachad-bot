"""Neovim in a PTY, editing private copies of documents.

- The browser renders the PTY through xterm.js, the backend only relays bytes
- A document is known by its *path* (the app path, absolute for files). The first one opens with the session,
  `open_file` adds more: each has its own private copy (a directory of its own, so equal basenames never collide)
  and its own buffer, kept loaded when another is shown. Switching never saves
- `:w` never writes a copy: a buffer-local `BufWriteCmd` sends the text over msgpack-RPC to the caller's
  `save(path, text)`, which persists it and returns what was stored
- `base` is the text Neovim and the stored file last agreed on, per document (open, save, `set_text`). Edits made
  elsewhere merge into unsaved ones (`lib.text_merge.merge_theirs`, a conflict takes the other side's text)
- A `TextYankPost` autocmd notifies the caller's `on_yank` with the yanked text
- With an `on_selection` callback, visual-mode `<leader>a` reports the selected lines and their document
- With an `on_forward_search` callback, normal-mode `<leader>v` in a `.tex` document reports the document and its cursor line
- `on_switched` hears of a document Neovim itself put on screen (`:bnext`, `:b`), not of one `open_file` showed
- `set_text` takes edits made to the file elsewhere without moving the cursor: an unmodified buffer shows them,
  a modified one merges them and stays modified. It works on hidden buffers too
- `:w` of a document with a `disk_path` whose text left `base` saves the buffer merged with the disk text, and fails
  without writing when that text is not UTF-8 or the merge fails. `set_text` waits for a running `:w` of its document
- `rename` moves a document to the file's new path: the buffer is renamed keeping text, cursor and unsaved state,
  and `:w` merges with the new `disk_path`
- `insert_lines` puts lines below the cursor of the shown document
- `insert_frame` puts a beamer frame after the frame around the cursor, `insert_image` an image path into the cursor line's
  placeholder, else a reference below it. `lib.tex_source` works the edit out from the buffer's lines and Neovim applies it
  unless keystrokes changed the buffer meanwhile, then it is worked out once more before failing
- `show_error` shows a message in Neovim as an error
- With a `read_clipboard` callback, `+` and `*` are the clipboard it reads (the browser's, not the machine's Neovim runs on)
  and replace the provider of the user's config: a paste asks `read_clipboard(path)` with the path of the shown document,
  which answers the text to paste (linewise when it ends with a newline) or raises `ClipboardError`. Neovim shows its
  message and pastes its last copy instead, and so after 15 s without an answer ("Clipboard did not answer"). A copy
  stays in Neovim's registers, the yank reaches the caller through `on_yank`. Reads run beside the other requests, so a
  paste waiting for the browser never holds up `set_text`, `:w` or the others
- `set_background` sets 'background' to the browser theme's, so the colorscheme picks its light or dark palette
- A notebook's percent text (`spawn` with *notebook*) is a document whose buffer has a virtual name beside the notebook
  (`<notebook>.ipynb.py`), so an LSP finds the project and `.venv` from the notebook's directory. Nothing is read from or
  written to that name (`:e!` reloads `base`, no swap or undo file). `<C-CR>` saves it like `:w` and then the caller's
  `on_run_cell` hears the cursor line, `<S-CR>` also moves the cursor to the next cell
- Every session turns `more` off and lets messages time out (`messagesopt`) once the user's config ran: Neovim answers no RPC
  request while it waits at a hit-enter prompt
- Closing a session discards unsaved edits
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from lib.neovim_rpc import SETUP_LUA, NeovimRpc, NeovimRpcError
from lib.process_owner import owned_environment
from lib.pty_process import claim_controlling_terminal, set_terminal_size, terminal_environment
from lib.tex_source import fill_placeholder, frame_insertion
from lib.text_merge import TextMergeError, merge_theirs

SaveDocument = Callable[[str, str], Awaitable[str]]
YankText = Callable[[str], None]
Selection = dict[str, int | str]
SelectLines = Callable[[Selection], None]
SwitchDocument = Callable[[str], None]
ReadClipboard = Callable[[str], Awaitable[str]]
RunCell = Callable[[str, int], None]
ForwardSearch = Callable[[str, int], None]

# Keys that send a notification from the documents whose path ends in `suffix` (all without one), installed when the
# notification has a callback. A visual-mode keymap sends the selected lines, any other the document's path and cursor line.
_KEYMAPS = (
    {"mode": "x", "lhs": "<leader>a", "event": "selection"},
    {"mode": "n", "lhs": "<leader>v", "event": "forward_search", "suffix": ".tex"},
)

_STARTUP_SECONDS = 15
_RPC_SECONDS = 10
_SHUTDOWN_SECONDS = 2
_READ_CHUNK = 65536
# `set_text` retries while keystrokes change the buffer between reading and merging it.
_SET_TEXT_ATTEMPTS = 5
# An insert at the cursor works out its edit once more when keystrokes changed the buffer meanwhile, then fails.
_INSERT_ATTEMPTS = 2
# How long a paste waits for the browser, which may be asking the user for permission first.
_CLIPBOARD_SECONDS = 15


def is_neovim_installed() -> bool:
    """`nvim` on PATH and a config (under `XDG_CONFIG_HOME`) to load."""
    config_dir = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "nvim"
    return shutil.which("nvim") is not None and config_dir.is_dir()


def _document_name(filename: str) -> str:
    """Basename for the private copy; keeps the extension so filetype detection works."""
    name = Path(filename).name
    return name if name not in {"", ".", ".."} else "document.md"


def _buffer_lines(text: str) -> list[str]:
    """The lines Neovim makes of *text*: a final newline ends the last line instead of starting an empty one."""
    return text.removesuffix("\n").split("\n")


def _private_copy(directory: Path, filename: str, content: str) -> Path:
    """Write *content* to a new *directory* under the basename of *filename*."""
    directory.mkdir()
    document = directory / _document_name(filename)
    document.write_text(content, encoding="utf-8", newline="")
    return document


def _read_for_opening(file: Path) -> str:
    """The UTF-8 text of *file*, a `RuntimeError` saying why not otherwise."""
    try:
        return file.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{file.name} is not UTF-8 text") from exc
    except OSError as exc:
        raise RuntimeError(f"cannot read {file.name}: {exc}") from exc


class ClipboardError(RuntimeError):
    """The clipboard could not be read, Neovim shows the message."""


async def _read_clipboard(read: ReadClipboard, path: str) -> dict[str, str]:
    """The answer to Neovim's `paste` request: `{"text"}` to paste, or `{"error"}` to show before it pastes its last copy."""
    try:
        return {"text": await asyncio.wait_for(read(path), _CLIPBOARD_SECONDS)}
    except TimeoutError:
        return {"error": "Clipboard did not answer"}
    except ClipboardError as exc:
        return {"error": str(exc)}


@dataclass(frozen=True)
class _PrivateCopy:
    """A document's own file in the session directory, `:w` merges with later changes of the file at `disk_path`."""

    name: Path
    disk_path: Path | None

    def disk_text(self) -> str | None:
        """Current text at `disk_path`, None without one or once it was deleted. Raises when it cannot be read."""
        if self.disk_path is None:
            return None
        try:
            return self.disk_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"{self.disk_path.name} on disk is not UTF-8 text, nothing written") from exc
        except OSError as exc:
            raise RuntimeError(f"cannot read {self.disk_path.name} on disk to merge with it, nothing written: {exc}") from exc

    def renamed(self, path: str, disk_path: Path | None) -> _PrivateCopy:
        """The copy of the file renamed to *path*, named like it so filetype detection follows."""
        return _PrivateCopy(self.name.with_name(_document_name(path)), disk_path)

    def discard(self) -> None:
        self.name.unlink(missing_ok=True)


@dataclass(frozen=True)
class _NotebookText:
    """The virtual name of a notebook's text beside the notebook, never read from, written to or deleted."""

    name: Path

    def disk_text(self) -> str | None:
        return None

    def renamed(self, path: str, name: Path | None) -> _NotebookText:
        """The text under the virtual name *name* of the renamed notebook, refused without one."""
        if name is None:
            raise RuntimeError(f"{_document_name(path)} is no notebook text")
        # ponytail: a file already at the new virtual name makes Neovim refuse a plain `:w` (E13), `:w!` still saves
        return _NotebookText(name)

    def discard(self) -> None:
        """Nothing to delete, the name is not ours."""


@dataclass
class _Document:
    """A document of the session: its buffer's backing and what Neovim and the stored file last agreed on."""

    path: str
    backing: _PrivateCopy | _NotebookText
    base: str
    saves: int = 0
    # Held by `set_text` and by `:w` until `save` answered, so neither runs in the middle of the other
    sync: asyncio.Lock = field(default_factory=asyncio.Lock)


class NeovimSession:
    """One `nvim` process in a PTY, editing temporary copies of documents.

    Examples
    --------
    >>> session = NeovimSession.spawn("notes.md", "# Notes\\n", rows=40, cols=120)
    >>> await session.connect(save, on_yank)  # `:w` calls `save(path, text)`, yanks call `on_yank(text)`
    >>> await session.open_file("/notes/ideas.md", Path("/notes/ideas.md"))  # a second document, shown now
    >>> while (output := await session.read_output()) is not None: ...
    >>> await session.close()
    """

    def __init__(
        self,
        process: subprocess.Popen,
        terminal_fd: int,
        workdir: Path,
        first: _Document,
        *,
        baseline: Path | None = None,
    ) -> None:
        self._documents = {first.path: first}
        self._copies = 1
        # Serializes `open_file`, which looks a document up and then adds it
        self._opening = asyncio.Lock()
        self._baseline = baseline
        self._process = process
        self._terminal_fd = terminal_fd
        self._workdir = workdir
        self._output: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._rpc: NeovimRpc | None = None
        # 'background' the browser asked for, applied once setup finished, so the user's config cannot override it
        self._background: str | None = None
        # Set once the save hook is installed, requests that change documents wait for it
        self._ready = asyncio.Event()
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
        path: str | None = None,
        notebook: Path | None = None,
    ) -> NeovimSession:
        """Start `nvim` in *cwd* on a private copy of *content*, diffed against a read-only *base_content* when given.

        *path* names the document in `save` and the other calls, default *filename*. `:w` merges later changes of *disk_path*.
        With *notebook*, the virtual name of a notebook's percent text, the buffer takes that name instead of a private copy and
        `<S-CR>`/`<C-CR>` run cells, *disk_path* is unused. Call from a running event loop.
        """
        workdir = Path(tempfile.mkdtemp(prefix="gigachad-nvim-"))
        if notebook is None:
            backing: _PrivateCopy | _NotebookText = _PrivateCopy(_private_copy(workdir / "0", filename, content), disk_path)
            start = backing.name
        else:
            backing = _NotebookText(notebook)
            # Started on a file like any session, a user config would restore a session or show a start screen without one
            start = workdir / "start"
            start.touch()
        first = _Document(path or filename, backing, content)
        baseline: Path | None = None
        if base_content is not None:
            (workdir / "base").mkdir()
            baseline = workdir / "base" / _document_name(filename)
            baseline.write_text(base_content, encoding="utf-8", newline="")
        terminal_fd, child_fd = os.openpty()
        set_terminal_size(terminal_fd, rows, cols)
        try:
            process = subprocess.Popen(
                ["nvim", "-n", "--listen", str(workdir / "rpc.sock"), str(start)],
                stdin=child_fd,
                stdout=child_fd,
                stderr=child_fd,
                cwd=cwd,
                env=owned_environment(terminal_environment()),
                start_new_session=True,
                preexec_fn=claim_controlling_terminal,
            )
        except BaseException:
            os.close(terminal_fd)
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        finally:
            os.close(child_fd)
        return cls(process, terminal_fd, workdir, first, baseline=baseline)

    async def connect(
        self,
        save: SaveDocument,
        on_yank: YankText,
        on_selection: SelectLines | None = None,
        on_switched: SwitchDocument | None = None,
        read_clipboard: ReadClipboard | None = None,
        on_run_cell: RunCell | None = None,
        on_leave: Callable[[], None] | None = None,
        on_forward_search: ForwardSearch | None = None,
    ) -> None:
        """Route `:w` of every document buffer to *save* (its return value replaces the buffer if it differs) and yanks to *on_yank*.

        `:w` sends the buffer merged with the disk text when the file at the document's `disk_path` changed since its `base`
        (or with the text of a `set_text` that ran while the save waited), and holds off that document's `set_text` until
        *save* returned.

        Giving *on_selection* also installs the visual-mode `<leader>a` mapping, it receives `{"path", "start_line", "end_line",
        "text"}`. *on_switched* receives the path of the document that Neovim itself put on screen, not of one `open_file` showed.

        Giving *read_clipboard* makes it the clipboard of `+` and `*`: a paste calls it with the path of the shown document and
        pastes the text it returns (linewise when it ends with a newline). When it raises a `ClipboardError`, or gives no answer
        within 15 s ("Clipboard did not answer"), Neovim shows the message and pastes its last copy instead.
        It runs beside every other request, Neovim answers those while it waits.

        *on_run_cell* receives the path and cursor line (1-based) of `<S-CR>` or `<C-CR>` in a notebook's text, after the save it
        made. *on_leave* hears `<Esc>` in normal mode, after the user's own normal-mode `<Esc>` mapping ran.

        Giving *on_forward_search* also installs the normal-mode `<leader>v` mapping in `.tex` documents (a rename adds or
        removes it), it receives the document's path and cursor line (1-based).
        """

        notifications: dict[str, Callable[..., None]] = {
            event: callback
            for event, callback in (
                ("yank", on_yank),
                ("selection", on_selection),
                ("switched", on_switched),
                ("leave", on_leave),
                ("run_cell", on_run_cell),
                ("forward_search", on_forward_search),
            )
            if callback is not None
        }
        requests: dict[str, Callable[..., Awaitable[object]]] = {"save": partial(self._save, save), "base": self._base}
        if read_clipboard is not None:
            requests["paste"] = partial(_read_clipboard, read_clipboard)

        async def handle_request(method: str, params: list) -> object:
            if (answer := requests.get(method)) is None:
                raise NeovimRpcError(f"Unknown request {method!r}")
            return await answer(*params)

        def handle_notification(method: str, params: list) -> None:
            if (callback := notifications.get(method)) is not None:
                callback(*params)

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
        first = next(iter(self._documents.values()))
        options = {
            "channel": channel,
            "first_path": str(first.backing.name),
            "first_id": first.path,
            "base_path": str(self._baseline) if self._baseline is not None else None,
            "keymaps": [keymap for keymap in _KEYMAPS if keymap["event"] in notifications],
            "own_clipboard": read_clipboard is not None,
            "notebook": isinstance(first.backing, _NotebookText),
        }
        setup = self._rpc.request("nvim_exec_lua", SETUP_LUA, [options])
        await asyncio.wait_for(setup, _STARTUP_SECONDS)
        self._ready.set()
        if self._background is not None:
            await self._apply_background()

    async def open_file(self, path: str, disk_path: Path | None, line: int | None = None, column: int = 0) -> None:
        """Show the document *path*, first loading *disk_path* into a new private copy when it is not open yet.

        With *line* the cursor then goes to that 1-based line and 0-based byte *column* (both clamped), centred. The document
        shown before stays loaded with its unsaved edits, nothing is saved. Raises `RuntimeError` when there is no readable
        *disk_path* or it is not UTF-8 text.
        """
        async with self._opening:
            if not (path in self._documents and await self._lua("return _G.gigachad_show(...)", path)):
                self._forget(path)  # Neovim lost the buffer (`:bdelete`), the copy is stale
                if disk_path is None:
                    raise RuntimeError(f"{_document_name(path)} is not a readable file")
                content = await asyncio.to_thread(_read_for_opening, disk_path)
                copy = _PrivateCopy(_private_copy(self._workdir / str(self._copies), path, content), disk_path)
                document = _Document(path, copy, content)
                self._copies += 1
                await self._lua("return _G.gigachad_open(...)", str(copy.name), path)
                self._documents[path] = document
            if line is not None:
                await self._lua("return _G.gigachad_goto_line(...)", line, column)

    async def set_text(self, content: str, path: str) -> None:
        """Show *content* (now on disk) in the buffer of *path*, keeping cursor and view.

        A modified buffer gets it merged into its unsaved edits.

        Does nothing when that document is not open. Waits for a running `:w` of it. When one stored the buffer meanwhile,
        *content* and the stored text both grew from the old `base`: the buffer takes their merge, stays modified when
        that differs from the stored text, and `base` stays the stored text.
        """
        document = self._documents.get(path)
        if document is None:
            return
        base, saves = document.base, document.saves
        async with document.sync:
            for _ in range(_SET_TEXT_ATTEMPTS):
                state = await self._lua("return _G.gigachad_buffer_state(...)", document.path)
                if document.saves == saves:
                    merge = state["modified"]
                    text = await asyncio.to_thread(merge_theirs, document.base, state["text"], content) if merge else content
                    dirty, agreed = False, content
                else:
                    text = await asyncio.to_thread(merge_theirs, base, state["text"], content)
                    dirty, agreed = text != document.base, document.base
                if await self._lua("return _G.gigachad_set_text(...)", document.path, text, state["tick"], dirty):
                    document.base = agreed
                    return
        raise RuntimeError("Neovim buffer kept changing, the changed file was not applied")

    def is_notebook_text(self, path: str) -> bool:
        """Whether the open document *path* is a notebook's text, whose `rename` takes a virtual name instead of a disk file."""
        document = self._documents.get(path)
        return document is not None and isinstance(document.backing, _NotebookText)

    async def rename(self, old_path: str, path: str, target: Path | None) -> None:
        """Continue the document *old_path* on the renamed file *path*, the buffer keeps its text and modified flag.

        *target* is the file `:w` merges with, or a notebook text's new virtual name (refused when None). Does nothing when
        *old_path* is not open. Waits for a running `:w` of it.
        """
        document = self._documents.get(old_path)
        if document is None:
            return
        if path != old_path and path in self._documents:
            raise RuntimeError(f"{_document_name(path)} is already open")
        backing = document.backing.renamed(path, target)
        async with document.sync:
            await self._lua("return _G.gigachad_rename(...)", old_path, path, str(backing.name))
            if backing.name != document.backing.name:
                document.backing.discard()  # Neovim refuses to `:w` a renamed buffer onto a name that exists
            document.backing = backing
            document.path = path
            del self._documents[old_path]
            self._documents[path] = document

    async def insert_lines(self, text: str) -> None:
        """Insert *text* as lines below the cursor of the shown document, the cursor lands on the first inserted line."""
        await self._lua("return _G.gigachad_insert_lines(...)", text)

    async def insert_frame(self, text: str) -> None:
        """Insert the beamer frame *text* where `frame_insertion` places it for the shown document's cursor, which lands on
        its begin line."""
        for _ in range(_INSERT_ATTEMPTS):
            state = await self._lua("return _G.gigachad_cursor_state()")
            frame = frame_insertion(state["lines"], state["row"], _buffer_lines(text))
            if await self._lua(
                "return _G.gigachad_insert_at(...)", frame.row, frame.lines, frame.cursor_row, state["buffer"], state["tick"]
            ):
                return
        raise RuntimeError("Neovim buffer kept changing, the slide was not inserted")

    async def insert_image(self, path: str, reference: str) -> None:
        """Put the image *path* in the cursor line's first `example-image` placeholder, else insert *reference* below the cursor."""
        for _ in range(_INSERT_ATTEMPTS):
            state = await self._lua("return _G.gigachad_cursor_state()")
            row, buffer, tick = state["row"], state["buffer"], state["tick"]
            filled = fill_placeholder(state["lines"][row - 1], path)
            if filled is None:
                inserted = await self._lua(
                    "return _G.gigachad_insert_at(...)", row, _buffer_lines(reference), row + 1, buffer, tick
                )
            else:
                inserted = await self._lua("return _G.gigachad_set_line(...)", row, filled, buffer, tick)
            if inserted:
                return
        raise RuntimeError("Neovim buffer kept changing, the image was not inserted")

    async def show_error(self, text: str) -> None:
        """Show *text* in Neovim's message area as an error."""
        await self._lua("return _G.gigachad_show_error(...)", text)

    async def _base(self, path: str) -> str:
        """Answer the `base` request: the text Neovim and the stored file last agreed on for document *path*."""
        document = self._documents.get(path)
        if document is None:
            raise RuntimeError(f"{_document_name(path)} is not open")
        return document.base

    async def _save(self, save: SaveDocument, path: str, text: str) -> dict[str, str | bool]:
        """Store *text* of document *path* merged with the edits made elsewhere since its `base`, `merged` tells if they changed it."""
        document = self._documents.get(path)
        if document is None:
            raise RuntimeError(f"{_document_name(path)} is not open")
        base = document.base  # *text* grew from it, a `set_text` holding the lock may still move `base` on
        async with document.sync:
            theirs = await asyncio.to_thread(document.backing.disk_text)
            if theirs is None:
                theirs = document.base  # no disk file: the last `set_text` is the newest text from elsewhere
            merged = text
            if theirs != base:
                try:
                    merged = await asyncio.to_thread(merge_theirs, base, text, theirs)
                except TextMergeError as exc:
                    raise TextMergeError(f"could not merge with the changed file, nothing written ({exc})") from exc
            stored = await save(document.path, merged)
            document.base = stored
            document.saves += 1
        return {"text": stored, "merged": merged != text}

    def _forget(self, path: str) -> None:
        """Drop the document whose buffer Neovim deleted, a private copy goes with it."""
        if (document := self._documents.pop(path, None)) is not None:
            document.backing.discard()

    async def _lua(self, code: str, *args: object) -> object:
        await asyncio.wait_for(self._ready.wait(), _STARTUP_SECONDS * 2)  # a request that beat the save hook waits for it
        assert self._rpc is not None
        return await asyncio.wait_for(self._rpc.request("nvim_exec_lua", code, list(args)), _RPC_SECONDS)

    async def set_background(self, light: bool) -> None:
        """Set 'background' to light or dark now, or once connected. Neovim reloads the colorscheme on a change."""
        self._background = "light" if light else "dark"
        if self._ready.is_set():
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
            set_terminal_size(self._terminal_fd, rows, cols)

    async def wait(self) -> int:
        """Exit code of `nvim`."""
        return await asyncio.to_thread(self._process.wait)

    async def close(self) -> None:
        """Stop `nvim` and delete the private copies, discarding unsaved edits."""
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
