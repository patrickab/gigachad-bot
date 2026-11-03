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
- With an `on_selection` callback, visual-mode `<leader>a` reports the selected lines and their document. In a notebook
  buffer the path is the buffer's name (`.ipynb.py`, `.ipynb.md`) and the lines count from the active cell's first line
- `on_switched` hears of a document Neovim itself put on screen (`:bnext`, `:b`), not of one `open_file` showed
- `set_text` takes edits made to the file elsewhere without moving the cursor: an unmodified buffer shows them,
  a modified one merges them and stays modified. It works on hidden buffers too
- `:w` of a document with a `disk_path` whose text left `base` saves the buffer merged with the disk text, and fails
  without writing when that text is not UTF-8 or the merge fails. `set_text` waits for a running `:w` of its document
- `rename` moves a document to the file's new path: the buffer is renamed keeping text, cursor and unsaved state,
  and `:w` merges with the new `disk_path`
- `insert_lines` puts lines below the cursor of the shown document
- `show_error` shows a message in Neovim as an error
- With a `read_clipboard` callback, `+` and `*` are the clipboard it reads (the browser's, not the machine's Neovim runs on)
  and replace the provider of the user's config: a paste asks `read_clipboard(path)` with the path of the shown document,
  which answers the text to paste (linewise when it ends with a newline) or raises `ClipboardError`. Neovim shows its
  message and pastes its last copy instead, and so after 15 s without an answer ("Clipboard did not answer"). A copy
  stays in Neovim's registers, the yank reaches the caller through `on_yank`. Reads run beside the other requests, so a
  paste waiting for the browser never holds up `set_text`, `:w` or the others
- `set_background` sets 'background' to the browser theme's, so the colorscheme picks its light or dark palette
- A notebook session (`cell_session`) starts on an empty placeholder and holds the notebook in TWO buffers: `set_cells` shows all
  code cells as one script (a `# %%` marker line before each cell, so an LSP sees every cell) and all markdown cells as one
  markdown document (`<!-- %% -->` before each cell, made with the first markdown cell). With an absolute `notebook_path` the
  buffers' names are virtual, `<notebook_path><extension of the set_cells filename>` and `<notebook_path>.md` beside the notebook
  (`.ipynb.py` for a Python notebook), never created on disk (every write is intercepted, no swap or undo file) so an LSP finds
  the project and `.venv` from the notebook's directory. Without it they are private copies named like the filename and its
  stem with `.md` (they pick the filetype, never written back). Neovim's window shows one of them, pinned to its active cell,
  `focus_cell` activates one from outside (and puts its buffer in the window). Later `set_cells` replace only the text of the
  cells that differ (no `cell_edit` comes back for it) or rebuild a buffer when its ids or their order differ (the filename is
  not read again, a rebuild drops the undo history). Cells are found by extmarks on their marker lines, edits that alter, remove
  or add a marker line are undone, lines put above the first marker join the first cell. With an `on_cell` callback both
  buffers report `{"type": "cell_edit", "id", "text"}` (full text of a cell after typing, coalesced over 40 ms, only when it
  differs from what the caller knows), `{"type": "cell_focus", "id"}` (the cursor moved into another cell of the buffer, never
  past a cell of the other kind), `{"type": "cell_exit", "id", "direction": "up" | "down"}` (an up or down key on the first or
  last line of a cell whose neighbor in the notebook that way is of the other kind), `{"type": "cell_run", "id", "text",
  "advance"}` (`<S-CR>` advances, `<C-CR>` does not), `{"type": "cell_leave", "id"}` (`<Esc>` in normal mode in a markdown cell),
  `{"type": "save_notebook"}` (`:w`), always after the `cell_edit`s they follow, `{"type": "cell_height", "id", "rows"}` (the
  screen rows the active cell takes in the window with wrapped lines, folds and virtual lines, sent when a cell is pinned and
  whenever they change, never twice the same), and for code cells only `{"type": "cell_render", "id", "styles", "lines", "base",
  "rows", "height", "wrap"?}`: the cell as Neovim draws it (see `lib.neovim_render`), its screen rows and 'wrap' column included,
  150 ms after the cell, its diagnostics, its semantic tokens (or the tokens its last render lacked), the colorscheme, the number
  column options, 'wrap', the width or (with absolute line numbers) its first line changed, never the same one twice in a row. The
  active cell gets only its first one, a cell that was left is rendered before its `cell_focus`, the code cell left for a
  markdown cell before the markdown buffer is shown (renders wait while it is). `{"type": "cell_render", "id"}`
  without lines takes a render back (it failed, or the buffer grew past 4000 lines: every cell), the caller shows its own editor
  there. A `rename` of the notebook's file gives the virtual buffers their names beside the new file
- Every session turns `more` off and lets messages time out (`messagesopt`) once the user's config ran: Neovim answers no RPC
  request while it waits at a hit-enter prompt
- Closing a session discards unsaved edits
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from lib.neovim_render import Rendering, parse_tohtml
from lib.neovim_rpc import SETUP_LUA, NeovimRpc, NeovimRpcError
from lib.process_owner import owned_environment
from lib.pty_process import claim_controlling_terminal, set_terminal_size, terminal_environment
from lib.text_merge import TextMergeError, merge_theirs

SaveDocument = Callable[[str, str], Awaitable[str]]
YankText = Callable[[str], None]
Selection = dict[str, int | str]
SelectLines = Callable[[Selection], None]
SwitchDocument = Callable[[str], None]
ReadClipboard = Callable[[str], Awaitable[str]]
CellEvent = dict[str, object]
CellEvents = Callable[[CellEvent], None]

# The line that leads every cell in the notebook's code buffer, and in its markdown buffer
NOTEBOOK_MARKER = "# %%"
MARKDOWN_MARKER = "<!-- %% -->"
_MARKERS = {"code": NOTEBOOK_MARKER, "markdown": MARKDOWN_MARKER}

_STARTUP_SECONDS = 15
_RPC_SECONDS = 10
_SHUTDOWN_SECONDS = 2
_READ_CHUNK = 65536
# `set_text` retries while keystrokes change the buffer between reading and merging it.
_SET_TEXT_ATTEMPTS = 5
# How long a paste waits for the browser, which may be asking the user for permission first.
_CLIPBOARD_SECONDS = 15
_EXTENSION = re.compile(r"\.[A-Za-z0-9_+-]{1,16}")


def is_neovim_installed() -> bool:
    """`nvim` on PATH and a config (under `XDG_CONFIG_HOME`) to load."""
    config_dir = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "nvim"
    return shutil.which("nvim") is not None and config_dir.is_dir()


def _document_name(filename: str) -> str:
    """Basename for the private copy; keeps the extension so filetype detection works."""
    name = Path(filename).name
    return name if name not in {"", ".", ".."} else "document.md"


def _private_copy(directory: Path, filename: str, content: str) -> Path:
    """Write *content* to a new *directory* under the basename of *filename*."""
    directory.mkdir()
    document = directory / _document_name(filename)
    document.write_text(content, encoding="utf-8", newline="")
    return document


def _notebook_script(cells: Sequence[Mapping[str, str]], marker: str) -> str:
    """The text of a notebook buffer for *cells*: a *marker* line before the lines of each cell."""
    lines = [line for cell in cells for line in (marker, *cell["text"].split("\n"))]
    return "\n".join(lines) + "\n" if lines else ""


def _book_filename(kind: str, filename: str) -> str:
    """The file name the buffer of the *kind* cells is named like: the code filename, its stem with `.md` for markdown."""
    return filename if kind == "code" else f"{Path(_document_name(filename)).stem}.md"


def _virtual_name(notebook: Path, filename: str) -> Path:
    """The name of a notebook buffer beside *notebook*: its path plus the extension of *filename* (`.py` when that is none)."""
    suffix = Path(_document_name(filename)).suffix
    return Path(f"{notebook}{suffix if _EXTENSION.fullmatch(suffix) else '.py'}")


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


@dataclass
class _Document:
    """A document of the session: its private copy and what Neovim and the stored file last agreed on."""

    path: str
    copy: Path
    base: str
    disk_path: Path | None
    saves: int = 0
    # Held by `set_text` and by `:w` until `save` answered, so neither runs in the middle of the other
    sync: asyncio.Lock = field(default_factory=asyncio.Lock)

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
        cell_session: bool = False,
        notebook_path: Path | None = None,
    ) -> None:
        self._documents = {first.path: first}
        # The buffer names of the notebook by kind of cell (`code`, `markdown`): virtual names, or private copies
        self._books: dict[str, Path] = {}
        # The last `cell_render` the caller got per cell, an equal one is not sent again
        self._renderings: dict[str, Rendering] = {}
        # The notebook's file when its buffer is virtual: named `<notebook_path><extension>` beside the notebook, never on disk
        self._notebook_path = notebook_path if cell_session and notebook_path is not None and notebook_path.is_absolute() else None
        self._copies = 1
        # Serializes `open_file` and `set_cells`, which look a document up and then add it
        self._opening = asyncio.Lock()
        self._baseline = baseline
        # The placeholder document of a notebook session, until the notebook buffer replaces it
        self._placeholder = first.path if cell_session else None
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
        cell_session: bool = False,
        notebook_path: Path | None = None,
    ) -> NeovimSession:
        """Start `nvim` in *cwd* on a private copy of *content*, diffed against a read-only *base_content* when given.

        *path* names the document in `save`, `selection` and the other calls, default *filename*. A *cell_session* takes
        *filename*, *content* and *path* as its placeholder document, the first `set_cells` wipes it. With an absolute
        *notebook_path* the buffer is named `<notebook_path><extension>` (virtual: nothing is created there), so an LSP finds the
        project and its `.venv` from the notebook's directory, else it is a private copy named like the `set_cells` filename.
        *disk_path* is the file *content* was read from, `:w` merges its later changes. Call from a running event loop.
        """
        workdir = Path(tempfile.mkdtemp(prefix="gigachad-nvim-"))
        first = _Document(path or filename, _private_copy(workdir / "0", filename, content), content, disk_path)
        baseline: Path | None = None
        if base_content is not None:
            (workdir / "base").mkdir()
            baseline = workdir / "base" / _document_name(filename)
            baseline.write_text(base_content, encoding="utf-8", newline="")
        terminal_fd, child_fd = os.openpty()
        set_terminal_size(terminal_fd, rows, cols)
        try:
            process = subprocess.Popen(
                ["nvim", "-n", "--listen", str(workdir / "rpc.sock"), str(first.copy)],
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
        return cls(process, terminal_fd, workdir, first, baseline=baseline, cell_session=cell_session, notebook_path=notebook_path)

    async def connect(
        self,
        save: SaveDocument,
        on_yank: YankText,
        on_selection: SelectLines | None = None,
        on_switched: SwitchDocument | None = None,
        read_clipboard: ReadClipboard | None = None,
        on_cell: CellEvents | None = None,
        on_leave: Callable[[], None] | None = None,
    ) -> None:
        """Route `:w` of every document buffer to *save* (its return value replaces the buffer if it differs) and yanks to *on_yank*.

        `:w` sends the buffer merged with the disk text when the file at the document's `disk_path` changed since its `base`
        (or with the text of a `set_text` that ran while the save waited), and holds off that document's `set_text` until
        *save* returned.

        Giving *on_selection* also installs the visual-mode `<leader>a` mapping, it receives `{"path", "start_line", "end_line",
        "text"}` (in a notebook buffer the buffer's name and the lines of the active cell, 1 is its first). *on_switched* receives
        the path of the document that Neovim itself put on screen, not of one `open_file` showed.

        Giving *read_clipboard* makes it the clipboard of `+` and `*`: a paste calls it with the path of the shown document and
        pastes the text it returns (linewise when it ends with a newline). When it raises a `ClipboardError`, or gives no answer
        within 15 s ("Clipboard did not answer"), Neovim shows the message and pastes its last copy instead.
        It runs beside every other request, Neovim answers those while it waits.

        *on_cell* receives the events of the notebook buffer (`set_cells`) in the order Neovim sent them. *on_leave* hears `<Esc>` in
        normal mode, after the user's own normal-mode `<Esc>` mapping ran.
        """

        async def handle_request(method: str, params: list) -> object:
            if method == "save":
                return await self._save(save, params[0], params[1])
            if method == "paste" and read_clipboard is not None:
                return await _read_clipboard(read_clipboard, params[0])
            raise NeovimRpcError(f"Unknown request {method!r}")

        def handle_notification(method: str, params: list) -> None:
            if method == "yank":
                on_yank(params[0])
            elif method == "selection" and on_selection is not None:
                on_selection(params[0])
            elif method == "switched" and on_switched is not None:
                on_switched(params[0])
            elif method == "leave" and on_leave is not None:
                on_leave()
            elif on_cell is not None and method == "cell_edit":
                on_cell({"type": method, "id": params[0], "text": params[1]})
            elif on_cell is not None and method == "cell_run":
                on_cell({"type": method, "id": params[0], "text": params[1], "advance": params[2]})
            elif on_cell is not None and method == "cell_exit":
                on_cell({"type": method, "id": params[0], "direction": params[1]})
            elif on_cell is not None and method in ("cell_focus", "cell_leave"):
                on_cell({"type": method, "id": params[0]})
            elif on_cell is not None and method == "cell_height":
                on_cell({"type": method, "id": params[0], "rows": params[1]})
            elif on_cell is not None and method == "save_notebook":
                on_cell({"type": method})
            elif on_cell is not None and method == "cell_render":
                self._render(params, on_cell)
            elif on_cell is not None and method == "cell_render_drop":
                self._drop(params, on_cell)

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
        base = str(self._baseline) if self._baseline is not None else None
        arguments = [
            channel,
            str(first.copy),
            first.path,
            base,
            on_selection is not None,
            read_clipboard is not None,
            self._placeholder is not None,
        ]
        setup = self._rpc.request("nvim_exec_lua", SETUP_LUA, arguments)
        await asyncio.wait_for(setup, _STARTUP_SECONDS)
        self._ready.set()
        if self._background is not None:
            await self._apply_background()

    async def open_file(self, path: str, disk_path: Path | None) -> None:
        """Show the document *path*, first loading *disk_path* into a new private copy when it is not open yet.

        The document shown before stays loaded with its unsaved edits, nothing is saved. Raises `RuntimeError` when there is
        no readable *disk_path* or it is not UTF-8 text.
        """
        async with self._opening:
            if path in self._documents and await self._lua("return _G.gigachad_show(...)", path):
                return
            self._forget(path)  # Neovim lost the buffer (`:bdelete`), the copy is stale
            if disk_path is None:
                raise RuntimeError(f"{_document_name(path)} is not a readable file")
            content = await asyncio.to_thread(_read_for_opening, disk_path)
            document = _Document(path, _private_copy(self._workdir / str(self._copies), path, content), content, disk_path)
            self._copies += 1
            await self._lua("return _G.gigachad_open(...)", str(document.copy), path)
            self._documents[path] = document

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

    async def rename(self, old_path: str, path: str, disk_path: Path | None) -> None:
        """Continue the document *old_path* on the renamed file *path*: the buffer takes its name keeping its text and modified flag.

        `:w` merges what is read from *disk_path*. When *old_path* is the notebook of a virtual notebook buffer, that buffer is
        named beside *disk_path* (the validated file, nothing happens without one).

        Does nothing when *old_path* is not open. Waits for a running `:w` of it. The private copy is dropped: Neovim refuses
        to `:w` a renamed buffer onto a name that exists.
        """
        if self._notebook_path is not None and Path(old_path).is_absolute() and Path(old_path).resolve() == self._notebook_path:
            await self._rename_notebook(disk_path)
            return
        document = self._documents.get(old_path)
        if document is None:
            return
        if path != old_path and path in self._documents:
            raise RuntimeError(f"{_document_name(path)} is already open")
        renamed = document.copy.with_name(_document_name(path))
        async with document.sync:
            await self._lua("return _G.gigachad_rename(...)", old_path, path, str(renamed))
            if renamed != document.copy:
                document.copy.unlink(missing_ok=True)
                document.copy = renamed
            document.disk_path = disk_path
            document.path = path
            del self._documents[old_path]
            self._documents[path] = document

    async def _rename_notebook(self, disk_path: Path | None) -> None:
        """The notebook moved to *disk_path*: the virtual buffers take their names beside it, keeping their extensions."""
        if disk_path is None:
            return
        async with self._opening:
            self._notebook_path = disk_path
            if self._books:
                self._books = {kind: _virtual_name(disk_path, name.name) for kind, name in self._books.items()}
                await self._lua("return _G.gigachad_rename_notebook(...)", {kind: str(name) for kind, name in self._books.items()})

    async def insert_lines(self, text: str) -> None:
        """Insert *text* as lines below the cursor of the shown document, the cursor lands on the first inserted line."""
        await self._lua("return _G.gigachad_insert_lines(...)", text)

    async def show_error(self, text: str) -> None:
        """Show *text* in Neovim's message area as an error."""
        await self._lua("return _G.gigachad_show_error(...)", text)

    async def set_cells(self, filename: str, cells: Sequence[Mapping[str, str]]) -> None:
        """Make the notebook buffers hold *cells*, `{"id", "text", "kind"}` for every code and markdown cell in notebook order.

        `kind` is `code` (also when it is missing) or `markdown`. The code cells are one buffer (a script, `# %%` before each
        cell), the markdown cells another (`<!-- %% -->` before each cell), made when the first markdown cell comes. The first
        call (or one after Neovim lost a buffer) creates it, the code buffer replaces the placeholder. The names come from
        *filename*, so filetype detection and the user's plugins work: beside the notebook as `<notebook_path><extension of
        filename>` and `<notebook_path>.md` (virtual, `.py` for an extension that is none), else private copies named like
        *filename* and its stem with `.md`. Later calls do not read *filename* again. They replace the text of the cells that
        differ and rebuild a buffer when its ids or their order differ (the undo history is dropped then), the cursor stays in its
        cell where possible. The ids must be unique across both kinds. No `cell_edit` comes back for what was given. A cell whose
        text still is what `cell_edit` last reported keeps the text typed since.
        """
        async with self._opening:
            books: dict[str, dict[str, object]] = {}
            for kind, marker in _MARKERS.items():
                mine = [{"id": cell["id"], "text": cell["text"]} for cell in cells if cell.get("kind", "code") == kind]
                if kind != "code" and not mine and kind not in self._books:
                    continue
                book: dict[str, object] = {"marker": marker, "cells": mine}
                if kind not in self._books or not await self._lua("return _G.gigachad_has_book(...)", kind):
                    stale = self._books.get(kind)
                    if stale is not None and self._notebook_path is None:
                        stale.unlink(missing_ok=True)  # Neovim lost the buffer (`:bwipeout`), the copy is stale
                    name, script = _book_filename(kind, filename), _notebook_script(mine, marker)
                    if self._notebook_path is not None:
                        self._books[kind] = _virtual_name(self._notebook_path, name)
                        book["script"] = script
                    else:
                        self._books[kind] = _private_copy(self._workdir / str(self._copies), name, script)
                        self._copies += 1
                    book["path"] = str(self._books[kind])
                    if kind == "code":
                        self._renderings.clear()  # a new buffer: nothing it rendered before is on the caller's side any more
                books[kind] = book
            await self._lua("return _G.gigachad_set_cells(...)", books, [cell["id"] for cell in cells])
            if self._placeholder is not None:
                self._forget(self._placeholder)
                self._placeholder = None
            code = {cell["id"] for cell in cells if cell.get("kind", "code") == "code"}
            self._renderings = {cell_id: rendering for cell_id, rendering in self._renderings.items() if cell_id in code}

    async def focus_cell(self, cell_id: str, cursor: str | int) -> None:
        """Pin the window to the cell *cell_id*, the cursor on its `first` or `last` line, or `keep` where it is when inside it.

        An integer *cursor* is the line of the cell to put the cursor on (0 is the first, past the end is the last). Does nothing
        for an unknown cell. No `cell_focus` comes back.
        """
        where = {"offset": cursor, "col": 0} if isinstance(cursor, int) else cursor
        await self._lua("return _G.gigachad_focus_cell(...)", cell_id, where)

    def _render(self, params: list, on_cell: CellEvents) -> None:
        """Hand *on_cell* a cell's render (`cell_render`: HTML, gutter, base, rows, height, wrap, see `lib.neovim_render`) if new."""
        if len(params) != 7 or not isinstance(params[0], str) or not isinstance(params[1], str):
            return
        rendering = parse_tohtml(*params[1:])
        if self._renderings.get(params[0]) != rendering:
            self._renderings[params[0]] = rendering
            on_cell({"type": "cell_render", "id": params[0], **rendering})

    def _drop(self, params: list, on_cell: CellEvents) -> None:
        """Take back the renders Neovim dropped (`cell_render_drop`, ids): a `cell_render` without lines for each one the caller has."""
        ids = params[0] if len(params) == 1 and isinstance(params[0], list) else []
        for cell_id in ids:
            if isinstance(cell_id, str) and self._renderings.pop(cell_id, None) is not None:
                on_cell({"type": "cell_render", "id": cell_id})

    async def _save(self, save: SaveDocument, path: str, text: str) -> dict[str, str | bool]:
        """Store *text* of document *path* merged with the edits made elsewhere since its `base`, `merged` tells if they changed it."""
        document = self._documents.get(path)
        if document is None:
            raise RuntimeError(f"{_document_name(path)} is not open")
        base = document.base  # *text* grew from it, a `set_text` holding the lock may still move `base` on
        async with document.sync:
            theirs = await asyncio.to_thread(document.disk_text)
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
        """Drop the document whose buffer Neovim deleted, its private copy goes with it."""
        if (document := self._documents.pop(path, None)) is not None:
            document.copy.unlink(missing_ok=True)

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
