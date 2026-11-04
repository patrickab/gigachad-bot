"""Neovim terminal sessions for logins with host access (`backend.identity.HOST_CAPABILITY`).

`WS /api/neovim/session` runs one `nvim` per connection. Binary frames carry terminal
bytes both ways, text frames carry JSON control messages:

- client -> server: `{"type": "open", "path", "filename", "content", "base_content"?, "ask_keymap"?, "notebook"?, "rows", "cols"}`
  first (the first document, a string `base_content` opens a read-only left diff buffer against the editable document,
  `ask_keymap` maps visual-mode `<leader>a` to a `selection` frame, `notebook` opens a notebook's text, see below), then
  `{"type": "resize", "rows", "cols"}`,
  `{"type": "saved", "id", "content" | "error"}`, `{"type": "open_file", "path"}` (show that file's buffer, loading it from
  disk into a private copy of its own when not open yet, other buffers stay loaded with their unsaved edits and nothing is
  saved), `{"type": "set_content", "content", "path"}` (the file changed on disk, the buffer of *path* takes the text
  without moving the cursor, merged into unsaved edits, after a running `:w`),
  `{"type": "renamed", "from", "path"}` (the file of buffer *from* was renamed, the session continues on the new
  name with its unsaved edits, nothing for a buffer that is not open, a notebook's text needs a valid notebook text *path*),
  `{"type": "insert_lines", "text"}` (lines below the
  cursor of the shown buffer, the cursor lands on the first one), `{"type": "background", "light"}` (the theme behind
  the terminal, Neovim's 'background' follows it), `{"type": "clipboard", "id", "text" | "error"}` (the answer to
  `clipboard_read`: what to paste, linewise when it ends with a newline and charwise otherwise, an empty text pastes
  nothing, an `error` is shown as an error in Neovim and its last copy is pasted instead) and `{"type": "message", "text"}`
  (Neovim shows *text* as an error, for failures that have no request)
- server -> client: `{"type": "save", "id", "path", "content"}` on `:w` of the buffer of file *path*, `{"type": "yank", "content"}`,
  `{"type": "selection", "path", "start_line", "end_line", "text"}` (1-based inclusive lines),
  `{"type": "switched", "path"}` when Neovim itself shows another file's buffer (`:bnext`, not after `open_file`),
  `{"type": "open_failed", "path", "message"}` when `open_file` could not load the file, `{"type": "leave"}` on `<Esc>` in normal
  mode (the browser gives focus back to the app),
  `{"type": "clipboard_read", "id", "path"}` when Neovim pastes from `+` or `*` (`path` names the shown buffer, the browser
  answers with `clipboard`, none within 15 s shows "Clipboard did not answer" and pastes the last copy, so does a closed
  socket), `{"type": "run_cell", "path", "line"}` after `<S-CR>` or `<C-CR>` in a notebook's text saved it (the 1-based cursor
  line in the saved text) and `{"type": "exit", "code"}` when `nvim` quits

`nvim` starts in the mounted root holding a file-root document, in the home directory otherwise.

The browser persists `:w` through the document's own API, so every save keeps that
API's validation and side effects. A file-root file changed on disk since Neovim last
agreed with it saves merged (`NeovimSession`). Disconnecting discards unsaved edits.

With `"notebook": true` the document is a notebook's percent text (`# %%` before each cell) and `path` is
`<notebook>.ipynb<ext>`: absolute, a short alphanumeric extension, the notebook a readable file under the file roots, else the
socket closes with 1003. The buffer is named beside the validated notebook, so the user's LSP finds its project and `.venv`,
and nothing is ever read from or written to that name (no swap or undo file, every write goes to `save`). `<C-CR>` saves the
text and then sends `run_cell`, `<S-CR>` also moves the cursor to the next cell, appending one after the last. Shift+Enter and
Ctrl+Enter reach Neovim as CSI-u `ESC [ 13 ; 2 u` and `ESC [ 13 ; 5 u`.

Neovim answers no request while it waits at a prompt, so the session turns `more` off and lets messages time out
(`messagesopt`) after the user's config ran: what the config itself prints at startup is out of reach.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, WebSocket
from pydantic import BaseModel, Field, ValidationError
from starlette.websockets import WebSocketState

from backend.identity import RequestIdentity, get_request_identity
from backend.routes.deps import get_file_roots
from backend.websocket_origin import POLICY_VIOLATION, is_allowed_websocket
from lib.file_roots import FileRoots
from lib.neovim_session import ClipboardError, NeovimSession, Selection, is_neovim_installed

if TYPE_CHECKING:
    from collections.abc import Coroutine

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/neovim", tags=["neovim"])

_SAVE_TIMEOUT_SECONDS = 60
_UNSUPPORTED_DATA = 1003


class OpenMessage(BaseModel):
    """`path` is the document's app path: absolute for file-root files, a logical key otherwise, a notebook's text with `notebook`."""

    path: str
    filename: str
    content: str
    base_content: str | None = None
    ask_keymap: bool = False
    notebook: bool = False
    rows: int = Field(ge=1, le=1000)
    cols: int = Field(ge=1, le=1000)


_NOTEBOOK_TEXT = re.compile(r"(?P<notebook>.+\.ipynb)\.[A-Za-z0-9]{1,8}")


def working_directory(roots: FileRoots, path: str) -> Path:
    """The mounted root holding a file-root document, home for everything else."""
    if Path(path).is_absolute():
        with contextlib.suppress(ValueError):
            return roots.root_of(path)
    return Path.home()


def disk_file(roots: FileRoots, path: str) -> Path | None:
    """The readable file behind an absolute document *path*, None for logical keys and refused paths."""
    if not Path(path).is_absolute():
        return None
    try:
        return roots.resolve(path)
    except (OSError, ValueError):
        return None


def notebook_buffer(roots: FileRoots, path: str) -> Path | None:
    """The virtual buffer name of the notebook text *path* (`<notebook>.ipynb<ext>`) beside the validated notebook, None if invalid."""
    match = _NOTEBOOK_TEXT.fullmatch(path)
    if match is None:
        return None
    notebook = disk_file(roots, match["notebook"])
    return None if notebook is None else notebook.with_name(notebook.name + path[match.end("notebook") :])


class SaveFailed(RuntimeError):
    """The browser could not persist a `:w`."""


class _BrowserSaves:
    """Forward `:w` to the browser and wait for what it stored, ask it for its clipboard, and tell it what happens in Neovim."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._pending: dict[int, asyncio.Future[dict]] = {}
        self._next_id = 0
        # What happens in Neovim, sent by `relay_notices` in the order it happened (one task per frame could reorder them)
        self._notices: asyncio.Queue[dict] = asyncio.Queue()

    async def _ask(self, frame: dict, seconds: float | None = None) -> dict:
        """Send *frame* under a new id and wait for the browser's reply to it."""
        self._next_id += 1
        request_id = self._next_id
        reply = self._pending[request_id] = asyncio.get_running_loop().create_future()
        try:
            await self._websocket.send_json({**frame, "id": request_id})
            return await asyncio.wait_for(reply, seconds)
        finally:
            self._pending.pop(request_id, None)

    async def save(self, path: str, text: str) -> str:
        result = await self._ask({"type": "save", "path": path, "content": text}, _SAVE_TIMEOUT_SECONDS)
        if result.get("error"):
            raise SaveFailed(str(result["error"]))
        return str(result.get("content", text))

    async def read_clipboard(self, path: str) -> str:
        """The text the browser answers for a paste into *path*, a `ClipboardError` when it answers an error."""
        reply = await self._ask({"type": "clipboard_read", "path": path})
        if reply.get("error"):
            raise ClipboardError(str(reply["error"]))
        text = reply.get("text")
        return text if isinstance(text, str) else ""

    def notice(self, frame: dict) -> None:
        """Queue *frame* for the browser, behind the notices queued before it."""
        self._notices.put_nowait(frame)

    async def relay_notices(self) -> None:
        while True:
            await self._websocket.send_json(await self._notices.get())

    def yank(self, text: str) -> None:
        self.notice({"type": "yank", "content": text})

    def selection(self, selection: Selection) -> None:
        self.notice({"type": "selection", **selection})

    def switched(self, path: str) -> None:
        self.notice({"type": "switched", "path": path})

    def leave(self) -> None:
        self.notice({"type": "leave"})

    def run_cell(self, path: str, line: int) -> None:
        self.notice({"type": "run_cell", "path": path, "line": line})

    def open_failed(self, path: str, message: str) -> None:
        self.notice({"type": "open_failed", "path": path, "message": message})

    def resolve(self, reply: dict) -> None:
        """The browser's `saved` or `clipboard` answer to the question with the same id."""
        pending = self._pending.get(reply.get("id"))
        if pending is not None and not pending.done():
            pending.set_result(reply)


async def _relay_output(websocket: WebSocket, session: NeovimSession) -> None:
    while (output := await session.read_output()) is not None:
        await websocket.send_bytes(output)
    await websocket.send_json({"type": "exit", "code": await session.wait()})


async def _guarded(what: str, work: Coroutine[None, None, None]) -> None:
    """Await *work*, logging instead of raising when Neovim cannot *what*."""
    try:
        await work
    except (RuntimeError, ConnectionError, TimeoutError) as exc:
        log.warning("Neovim could not %s: %s", what, exc)


async def _open_file(session: NeovimSession, saves: _BrowserSaves, roots: FileRoots, path: str) -> None:
    try:
        await session.open_file(path, disk_file(roots, path))
    except (RuntimeError, ConnectionError, TimeoutError) as exc:
        log.warning("Neovim could not open %s: %s", path, exc)
        saves.open_failed(path, str(exc))


async def _relay_input(websocket: WebSocket, session: NeovimSession, saves: _BrowserSaves, roots: FileRoots) -> None:
    # `set_text`, `rename` and the others wait for a running `:w`, whose answer arrives on this socket, or for each other
    # in Neovim, so they run beside the relay
    updates: set[asyncio.Task] = set()

    def beside_relay(work: Coroutine[None, None, None]) -> None:
        update = asyncio.create_task(work)
        updates.add(update)
        update.add_done_callback(updates.discard)

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                session.write_input(message["bytes"])
                continue
            control = json.loads(message.get("text") or "{}")
            kind = control.get("type")
            if kind == "resize":
                session.resize(int(control["rows"]), int(control["cols"]))
            elif kind in ("saved", "clipboard"):
                saves.resolve(control)
            elif kind == "set_content" and isinstance(control.get("content"), str) and isinstance(control.get("path"), str):
                beside_relay(_guarded("take the changed file", session.set_text(control["content"], control["path"])))
            elif kind == "open_file" and isinstance(control.get("path"), str):
                beside_relay(_open_file(session, saves, roots, control["path"]))
            elif kind == "insert_lines" and isinstance(control.get("text"), str):
                beside_relay(_guarded("take the inserted lines", session.insert_lines(control["text"])))
            elif kind == "renamed" and all(isinstance(control.get(key), str) for key in ("from", "path")):
                path = control["path"]
                rename = session.rename(control["from"], path, disk_file(roots, path), notebook_buffer(roots, path))
                beside_relay(_guarded("take the renamed file", rename))
            elif kind == "background" and isinstance(control.get("light"), bool):
                await _guarded("take the theme", session.set_background(control["light"]))
            elif kind == "message" and isinstance(control.get("text"), str):
                beside_relay(_guarded("show the message", session.show_error(control["text"])))
    finally:
        for update in updates:
            update.cancel()


async def _run(websocket: WebSocket, session: NeovimSession, roots: FileRoots, *, ask_keymap: bool) -> None:
    """Relay until `nvim` exits, the browser leaves, or the save hook cannot be installed."""
    saves = _BrowserSaves(websocket)
    relays = [
        asyncio.create_task(_relay_output(websocket, session)),
        asyncio.create_task(_relay_input(websocket, session, saves, roots)),
        asyncio.create_task(saves.relay_notices()),
    ]
    # Input flows while connecting, so a startup prompt from the user's config can be answered.
    connecting = asyncio.create_task(
        session.connect(
            saves.save,
            saves.yank,
            saves.selection if ask_keymap else None,
            saves.switched,
            saves.read_clipboard,
            saves.run_cell,
            on_leave=saves.leave,
        )
    )
    try:
        done, _ = await asyncio.wait([*relays, connecting], return_when=asyncio.FIRST_COMPLETED)
        if connecting in done and connecting.exception() is not None:
            log.warning("Neovim save hook failed: %s", connecting.exception())
            return
        await asyncio.wait(relays, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (*relays, connecting):
            task.cancel()
        await asyncio.gather(*relays, connecting, return_exceptions=True)


@router.websocket("/session")
async def neovim_session(
    websocket: WebSocket,
    identity: RequestIdentity = Depends(get_request_identity),
    roots: FileRoots = Depends(get_file_roots),
) -> None:
    if not is_allowed_websocket(websocket, identity) or not is_neovim_installed():
        await websocket.close(code=POLICY_VIOLATION)
        return
    await websocket.accept()
    try:
        opening = OpenMessage.model_validate(await websocket.receive_json())
    except (ValidationError, ValueError):
        await websocket.close(code=_UNSUPPORTED_DATA)
        return
    notebook = notebook_buffer(roots, opening.path) if opening.notebook else None
    if opening.notebook and notebook is None:
        await websocket.close(code=_UNSUPPORTED_DATA)
        return
    session = NeovimSession.spawn(
        opening.filename,
        opening.content,
        base_content=opening.base_content,
        rows=opening.rows,
        cols=opening.cols,
        cwd=working_directory(roots, opening.path),
        disk_path=None if opening.notebook else disk_file(roots, opening.path),
        path=opening.path,
        notebook=notebook,
    )
    try:
        await _run(websocket, session, roots, ask_keymap=opening.ask_keymap)
    finally:
        await session.close()
        if websocket.client_state == WebSocketState.CONNECTED:
            with contextlib.suppress(RuntimeError):
                await websocket.close()
