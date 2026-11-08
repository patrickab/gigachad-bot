"""Neovim terminal sessions for logins with host access (`backend.identity.HOST_CAPABILITY`).

`WS /api/neovim/session` runs one `nvim` per connection, in the mounted root holding a file-root document, else in the home
directory. Binary frames carry terminal bytes both ways, text frames carry JSON control messages, `type` names each:

- client -> server: `open` first (`OpenMessage`), then `resize`, `saved` and `clipboard` (answers by `id`), `set_content`,
  `open_file`, `renamed`, `insert_lines`, `insert_frame`, `insert_image`, `background` and `message` (the `_Control` models
  below). A frame that fits none of them is ignored
- server -> client: `save {id, path, content}` on `:w`, `clipboard_read {id, path}` when Neovim pastes from `+` or `*` (no
  answer within 15 s, or a closed socket, shows "Clipboard did not answer" and pastes the last copy), `yank {content}`,
  `selection {path, start_line, end_line, text}` (visual-mode `<leader>a` with `ask_keymap`), `switched {path}` (Neovim
  itself showed another file's buffer, not after `open_file`), `open_failed {path, message}`, `leave` (normal-mode `<Esc>`,
  the browser gives focus back to the app), `run_cell {path, line}`, `forward_search {path, line}` (normal-mode `<leader>v`
  in a `.tex` file) and `exit {code}` when `nvim` quits. Lines are 1-based

The browser persists `:w` through the document's own API, so every save keeps that API's validation and side effects. A
file-root file changed on disk since Neovim last agreed with it saves merged (`NeovimSession`). Disconnecting discards unsaved
edits.

With `"notebook": true` the document is a notebook's percent text (`# %%` before each cell) and `path` is
`<notebook>.ipynb<ext>`: absolute, a short alphanumeric extension, the notebook a readable file under the file roots, else the
socket closes with 1003. The buffer is named beside the notebook, so the user's LSP finds its project and `.venv`, and nothing
is read from or written to that name. `<C-CR>` saves the text and then sends `run_cell` with the cursor line, `<S-CR>` also
moves the cursor to the next cell, appending one after the last. Shift+Enter and Ctrl+Enter reach Neovim as CSI-u
`ESC [ 13 ; 2 u` and `ESC [ 13 ; 5 u`.

Neovim answers no request while it waits at a prompt, so the session turns `more` off and lets messages time out
(`messagesopt`) after the user's config ran: what the config itself prints at startup is out of reach.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, Depends, WebSocket
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator
from starlette.websockets import WebSocketState

from backend.identity import RequestIdentity, get_request_identity
from backend.routes.deps import get_file_roots
from backend.websocket_origin import POLICY_VIOLATION, is_allowed_websocket
from lib.file_roots import FileRoots
from lib.neovim_session import ClipboardError, NeovimSession, is_neovim_installed

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/neovim", tags=["neovim"])

_SAVE_TIMEOUT_SECONDS = 60
_UNSUPPORTED_DATA = 1003


class OpenMessage(BaseModel):
    """The first frame, the document to edit.

    `path` is the document's app path: absolute for file-root files, a logical key otherwise, a notebook's text with `notebook`.
    A string `base_content` opens a read-only left diff against it, `ask_keymap` maps visual-mode `<leader>a` to `selection`.
    """

    path: str
    filename: str
    content: str
    base_content: str | None = None
    ask_keymap: bool = False
    notebook: bool = False
    rows: int = Field(ge=1, le=1000)
    cols: int = Field(ge=1, le=1000)


class _Control(BaseModel):
    """A control message after `open`, checked strictly: one that fits no model is ignored."""

    model_config = ConfigDict(strict=True)


class ResizeMessage(_Control):
    type: Literal["resize"]
    rows: int
    cols: int


class ReplyMessage(_Control):
    """The answer to question `id`: `saved` with `content` or `error`, `clipboard` with `text` or `error`.

    A clipboard text ending in a newline pastes linewise, an empty one pastes nothing, an `error` is shown in Neovim, which
    pastes its last copy instead.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["saved", "clipboard"]
    id: int


class SetContentMessage(_Control):
    """File *path* changed on disk: its buffer takes `content` keeping the cursor, merged into unsaved edits, after a running `:w`."""

    type: Literal["set_content"]
    content: str
    path: str


class OpenFileMessage(_Control):
    """Show the file *path*, loaded into a private copy of its own when not open yet, nothing is saved.

    A positive `line` then puts the cursor on that line at the 0-based byte `column` (both clamped), centred.
    """

    type: Literal["open_file"]
    path: str
    line: int | None = None
    column: int = 0

    @field_validator("line", mode="before")
    @classmethod
    def _positive_line(cls, line: object) -> int | None:
        return line if type(line) is int and line > 0 else None

    @field_validator("column", mode="before")
    @classmethod
    def _positive_column(cls, column: object) -> int:
        return column if type(column) is int and column > 0 else 0


class RenamedMessage(_Control):
    """The file of buffer `from` is now *path*, the buffer keeps its unsaved edits. A notebook's text needs a notebook text *path*."""

    type: Literal["renamed"]
    old_path: str = Field(alias="from")
    path: str


class InsertLinesMessage(_Control):
    """Put the lines of `text` below the cursor of the shown buffer, the cursor lands on the first one."""

    type: Literal["insert_lines"]
    text: str


class InsertFrameMessage(_Control):
    """Put the beamer frame `text` after the frame enclosing the cursor, else below it, never after the end of the document."""

    type: Literal["insert_frame"]
    text: str


class InsertImageMessage(_Control):
    """Put *path* in the cursor line's first `example-image` placeholder, else the lines of `reference` below the cursor."""

    type: Literal["insert_image"]
    path: str
    reference: str


class BackgroundMessage(_Control):
    """The theme behind the terminal, Neovim's 'background' follows it."""

    type: Literal["background"]
    light: bool


class ShowErrorMessage(_Control):
    """`message`: Neovim shows `text` as an error, for failures that have no request."""

    type: Literal["message"]
    text: str


_CONTROL_MESSAGES: TypeAdapter[_Control] = TypeAdapter(
    Annotated[
        ResizeMessage
        | ReplyMessage
        | SetContentMessage
        | OpenFileMessage
        | RenamedMessage
        | InsertLinesMessage
        | InsertFrameMessage
        | InsertImageMessage
        | BackgroundMessage
        | ShowErrorMessage,
        Field(discriminator="type"),
    ]
)
# Applied before the next message is read, so a later theme wins
_IN_ORDER = (BackgroundMessage,)


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

    def notify(self, event_type: str, **fields: object) -> None:
        """Queue the frame of *event_type* for the browser, behind the notices queued before it."""
        self._notices.put_nowait({"type": event_type, **fields})

    async def relay_notices(self) -> None:
        while True:
            await self._websocket.send_json(await self._notices.get())

    def resolve(self, request_id: int, reply: dict) -> None:
        """Hand the browser's `saved` or `clipboard` *reply* to the question *request_id*."""
        pending = self._pending.get(request_id)
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


async def _open_file(session: NeovimSession, saves: _BrowserSaves, roots: FileRoots, path: str, line: int | None, column: int) -> None:
    try:
        await session.open_file(path, disk_file(roots, path), line, column)
    except (RuntimeError, ConnectionError, TimeoutError) as exc:
        log.warning("Neovim could not open %s: %s", path, exc)
        saves.notify("open_failed", path=path, message=str(exc))


def _renaming(session: NeovimSession, roots: FileRoots, old_path: str, path: str) -> Coroutine[None, None, None]:
    """Move document *old_path* to *path*, whose target is a notebook text's virtual name or the renamed file."""
    resolve = notebook_buffer if session.is_notebook_text(old_path) else disk_file
    return _guarded("take the renamed file", session.rename(old_path, path, resolve(roots, path)))


async def _relay_input(websocket: WebSocket, session: NeovimSession, saves: _BrowserSaves, roots: FileRoots) -> None:
    # `set_text`, `rename` and the others wait for a running `:w`, whose answer arrives on this socket, or for each other
    # in Neovim, so the work a control message returns runs beside the relay, except `_IN_ORDER` messages
    updates: set[asyncio.Task] = set()

    def beside_relay(work: Coroutine[None, None, None]) -> None:
        update = asyncio.create_task(work)
        updates.add(update)
        update.add_done_callback(updates.discard)

    handlers: dict[type[BaseModel], Callable[[Any], Coroutine[None, None, None] | None]] = {
        ResizeMessage: lambda message: session.resize(message.rows, message.cols),
        ReplyMessage: lambda message: saves.resolve(message.id, message.model_extra or {}),
        SetContentMessage: lambda message: _guarded("take the changed file", session.set_text(message.content, message.path)),
        OpenFileMessage: lambda message: _open_file(session, saves, roots, message.path, message.line, message.column),
        RenamedMessage: lambda message: _renaming(session, roots, message.old_path, message.path),
        InsertLinesMessage: lambda message: _guarded("take the inserted lines", session.insert_lines(message.text)),
        InsertFrameMessage: lambda message: _guarded("take the inserted slide", session.insert_frame(message.text)),
        InsertImageMessage: lambda message: _guarded("take the image", session.insert_image(message.path, message.reference)),
        BackgroundMessage: lambda message: _guarded("take the theme", session.set_background(message.light)),
        ShowErrorMessage: lambda message: _guarded("show the message", session.show_error(message.text)),
    }
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                session.write_input(message["bytes"])
                continue
            try:
                control = _CONTROL_MESSAGES.validate_json(message.get("text") or "")
            except ValidationError:
                continue
            work = handlers[type(control)](control)
            if work is None:
                continue
            if isinstance(control, _IN_ORDER):
                await work
            else:
                beside_relay(work)
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
            save=saves.save,
            on_yank=lambda text: saves.notify("yank", content=text),
            on_selection=(lambda selection: saves.notify("selection", **selection)) if ask_keymap else None,
            on_switched=lambda path: saves.notify("switched", path=path),
            read_clipboard=saves.read_clipboard,
            on_run_cell=lambda path, line: saves.notify("run_cell", path=path, line=line),
            on_leave=lambda: saves.notify("leave"),
            on_forward_search=lambda path, line: saves.notify("forward_search", path=path, line=line),
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
