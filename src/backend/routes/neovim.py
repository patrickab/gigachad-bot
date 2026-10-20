"""Neovim terminal sessions for logins with host access (`backend.identity.HOST_CAPABILITY`).

`WS /api/neovim/session` runs one `nvim` per connection. Binary frames carry terminal
bytes both ways, text frames carry JSON control messages:

- client -> server: `{"type": "open", "path", "filename", "content", "base_content"?, "rows", "cols"}` first
  (a string `base_content` opens a read-only left diff buffer against the editable document),
  then `{"type": "resize", "rows", "cols"}` and `{"type": "saved", "id", "content" | "error"}`
- server -> client: `{"type": "save", "id", "content"}` on `:w`, `{"type": "exit", "code"}` when `nvim` quits

`nvim` starts in the mounted root holding a file-root document, in the home directory otherwise.

The browser persists `:w` through the document's own API, so every save keeps that
API's validation and side effects. Disconnecting discards unsaved edits.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, WebSocket
from pydantic import BaseModel, Field, ValidationError
from starlette.websockets import WebSocketState

from backend.identity import RequestIdentity, get_request_identity
from backend.routes.deps import get_file_roots
from config import CORS_ORIGINS
from lib.file_roots import FileRoots
from lib.neovim_session import NeovimSession, is_neovim_installed

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/neovim", tags=["neovim"])

_SAVE_TIMEOUT_SECONDS = 60
_POLICY_VIOLATION = 1008
_UNSUPPORTED_DATA = 1003
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class OpenMessage(BaseModel):
    """`path` is the document's app path: absolute for file-root files, a logical key otherwise."""

    path: str
    filename: str
    content: str
    base_content: str | None = None
    rows: int = Field(ge=1, le=1000)
    cols: int = Field(ge=1, le=1000)


def working_directory(roots: FileRoots, path: str) -> Path:
    """The mounted root holding a file-root document, home for everything else."""
    if Path(path).is_absolute():
        with contextlib.suppress(ValueError):
            return roots.root_of(path)
    return Path.home()


class SaveFailed(RuntimeError):
    """The browser could not persist a `:w`."""


def is_trusted_origin(origin: str | None) -> bool:
    """Allow-listed origins, or loopback ones while CORS is the `*` default.

    CORS does not cover WebSockets, and a terminal is code execution on this host,
    so any page a user visits must not be able to open one.
    """
    if not origin:
        return False
    if "*" not in CORS_ORIGINS:
        return origin in CORS_ORIGINS
    return urlsplit(origin).hostname in _LOOPBACK_HOSTS


class _BrowserSaves:
    """Forward `:w` to the browser and wait for what it stored."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._pending: dict[int, asyncio.Future[dict]] = {}
        self._next_id = 0

    async def save(self, text: str) -> str:
        self._next_id += 1
        save_id = self._next_id
        reply = self._pending[save_id] = asyncio.get_running_loop().create_future()
        try:
            await self._websocket.send_json({"type": "save", "id": save_id, "content": text})
            result = await asyncio.wait_for(reply, _SAVE_TIMEOUT_SECONDS)
        finally:
            self._pending.pop(save_id, None)
        if result.get("error"):
            raise SaveFailed(str(result["error"]))
        return str(result.get("content", text))

    def resolve(self, reply: dict) -> None:
        pending = self._pending.get(reply.get("id"))
        if pending is not None and not pending.done():
            pending.set_result(reply)


async def _relay_output(websocket: WebSocket, session: NeovimSession) -> None:
    while (output := await session.read_output()) is not None:
        await websocket.send_bytes(output)
    await websocket.send_json({"type": "exit", "code": await session.wait()})


async def _relay_input(websocket: WebSocket, session: NeovimSession, saves: _BrowserSaves) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        if message.get("bytes") is not None:
            session.write_input(message["bytes"])
            continue
        control = json.loads(message.get("text") or "{}")
        if control.get("type") == "resize":
            session.resize(int(control["rows"]), int(control["cols"]))
        elif control.get("type") == "saved":
            saves.resolve(control)


async def _run(websocket: WebSocket, session: NeovimSession) -> None:
    """Relay until `nvim` exits, the browser leaves, or the save hook cannot be installed."""
    saves = _BrowserSaves(websocket)
    relays = [
        asyncio.create_task(_relay_output(websocket, session)),
        asyncio.create_task(_relay_input(websocket, session, saves)),
    ]
    # Input flows while connecting, so a startup prompt from the user's config can be answered.
    connecting = asyncio.create_task(session.connect(saves.save))
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
    if not is_trusted_origin(websocket.headers.get("origin")) or not (identity.has_host_access and is_neovim_installed()):
        await websocket.close(code=_POLICY_VIOLATION)
        return
    await websocket.accept()
    try:
        opening = OpenMessage.model_validate(await websocket.receive_json())
    except (ValidationError, ValueError):
        await websocket.close(code=_UNSUPPORTED_DATA)
        return
    session = NeovimSession.spawn(
        opening.filename,
        opening.content,
        base_content=opening.base_content,
        rows=opening.rows,
        cols=opening.cols,
        cwd=working_directory(roots, opening.path),
    )
    try:
        await _run(websocket, session)
    finally:
        await session.close()
        if websocket.client_state == WebSocketState.CONNECTED:
            with contextlib.suppress(RuntimeError):
                await websocket.close()
