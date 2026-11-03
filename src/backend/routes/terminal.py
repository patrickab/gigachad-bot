"""Shell terminal sessions for logins with host access (`backend.identity.HOST_CAPABILITY`).

`WS /api/terminal/session` runs the user's login shell in the home directory, one per connection. Binary frames carry
terminal bytes both ways, text frames carry JSON control messages:

- client -> server: `{"type": "open", "rows", "cols"}` first, then `{"type": "resize", "rows", "cols"}`
- server -> client: `{"type": "exit", "code"}` when the shell exits

Leaving hangs up the shell's process group. Host code execution: the origin and host-access gate of `is_allowed_websocket`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
import pwd

from fastapi import APIRouter, Depends, WebSocket
from pydantic import BaseModel, Field, ValidationError
from starlette.websockets import WebSocketState

from backend.identity import RequestIdentity, get_request_identity
from backend.websocket_origin import POLICY_VIOLATION, is_allowed_websocket
from lib.pty_process import PtyProcess

router = APIRouter(prefix="/api/terminal", tags=["terminal"])

_UNSUPPORTED_DATA = 1003


class OpenMessage(BaseModel):
    type: str = Field(pattern="^open$")
    rows: int = Field(gt=0, le=1000)
    cols: int = Field(gt=0, le=1000)


def login_shell() -> str:
    """`$SHELL`, else the shell of the backend's user account."""
    return os.environ.get("SHELL") or pwd.getpwuid(os.getuid()).pw_shell or "/bin/sh"


async def _relay_output(websocket: WebSocket, shell: PtyProcess) -> None:
    while (output := await shell.read_output()) is not None:
        await websocket.send_bytes(output)
    await websocket.send_json({"type": "exit", "code": await shell.wait()})


async def _relay_input(websocket: WebSocket, shell: PtyProcess) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        if message.get("bytes") is not None:
            shell.write_input(message["bytes"])
            continue
        control = json.loads(message.get("text") or "{}")
        if control.get("type") == "resize":
            shell.resize(int(control["rows"]), int(control["cols"]))


@router.websocket("/session")
async def terminal_session(websocket: WebSocket, identity: RequestIdentity = Depends(get_request_identity)) -> None:
    if not is_allowed_websocket(websocket, identity):
        await websocket.close(code=POLICY_VIOLATION)
        return
    await websocket.accept()
    try:
        opening = OpenMessage.model_validate(await websocket.receive_json())
    except (ValidationError, ValueError):
        await websocket.close(code=_UNSUPPORTED_DATA)
        return
    shell = PtyProcess.spawn([login_shell(), "-l"], cwd=Path.home(), rows=opening.rows, cols=opening.cols)
    relays = [asyncio.create_task(_relay_output(websocket, shell)), asyncio.create_task(_relay_input(websocket, shell))]
    try:
        await asyncio.wait(relays, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for relay in relays:
            relay.cancel()
        await asyncio.gather(*relays, return_exceptions=True)
        await shell.close()
        if websocket.client_state == WebSocketState.CONNECTED:
            with contextlib.suppress(RuntimeError):
                await websocket.close()
