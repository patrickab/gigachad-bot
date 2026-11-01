"""Live file change reports for signed-in logins: `WS /api/files/watch`.

Text frames carry JSON, the watch lives as long as the socket:

- client -> server: `{"type": "watch", "files": [<absolute path>...], "folders": [<absolute path>...]}` replaces the whole
  watch set of this socket. A folder reports the files below it that changed, not LaTeX build output, `.git` or `node_modules`.
  Frames that are not a watch message are ignored
- server -> client: `{"type": "changed", "path"}` for a created, modified, moved-in or deleted watched file (a deleted file
  is reported too, the client reads it again and handles the 404) and `{"type": "refused", "path", "message"}` for a path the
  file roots refuse to read or that has no folder to watch in, the rest of the set is watched anyway
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from backend.routes.deps import get_file_roots
from backend.websocket_origin import is_trusted_origin
from lib.file_roots import FileRoots
from lib.file_watch import FileWatcher

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])

_POLICY_VIOLATION = 1008


class WatchMessage(BaseModel):
    type: Literal["watch"]
    files: list[str] = []
    folders: list[str] = []


def _refusal(roots: FileRoots, path: Path, *, folder: bool) -> str | None:
    """Why the file roots or the disk do not allow watching *path*, None when they do."""
    if not path.is_absolute():
        return "Path is not absolute"
    if not roots.contains(path):
        return "Path outside file roots"
    if not (path if folder else path.parent).is_dir():
        return "Not a folder" if folder else "Folder does not exist"
    return None


def _admit(roots: FileRoots, message: WatchMessage) -> tuple[list[Path], list[Path], list[dict[str, str]]]:
    """The files and folders of *message* that may be watched, and a `refused` frame per path that may not."""
    verdicts = [
        (path, folder, _refusal(roots, Path(path), folder=folder))
        for folder, paths in ((False, message.files), (True, message.folders))
        for path in paths
    ]
    refusals = [{"type": "refused", "path": path, "message": reason} for path, _folder, reason in verdicts if reason]
    files = [Path(path) for path, folder, reason in verdicts if not folder and not reason]
    folders = [Path(path) for path, folder, reason in verdicts if folder and not reason]
    return files, folders, refusals


async def _serve(websocket: WebSocket, roots: FileRoots, watcher: FileWatcher) -> None:
    while True:
        try:
            message = WatchMessage.model_validate(await websocket.receive_json())
        except (ValidationError, ValueError):
            log.warning("Ignoring a frame that is not a watch message")
            continue
        # The roots may list git worktrees, which runs git: off the event loop.
        files, folders, refusals = await asyncio.to_thread(_admit, roots, message)
        for refusal in refusals:
            await websocket.send_json(refusal)
        watcher.replace(files, folders)


@router.websocket("/watch")
async def watch_files(websocket: WebSocket, roots: FileRoots = Depends(get_file_roots)) -> None:
    """Refuse untrusted origins like `/api/neovim/session`, the login is the one `get_file_roots` requires."""
    if not is_trusted_origin(websocket.headers.get("origin")):
        await websocket.close(code=_POLICY_VIOLATION)
        return
    await websocket.accept()

    async def report(path: str) -> None:
        await websocket.send_json({"type": "changed", "path": path})

    watcher = FileWatcher(report)
    try:
        with contextlib.suppress(WebSocketDisconnect):
            await _serve(websocket, roots, watcher)
    finally:
        await watcher.close()
