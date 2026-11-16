import asyncio
from datetime import datetime
from pathlib import Path as FilePath
import re
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, UploadFile
from pydantic import BaseModel

from backend.http_plumbing import http_errors
from backend.identity import host_access_only
from backend.routes.deps import get_disk_files, get_file_roots
from backend.routes.upstream import call_upstream
from lib import omp_web
from lib.disk_files import DiskFiles
from lib.file_roots import FileRoots

router = APIRouter(prefix="/api/omp", tags=["omp"], dependencies=[host_access_only("omp")])

# omp session ids are UUIDs, the pattern keeps anything else out of omp-web's URL space.
SessionId = Annotated[str, Path(pattern=r"^[A-Za-z0-9-]+$")]

# Pasted videos, relative to the session's working directory, where omp's `@` mention reads them.
_PASTES_DIRECTORY = FilePath(".omp") / "pastes"
_MAX_PASTE_BYTES = 200 * 1024 * 1024
_VIDEO_SUFFIXES = frozenset({".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", ".wmv", ".flv", ".mpg", ".mpeg", ".3gp", ".ogv"})


class RestoreBody(BaseModel):
    key: str


class RestoredSession(BaseModel):
    session_id: str


class PastedVideo(BaseModel):
    """Where a pasted video was stored: `mention` is its path relative to the session's working directory."""

    path: str
    mention: str


@router.get("/overview")
async def get_overview() -> omp_web.OmpOverview:
    return await call_upstream(omp_web.overview())


@router.get("/archive")
async def get_archive() -> list[omp_web.OmpArchivedSession]:
    return await call_upstream(omp_web.archived_sessions())


@router.post("/archive/restore")
async def post_restore(body: RestoreBody) -> RestoredSession:
    return RestoredSession(session_id=await call_upstream(omp_web.restore_session(body.key)))


@router.post("/sessions/{session_id}/archive", status_code=204)
async def post_archive(session_id: SessionId) -> None:
    await call_upstream(omp_web.archive_session(session_id))


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: SessionId) -> None:
    await call_upstream(omp_web.delete_session(session_id))


@router.get("/sessions/{session_id}/system-prompt")
async def get_system_prompt(session_id: SessionId) -> omp_web.OmpSystemPrompt:
    return await call_upstream(omp_web.system_prompt(session_id))


@router.get("/sessions/{session_id}/history")
async def get_history(session_id: SessionId) -> list[omp_web.ChatMessage]:
    """The session as a chat history, what `/memorize` reads in an omp tab."""
    return await call_upstream(omp_web.session_history(session_id))


@router.post("/pastes")
async def post_paste(
    file: UploadFile = File(...),
    session_id: str | None = Query(default=None, pattern=r"^[A-Za-z0-9-]+$"),
    cwd: str | None = Query(default=None),
    roots: FileRoots = Depends(get_file_roots),
    files: DiskFiles = Depends(get_disk_files),
) -> PastedVideo:
    """Store a video pasted into an omp composer as `<cwd>/.omp/pastes/<YYYYMMDD-HHMMSS>-<name>`.

    - `session_id` -> the session's working directory, read from omp-web, `cwd` -> a new session's folder, exactly one of them
    - The folder must pass the file roots (a mount, or host-readable), `.omp` and `.omp/pastes` must not be symlinks
    - Anything but a video answers 400, more than 200 MB 413
    """
    if (session_id is None) == (cwd is None):
        raise HTTPException(status_code=400, detail="Give either a session or a folder")
    suffix = FilePath(file.filename or "").suffix.lower()
    if suffix not in _VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only videos can be pasted")
    content = await file.read(_MAX_PASTE_BYTES + 1)
    if len(content) > _MAX_PASTE_BYTES:
        raise HTTPException(status_code=413, detail="Videos are at most 200 MB")
    with http_errors((LookupError, 404, None)):
        folder = FilePath(cwd if session_id is None else await call_upstream(omp_web.session_cwd(session_id)))
    # The server names the file: the client's name loses its folders and anything an `@` mention would split on
    stem = re.sub(r"[^\w.-]+", "_", FilePath(file.filename or "").stem)[:80]
    mention = _PASTES_DIRECTORY / f"{datetime.now():%Y%m%d-%H%M%S}-{stem}{suffix}"
    with http_errors((FileExistsError, 409, "A paste with that name exists already"), (ValueError, 400, None), (OSError, 400, None)):
        stored = await asyncio.to_thread(_store_paste, roots, files, folder, mention, content)
    return PastedVideo(path=str(stored), mention=mention.as_posix())


def _store_paste(roots: FileRoots, files: DiskFiles, folder: FilePath, mention: FilePath, content: bytes) -> FilePath:
    """Create `folder / mention` exclusively, ValueError for a relative, missing, unreachable or symlinked folder."""
    if not (folder.is_absolute() and roots.contains(folder) and folder.is_dir()):
        raise ValueError(f"The folder {folder} is not reachable")
    for part in (folder / mention.parts[0], folder / mention.parent):
        if part.is_symlink():
            raise ValueError(f"{part} is a symlink")
    (folder / mention.parent).mkdir(parents=True, exist_ok=True)
    return files.create(folder / mention, content)
