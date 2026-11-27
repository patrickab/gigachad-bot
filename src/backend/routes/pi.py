"""`/api/pi/*`: Pi coding sessions for logins with host access (`backend.identity.HOST_CAPABILITY`), served by pi-host.

- Consumes `lib.pi_host`, the only module that knows pi-host (D6). These routes never build pi-host URLs
- `repo` is a name from `repository_index.list_repositories` (folders under `~/git` and `~/dev`, Git checkouts under `~`),
  resolved to the session folder here (`repository_index.resolve`)
- `GET /files` also takes an absolute `cwd` for the `@` completion of workbench chats: inside the file roots or the
  workbench folder only
- `POST /pastes` stores a video pasted into a Pi composer in the session folder's `.pi/pastes/` (`lib.pi_pastes`), the
  composer mentions it and the user's Pi extension `~/.pi/agent/extensions/video-paste.ts` (from `~/.os`) previews it
- pi-host failures map to HTTP: unreachable 503, unknown shape 502, refusal keeps its status and answers
  `{"detail": <error text>, "body": <pi-host body>}` (409 `locked`, 409 stale state with the current `{revision, data}`)
- `GET /media/{sha256}` and `/media/{sha256}/thumb`: tool-result images pi-host stored under their content hash, and a
  480px WebP of one (`lib.pi_media`). The hash is the whole identity, so both answer `ETag` + `immutable` and a repeat
  with `If-None-Match` is a 304 without asking pi-host
- `POST /sessions/{id}/turns/{entry_id}/delete` `{through}`: pi-host rewrites the session file (D18), refusals
  409 `running|compacted|locked|changed` pass through; the thumbnails of the removed images leave `lib.pi_media`.
  Needs a trusted request besides host access
- `WS /api/pi/ws`: byte-transparent proxy to pi-host's `/ws` (M16), nothing is parsed. Either side closing closes both

Host code execution: `host_access_only` on HTTP, the origin and host-access gate of `is_allowed_websocket` on the socket.
"""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from datetime import datetime
import logging
from pathlib import Path as FilePath
import re
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, Request, Response, UploadFile, WebSocket
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel
from starlette import status

from backend.http_plumbing import http_errors, not_modified
from backend.identity import RequestIdentity, get_request_identity, host_access_only
from backend.reverse_proxy import relay_websocket
from backend.routes.deps import get_file_roots
from backend.websocket_origin import POLICY_VIOLATION, is_allowed_websocket, require_trusted_request
import config
from lib import pi_host, pi_media, pi_pastes, repository_index
from lib.file_roots import FileRoots

log = logging.getLogger(__name__)


class _PassRefusals(APIRoute):
    """Answers a `pi_host.PiHostRejected` with pi-host's status, its error text as `detail` and its whole body as `body`."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def handle(request: Request) -> Response:
            try:
                return await handler(request)
            except pi_host.PiHostRejected as exc:
                return JSONResponse(status_code=exc.status, content={"detail": str(exc), "body": exc.body})

        return handle


router = APIRouter(prefix="/api/pi", tags=["pi"])
http_router = APIRouter(dependencies=[host_access_only("pi")], route_class=_PassRefusals)

MediaHash = Annotated[str, Path(pattern=pi_media.HASH_PATTERN)]
IMMUTABLE = "public, max-age=31536000, immutable"

# Pi session ids are UUIDs, the pattern keeps anything else out of pi-host's URL space.
SessionId = Annotated[str, Path(pattern=r"^[A-Za-z0-9-]+$")]



class RepositoryOut(BaseModel):
    name: str
    path: str


class OpenSessionBody(pi_host.ContractModel):
    repo: str
    resume: pi_host.Resume = None
    model: pi_host.ModelChoice | None = None
    thinking: str | None = None


class PinBody(pi_host.ContractModel):
    pinned: bool


class PastedVideo(BaseModel):
    """A stored paste: `path` absolute, `mention` relative to the session folder (what the composer puts after `@`)."""

    path: str
    mention: str


async def _call[Result](request: Awaitable[Result]) -> Result:
    """Await a `pi_host` call, mapping unreachable to 503 and unknown shape to 502. Refusals reach `_PassRefusals`."""
    try:
        return await request
    except pi_host.PiHostUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except pi_host.PiHostBadResponse as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _cwd(repo: str) -> str:
    folder = repository_index.resolve(repo)
    if folder is None:
        raise HTTPException(status_code=404, detail=f"Unknown repository {repo}")
    return str(folder)


@http_router.get("/settings")
async def get_settings() -> pi_host.PiSettings:
    return await _call(pi_host.get_settings())


@http_router.put("/settings")
async def put_settings(body: pi_host.PiSettings) -> pi_host.PiSettings:
    """Store pi-host's settings, `autoResume`: reopen the runs a pi-host crash or reboot interrupted (off by default)."""
    return await _call(pi_host.set_settings(auto_resume=body.auto_resume))


@http_router.get("/repos")
async def get_repositories() -> list[RepositoryOut]:
    """Start folders, the most recently used (newest session activity) first, never-used ones after."""
    repositories = await asyncio.to_thread(repository_index.list_repositories)
    try:
        sessions = await pi_host.list_sessions(None, archived=False)
    except (pi_host.PiHostUnavailable, pi_host.PiHostBadResponse):
        sessions = []  # no usage history without pi-host: keep the discovery order
    used: dict[str, str] = {}
    for session in sessions:
        used[session.cwd] = max(used.get(session.cwd, ""), session.modified_at)
    repositories.sort(key=lambda repository: used.get(repository.path, ""), reverse=True)  # stable: ties keep order
    return [RepositoryOut(name=repository.name, path=repository.path) for repository in repositories]


@http_router.get("/sessions")
async def get_sessions(repo: str | None = None, archived: bool = False) -> pi_host.SessionList:
    """Sessions newest first, of *repo* when given, factory tasks left out (the factory core lists those), so the
    session picker and sidebar never show one, also an older task whose file sits in Pi's normal root."""
    cwd = None if repo is None else await asyncio.to_thread(_cwd, repo)
    sessions = await _call(pi_host.list_sessions(cwd, archived=archived))
    return pi_host.SessionList(sessions=[session for session in sessions if session.factory is None])


@http_router.post("/sessions")
async def post_session(body: OpenSessionBody) -> pi_host.SessionAnswer:
    """Open a coding session in *repo*: new (`resume` null), a given file (`{sessionPath}`)
    or a fork of one up to an entry (`{forkOf: {sessionPath, entryId}}`)."""
    cwd = await asyncio.to_thread(_cwd, body.repo)
    session = await _call(pi_host.open_session(cwd, body.resume, model=body.model, thinking=body.thinking))
    return pi_host.SessionAnswer(session=session)


@http_router.post("/sessions/{session_id}/close", status_code=204)
async def post_close(session_id: SessionId) -> None:
    await _call(pi_host.close_session(session_id))


@http_router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: SessionId) -> None:
    await _call(pi_host.delete_session(session_id))


@http_router.post("/sessions/{session_id}/archive")
async def post_archive(session_id: SessionId) -> pi_host.SessionAnswer:
    return pi_host.SessionAnswer(session=await _call(pi_host.archive_session(session_id)))


class TurnDeleteBody(BaseModel):
    through: Literal["turn", "end"]


@http_router.post("/sessions/{session_id}/turns/{entry_id}/delete", dependencies=[Depends(require_trusted_request)])
async def post_delete_turn(session_id: SessionId, entry_id: SessionId, body: TurnDeleteBody) -> pi_host.TurnDeleted:
    """Delete a turn for good. Its thumbnails of the removed tool images go too: pi-host's cleanup does not reach them."""
    deleted = await _call(pi_host.delete_turn(session_id, entry_id, body.through))
    # A failed scan leaves the images' hashes only in `not_removed`, among paste ids and other names.
    pi_media.forget([*deleted.removed_media, *(name for name in deleted.not_removed if re.fullmatch(pi_media.HASH_PATTERN, name))])
    return deleted


@http_router.post("/sessions/{session_id}/restore")
async def post_restore(session_id: SessionId) -> pi_host.SessionAnswer:
    return pi_host.SessionAnswer(session=await _call(pi_host.restore_session(session_id)))


@http_router.put("/sessions/{session_id}/pin")
async def put_pin(session_id: SessionId, body: PinBody) -> pi_host.SessionAnswer:
    return pi_host.SessionAnswer(session=await _call(pi_host.pin_session(session_id, body.pinned)))


@http_router.get("/sessions/{session_id}/state")
async def get_state(session_id: SessionId) -> pi_host.SessionState:
    return await _call(pi_host.session_state(session_id))


@http_router.put("/sessions/{session_id}/state")
async def put_state(session_id: SessionId, body: pi_host.SessionState) -> pi_host.StateRevision:
    """Store the session's private state, *body.revision* being the one read. A stale one answers 409 with the current state."""
    return pi_host.StateRevision(revision=await _call(pi_host.write_session_state(session_id, body)))


@http_router.get("/sessions/{session_id}/entries")
async def get_entries(
    session_id: SessionId,
    turns: int | None = Query(default=None, ge=1),
    custom_type: str | None = Query(default=None, alias="customType"),
    since: str | None = None,
    assistant: bool = False,
) -> pi_host.SessionEntries:
    """The session's persisted entries, for a preview of a session that is not open; *turns* keeps only the last ones.

    *customType* keeps the `custom` entries whose type starts with it and *since* those after that entry id: the
    timeline of a factory task. pi-host refuses them together with *turns* (400 `invalid_query`) and a *since* it
    does not know (400 `invalid_since`). *assistant* adds the assistant answers among them (text, model, usage): what
    the coder said in each round.
    """
    return await _call(pi_host.session_entries(session_id, turns, custom_type=custom_type, since=since, assistant=assistant))


@http_router.get("/search")
async def get_search(q: str, limit: int = Query(default=20, ge=1, le=100)) -> pi_host.SearchResult:
    """Live sessions of every repo whose transcript matches *q*, best first (M24)."""
    return pi_host.SearchResult(hits=await _call(pi_host.search_sessions(q, limit)))


@http_router.get("/sessions/{session_id}/matches")
async def get_session_matches(
    session_id: SessionId, q: str, limit: int = Query(default=50, ge=1, le=500)
) -> pi_host.SessionMatches:
    """The best *limit* hits of the session's transcript for *q*, ranked, each addressed by entry id and ordinal."""
    return await _call(pi_host.session_matches(session_id, q, limit))


def _allowed_cwd(cwd: str, roots: FileRoots) -> str:
    """The absolute folder *cwd* resolved, 400 when relative, 404 unless it is a folder in the file roots or the workbench."""
    folder = FilePath(cwd)
    if not folder.is_absolute():
        raise HTTPException(status_code=400, detail="cwd must be absolute")
    folder = folder.resolve()
    if not folder.is_dir() or not (folder.is_relative_to(config.WORKBENCH.resolve()) or roots.contains(folder)):
        raise HTTPException(status_code=404, detail=f"Unknown folder {cwd}")
    return str(folder)


@http_router.get("/files")
async def get_files(
    roots: Annotated[FileRoots, Depends(get_file_roots)],
    repo: str | None = None,
    cwd: str | None = None,
    q: str = "",
    limit: int | None = Query(default=None, ge=1, le=500),
) -> pi_host.FileList:
    """Files of *repo*, or of the absolute folder *cwd* of a workbench chat, fuzzy-ranked against *q* (50 unless
    *limit*), for `@` completion. Without *q* the whole list unranked unless *limit* is given (K6: the UI ranks it)."""
    if (repo is None) == (cwd is None):
        raise HTTPException(status_code=400, detail="Give either repo or cwd")
    folder = await asyncio.to_thread(_cwd, repo) if repo is not None else await asyncio.to_thread(_allowed_cwd, cwd, roots)
    return pi_host.FileList(files=await _call(pi_host.find_files(folder, q, limit)))


@http_router.post("/pastes")
async def post_paste(
    roots: Annotated[FileRoots, Depends(get_file_roots)],
    file: Annotated[UploadFile, File()],
    repo: str | None = None,
    cwd: str | None = None,
) -> PastedVideo:
    """Store a video pasted into the composer of a session in *repo* or the absolute folder *cwd* as
    `.pi/pastes/<YYYYMMDD-HHMMSS>-<name>.<ext>`. Unknown folder 404, no mp4/webm/mov/mkv 400, over 200 MB 413."""
    if (repo is None) == (cwd is None):
        raise HTTPException(status_code=400, detail="Give either repo or cwd")
    filename = file.filename or ""
    with http_errors((ValueError, 400, None)):
        suffix = pi_pastes.paste_suffix(filename, file.content_type)
    content = await file.read(pi_pastes.MAX_PASTE_BYTES + 1)
    if len(content) > pi_pastes.MAX_PASTE_BYTES:
        raise HTTPException(status_code=413, detail="Videos are at most 200 MB")
    folder = FilePath(await asyncio.to_thread(_cwd, repo) if repo is not None else cwd or "")
    mention = pi_pastes.paste_mention(filename, suffix, datetime.now())
    with http_errors(
        (pi_pastes.UnknownFolder, 404, None),
        (FileExistsError, 409, "A paste with that name exists already"),
        (ValueError, 400, None),
    ):
        stored = await asyncio.to_thread(pi_pastes.store_paste, roots, folder, mention, content)
    return PastedVideo(path=str(stored), mention=mention.as_posix())


@http_router.get("/media/{media_hash}")
async def get_media(request: Request, media_hash: MediaHash) -> Response:
    etag = f'"{media_hash}"'
    if (unchanged := not_modified(request, etag, IMMUTABLE)) is not None:
        return unchanged
    data, media_type = await _call(pi_host.media(media_hash))
    return Response(content=data, media_type=media_type, headers={"ETag": etag, "Cache-Control": IMMUTABLE})


@http_router.get("/media/{media_hash}/thumb")
async def get_media_thumbnail(request: Request, media_hash: MediaHash) -> Response:
    etag = f'"{media_hash}-{pi_media.THUMB_PX}w"'
    if (unchanged := not_modified(request, etag, IMMUTABLE)) is not None:
        return unchanged
    thumb = pi_media.thumbnail_for(media_hash)
    if thumb is None:
        data, media_type = await _call(pi_host.media(media_hash))
        try:
            thumb = await asyncio.to_thread(pi_media.thumbnail_for, media_hash, data)
        except ValueError:  # unreadable for Pillow: the original is the best preview
            etag = f'"{media_hash}"'
            headers = {"ETag": etag, "Cache-Control": IMMUTABLE}
            return not_modified(request, etag, IMMUTABLE) or Response(content=data, media_type=media_type, headers=headers)
    return Response(content=thumb or b"", media_type="image/webp", headers={"ETag": etag, "Cache-Control": IMMUTABLE})


@router.websocket("/ws")
async def pi_socket(websocket: WebSocket, identity: Annotated[RequestIdentity, Depends(get_request_identity)]) -> None:
    # Refusals accept first: a close before the handshake becomes HTTP 403, and the browser sees 1006 for every case.
    if not is_allowed_websocket(websocket, identity):
        await websocket.accept()
        await websocket.close(code=POLICY_VIOLATION)
        return
    try:
        upstream = await pi_host.open_websocket()
    except pi_host.PiHostUnavailable as exc:
        log.warning("%s", exc)
        await websocket.accept()
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return
    await relay_websocket(websocket, upstream)


router.include_router(http_router)
