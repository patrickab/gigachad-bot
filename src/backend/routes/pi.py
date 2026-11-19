"""`/api/pi/*`: Pi coding sessions for logins with host access (`backend.identity.HOST_CAPABILITY`), served by pi-host.

- Consumes `lib.pi_host`, the only module that knows pi-host (D6). These routes never build pi-host URLs
- `repo` is a checkout name under `~/git`, resolved to the session folder here (`pi_host.resolve_repository`)
- pi-host failures map to HTTP: unreachable 503, unknown shape 502, refusal keeps its status and answers
  `{"detail": <error text>, "body": <pi-host body>}` (409 `locked`, 409 stale state with the current `{revision, data}`)
- `WS /api/pi/ws`: byte-transparent proxy to pi-host's `/ws` (M16), nothing is parsed. Either side closing closes both

Host code execution: `host_access_only` on HTTP, the origin and host-access gate of `is_allowed_websocket` on the socket.
"""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response, WebSocket
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel

from backend.identity import RequestIdentity, get_request_identity, host_access_only
from backend.reverse_proxy import relay_websocket
from backend.websocket_origin import POLICY_VIOLATION, is_allowed_websocket
from lib import pi_host

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

# Pi session ids are UUIDs, the pattern keeps anything else out of pi-host's URL space.
SessionId = Annotated[str, Path(pattern=r"^[A-Za-z0-9-]+$")]

_TRY_AGAIN_LATER = 1013


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


async def _call[Result](request: Awaitable[Result]) -> Result:
    """Await a `pi_host` call, mapping unreachable to 503 and unknown shape to 502. Refusals reach `_PassRefusals`."""
    try:
        return await request
    except pi_host.PiHostUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except pi_host.PiHostBadResponse as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _cwd(repo: str) -> str:
    folder = pi_host.resolve_repository(repo)
    if folder is None:
        raise HTTPException(status_code=404, detail=f"Unknown repository {repo}")
    return str(folder)


@http_router.get("/repos")
async def get_repositories() -> list[RepositoryOut]:
    repositories = await asyncio.to_thread(pi_host.list_repositories)
    return [RepositoryOut(name=repository.name, path=repository.path) for repository in repositories]


@http_router.get("/sessions")
async def get_sessions(repo: str | None = None, archived: bool = False) -> pi_host.SessionList:
    """Sessions newest first, of *repo* when given."""
    cwd = None if repo is None else await asyncio.to_thread(_cwd, repo)
    return pi_host.SessionList(sessions=await _call(pi_host.list_sessions(cwd, archived=archived)))


@http_router.post("/sessions")
async def post_session(body: OpenSessionBody) -> pi_host.SessionAnswer:
    """Open a coding session in *repo*: new (`resume` null), the newest (`"continue"`), a given file (`{sessionPath}`)
    or a fork of one up to an entry (`{forkOf: {sessionPath, entryId}}`)."""
    cwd = await asyncio.to_thread(_cwd, body.repo)
    session = await _call(pi_host.open_session(cwd, body.resume, model=body.model, thinking=body.thinking))
    return pi_host.SessionAnswer(session=session)


@http_router.post("/sessions/{session_id}/close", status_code=204)
async def post_close(session_id: SessionId) -> None:
    await _call(pi_host.close_session(session_id))


@http_router.post("/sessions/{session_id}/archive")
async def post_archive(session_id: SessionId) -> pi_host.SessionAnswer:
    return pi_host.SessionAnswer(session=await _call(pi_host.archive_session(session_id)))


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


@http_router.get("/search")
async def get_search(q: str, limit: int = Query(default=20, ge=1, le=100)) -> pi_host.SearchResult:
    """Live sessions of every repo whose transcript matches *q*, best first (M24)."""
    return pi_host.SearchResult(hits=await _call(pi_host.search_sessions(q, limit)))


@http_router.get("/files")
async def get_files(repo: str, q: str = "", limit: int | None = Query(default=None, ge=1, le=500)) -> pi_host.FileList:
    """Files of *repo* fuzzy-ranked against *q* (50 unless *limit*), for `@` completion. Without *q* the whole
    list unranked unless *limit* is given (K6: the UI ranks it)."""
    cwd = await asyncio.to_thread(_cwd, repo)
    return pi_host.FileList(files=await _call(pi_host.find_files(cwd, q, limit)))


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
        await websocket.close(code=_TRY_AGAIN_LATER)
        return
    await relay_websocket(websocket, upstream)


router.include_router(http_router)
