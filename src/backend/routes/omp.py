from collections.abc import Awaitable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel

from backend.identity import HOST_CAPABILITY, RequestIdentity, get_request_identity
from lib import omp_web


def require_host_access(identity: RequestIdentity = Depends(get_request_identity)) -> None:
    """Refuse logins without the host capability: omp-web runs the machine owner's omp, it has no users of its own."""
    if not identity.has_host_access:
        raise HTTPException(status_code=403, detail=f"omp is limited to logins granted {HOST_CAPABILITY}")


router = APIRouter(prefix="/api/omp", tags=["omp"], dependencies=[Depends(require_host_access)])

# omp session ids are UUIDs, the pattern keeps anything else out of omp-web's URL space.
SessionId = Annotated[str, Path(pattern=r"^[A-Za-z0-9-]+$")]


class WorkspaceBody(BaseModel):
    path: str


class RestoreBody(BaseModel):
    key: str


class RestoredSession(BaseModel):
    session_id: str


async def _call[Result](request: Awaitable[Result]) -> Result:
    """Await an omp-web call, mapping its failures to HTTP: down 503, unknown shape 502, refusal passed through."""
    try:
        return await request
    except omp_web.OmpWebUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except omp_web.OmpWebContractError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except omp_web.OmpWebRequestFailed as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


@router.get("/overview")
async def get_overview() -> omp_web.OmpOverview:
    return await _call(omp_web.overview())


@router.post("/workspaces")
async def post_workspace(body: WorkspaceBody) -> omp_web.OmpWorkspace:
    return await _call(omp_web.add_workspace(body.path))


@router.get("/archive")
async def get_archive() -> list[omp_web.OmpArchivedSession]:
    return await _call(omp_web.archived_sessions())


@router.post("/archive/restore")
async def post_restore(body: RestoreBody) -> RestoredSession:
    return RestoredSession(session_id=await _call(omp_web.restore_session(body.key)))


@router.post("/sessions/{session_id}/archive", status_code=204)
async def post_archive(session_id: SessionId) -> None:
    await _call(omp_web.archive_session(session_id))


@router.get("/sessions/{session_id}/system-prompt")
async def get_system_prompt(session_id: SessionId) -> omp_web.OmpSystemPrompt:
    return await _call(omp_web.system_prompt(session_id))
