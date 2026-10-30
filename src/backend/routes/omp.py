from typing import Annotated

from fastapi import APIRouter, Path
from pydantic import BaseModel

from backend.identity import host_access_only
from backend.routes.upstream import call_upstream
from lib import omp_web

router = APIRouter(prefix="/api/omp", tags=["omp"], dependencies=[host_access_only("omp")])

# omp session ids are UUIDs, the pattern keeps anything else out of omp-web's URL space.
SessionId = Annotated[str, Path(pattern=r"^[A-Za-z0-9-]+$")]


class RestoreBody(BaseModel):
    key: str


class RestoredSession(BaseModel):
    session_id: str


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


@router.get("/sessions/{session_id}/system-prompt")
async def get_system_prompt(session_id: SessionId) -> omp_web.OmpSystemPrompt:
    return await call_upstream(omp_web.system_prompt(session_id))


@router.get("/sessions/{session_id}/history")
async def get_history(session_id: SessionId) -> list[omp_web.ChatMessage]:
    """The session as a chat history, what `/memorize` reads in an omp tab."""
    return await call_upstream(omp_web.session_history(session_id))
