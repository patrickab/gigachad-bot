import asyncio
import contextlib
from typing import Annotated

from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel, StringConstraints

from backend.http_plumbing import http_errors
from backend.identity import host_access_only
from backend.routes.deps import get_project_store
from backend.routes.upstream import call_upstream
from lib import learnharness
from lib.data_store import StorageConflictError
from lib.project_store import ProjectStore

router = APIRouter(prefix="/api/learn", tags=["learn"], dependencies=[host_access_only("learn")])

# LearnHarness session ids are `chat_<time>_<hex>`, the pattern keeps anything else out of its URL space.
SessionId = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]+$")]
ProjectStoreDep = Annotated[ProjectStore, Depends(get_project_store)]


class LearnSessionEntry(BaseModel):
    id: str
    title: str
    kind: str | None
    busy: bool
    compacting: bool
    pending_question: bool
    pending_problem: bool
    updated_at: str | None
    # The project the session is grouped under, None when it belongs to none.
    project_slug: str | None


class SessionProject(BaseModel):
    project_slug: str | None


class SessionTitle(BaseModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@router.get("/sessions")
async def get_sessions(store: ProjectStoreDep) -> list[LearnSessionEntry]:
    """Every learn session with its project. Mapped sessions LearnHarness no longer lists are dropped from their project."""
    # The mapping is read before the list: a session it holds was created before the list was read, so a session it
    # holds and the list lacks was deleted in LearnHarness, never one created in between.
    projects = await asyncio.to_thread(store.learn_session_projects)
    sessions = await call_upstream(learnharness.list_sessions())
    if gone := projects.keys() - {session.id for session in sessions}:
        # Best effort: a busy project.json keeps the stale ids until the next read.
        with contextlib.suppress(StorageConflictError):
            await asyncio.to_thread(store.forget_learn_sessions, gone)
    return [LearnSessionEntry(**session.model_dump(), project_slug=projects.get(session.id)) for session in sessions]


@router.put("/sessions/{session_id}/project", status_code=204)
def put_session_project(session_id: SessionId, body: SessionProject, store: ProjectStoreDep) -> None:
    """Group the session under a project (out of any other), or under none."""
    with http_errors((FileNotFoundError, 404, "Project not found"), (StorageConflictError, 409, None)):
        store.assign_learn_session(session_id, body.project_slug)


@router.patch("/sessions/{session_id}")
async def patch_session(session_id: SessionId, body: SessionTitle) -> SessionTitle:
    """Rename the session in LearnHarness, answering the title it stored."""
    return SessionTitle(title=await call_upstream(learnharness.rename_session(session_id, body.title)))


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: SessionId, store: ProjectStoreDep, mode: learnharness.DeleteMode = "preserve-graph") -> None:
    """Delete the session in LearnHarness, its Knowledge kept or (`delete-nodes`) its graph traces too, then drop it from its project.

    `delete-nodes` is upstream's "Also remove its graph traces".
    """
    await call_upstream(learnharness.delete_session(session_id, mode))
    # Best effort: the next session list drops an id LearnHarness no longer lists anyway.
    with contextlib.suppress(StorageConflictError):
        await asyncio.to_thread(store.forget_learn_sessions, {session_id})


class SessionMarkdown(BaseModel):
    # LearnHarness's file name for the export (`<slugified title>.md`).
    filename: str
    markdown: str


@router.get("/sessions/{session_id}/markdown")
async def get_session_markdown(session_id: SessionId) -> SessionMarkdown:
    """The session as LearnHarness exports it: one Markdown document and its file name (a header the browser hides cross-origin)."""
    export = await call_upstream(learnharness.export_markdown(session_id))
    return SessionMarkdown(filename=export.filename, markdown=export.markdown)
