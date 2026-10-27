"""HTTP boundary for per-project LikeC4 architecture workspaces.

A project's workspace is every ``.c4`` file under ``docs/architecture/`` of its linked
file mount (the file root named like the project) when that directory exists, else under
``graph/<slug>/`` in the database, nested folders included; LikeC4 merges them into
one model. The UI organises it as
systems (``backend/backend.c4``: one C4 system and its view) holding modules
(``backend/api.c4``: ``extend backend { api = container }``), plus saved views
in ``views/<id>.c4``. Saved positions are LikeC4 manual layouts under ``.likec4/``.

Writes go through ``lib/architecture_service.apply``: read -> C4 service -> commit,
retried when another edit committed first.
"""

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from backend.routes.deps import get_architecture_store, get_project_store
from lib import architecture_service, c4_service
from lib.architecture_service import Change, require_project
from lib.architecture_workspace import ArchitectureError, ArchitectureNotFound, ArchitectureStore, Snapshot
from lib.c4_service import ApplyResult, C4Model
from lib.project_store import ProjectStore

router = APIRouter(prefix="/api/architecture", tags=["architecture"])


class WorkspaceResponse(BaseModel):
    model: C4Model
    # Per operation, what it created: an element id or a new file's path.
    created: list[str | None] = Field(default_factory=list)


class OperationsRequest(BaseModel):
    ops: list[dict[str, Any]] = Field(min_length=1)
    # The source file of the editing window; new top-level elements land there.
    file: str | None = None


class SourceFile(BaseModel):
    path: str
    content: str


class NewSourceFile(BaseModel):
    path: str
    content: str = ""


def _response(result: ApplyResult) -> WorkspaceResponse:
    return WorkspaceResponse(model=result.model, created=result.created)


@router.get("/{slug}", response_model=WorkspaceResponse)
async def read_workspace(
    slug: str,
    store: ArchitectureStore = Depends(get_architecture_store),
    projects: ProjectStore = Depends(get_project_store),
) -> WorkspaceResponse:
    """The whole model; a project without an architecture yet has an empty one."""

    def read() -> Snapshot:
        require_project(projects, slug)
        return store.read(allow_empty=True)

    # The project check and source reads block on Postgres or the mount, so they run off the event loop.
    snapshot = await asyncio.to_thread(read)
    return WorkspaceResponse(model=(await c4_service.render(snapshot.sources, snapshot.snapshots)).model)


@router.post("/{slug}/ops", response_model=WorkspaceResponse)
async def apply_operations(
    slug: str,
    req: OperationsRequest,
    store: ArchitectureStore = Depends(get_architecture_store),
    projects: ProjectStore = Depends(get_project_store),
) -> WorkspaceResponse:
    """Model, view and layout edits (``{"op": "layout", ...}``), applied all or nothing. New files come
    from ``createSystem`` (``<id>/<id>.c4``: one system and its view; the first also declares the
    element kinds), ``createModule`` (beside its system, extending it with one container) and
    ``createView`` (an empty ``views/<id>.c4``); ``created`` holds their paths."""
    change = Change(req.ops, file=req.file)
    return _response(await architecture_service.apply(store, change, check_project=lambda: require_project(projects, slug)))


@router.put("/{slug}/source", response_model=WorkspaceResponse)
async def write_source(file: SourceFile, store: ArchitectureStore = Depends(get_architecture_store)) -> WorkspaceResponse:
    """Hand edits must parse cleanly; the model is never saved in a state the diagram cannot show.
    Connections and view entries left naming an element the edit removed are dropped with it."""
    change = Change([{"op": "pruneDangling"}], edit=lambda sources: {**sources, file.path: file.content}, strict=True)
    return _response(await architecture_service.apply(store, change))


@router.post("/{slug}/source", response_model=WorkspaceResponse)
async def create_source(
    slug: str,
    file: NewSourceFile,
    store: ArchitectureStore = Depends(get_architecture_store),
    projects: ProjectStore = Depends(get_project_store),
) -> WorkspaceResponse:
    """Add a source file as written (empty by default); it must parse."""

    def add(sources: dict[str, str]) -> dict[str, str]:
        if file.path in sources:
            raise ArchitectureError(f"{file.path} already exists")
        return {**sources, file.path: file.content}

    change = Change([], edit=add, strict=True)
    return _response(await architecture_service.apply(store, change, check_project=lambda: require_project(projects, slug)))


@router.delete("/{slug}/source", response_model=WorkspaceResponse)
async def delete_source(path: list[str] = Query(...), store: ArchitectureStore = Depends(get_architecture_store)) -> WorkspaceResponse:
    """Deletes every ``path`` together (a system with its modules). Connections and view
    entries naming their elements go with them; it is refused only while the remaining
    files still need something else in them, e.g. the element kinds a ``specification`` declares.

    The layouts of views they declared go with them. Deleting the last file
    leaves an empty model.
    """

    def drop(sources: dict[str, str]) -> dict[str, str]:
        if missing := sorted(set(path) - set(sources)):
            raise ArchitectureNotFound(f"Source file not found: {missing[0]}")
        return {name: text for name, text in sources.items() if name not in path}

    change = Change([{"op": "pruneDangling"}], edit=drop, strict=True)
    return _response(await architecture_service.apply(store, change))
