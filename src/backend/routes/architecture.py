"""HTTP boundary for per-project LikeC4 architecture workspaces.

A project's workspace is every ``.c4`` file under ``graph/<slug>/``, nested
folders included; LikeC4 merges them into one model. The UI organises it as
packages (``backend/backend.c4``: one system and its view) holding modules
(``backend/api.c4``: ``extend backend { api = container }``), plus saved views
in ``views/<id>.c4``. Saved positions are LikeC4 manual layouts under ``.likec4/``.

Writes go read -> C4 service -> commit. The commit writes every changed file in
one transaction, each checked against the revision it was read at. When another
edit got there first, ``_write`` rereads the workspace and re-validates the
operations before retrying. New files, deletions, and files read but not changed
are not revision-checked: two devices racing on those can still overwrite each
other. That is accepted, as one user edits one workspace.
"""

from collections.abc import Callable, Iterable
import re
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from backend.routes.deps import get_document_store, get_project_store
from lib import c4_service
from lib.architecture_workspace import ArchitectureError, ArchitectureNotFound, ArchitectureStore
from lib.data_store import DataStore, StorageConflictError
from lib.project_store import ProjectStore

router = APIRouter(prefix="/api/architecture", tags=["architecture"])

_COMMIT_ATTEMPTS = 3


class WorkspaceResponse(BaseModel):
    model: dict[str, Any]
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


def _require_project(projects: ProjectStore, slug: str) -> None:
    try:
        projects.list_files(slug)
    except FileNotFoundError as exc:
        raise ArchitectureNotFound(str(exc)) from exc


def _refusal(error: dict[str, Any], deleted: Iterable[str]) -> str:
    """The first parse error as a reason. A dangling reference, the usual cause after a
    delete or a hand edit removing an element, is phrased as which line still uses what."""
    where = f"{error['file']}:{error['line'] + 1}"
    # LikeC4 reports a dangling reference as "Could not resolve reference to ... named 'x'".
    named = re.search(r"Could not resolve reference to .* named '([^']+)'", error["message"])
    if deleted := sorted(deleted):
        use = f"still refers to '{named[1]}'" if named else f"still needs it ({error['message']})"
        return f"Cannot delete {', '.join(deleted)}: {where} {use}. Remove that reference first."
    if named:
        return f"Cannot save: {where} refers to '{named[1]}', which does not exist. Fix or remove that reference first."
    return f"{where}: {error['message']}"


async def _write(
    slug: str,
    docs: DataStore,
    ops: list[dict[str, Any]],
    *,
    file: str | None = None,
    edit: Callable[[dict[str, str]], dict[str, str]] | None = None,
    must_parse: bool = False,
    projects: ProjectStore | None = None,
) -> WorkspaceResponse:
    """Runs *ops* over the workspace's sources, or over ``edit(sources)`` (hand edits,
    deletions), and commits the outcome; new top-level elements go into *file*.

    With *must_parse*, any parse error refuses the change: hand edits must leave a
    model the diagram can show. Operations only refuse errors they introduce.
    With *projects*, an empty workspace is accepted when *slug* is one of the user's projects.
    """
    store = ArchitectureStore.for_project(docs, slug)
    if file is not None:
        store.source_key(file)
    for attempt in range(_COMMIT_ATTEMPTS):
        snapshot = store.read(allow_empty=projects is not None)
        if projects is not None and not snapshot.sources:
            _require_project(projects, slug)
        sources = edit(snapshot.sources) if edit else snapshot.sources
        deleted = set(snapshot.sources) - set(sources)
        result = await c4_service.apply_operations(sources, snapshot.snapshots, ops, file)
        if must_parse and (errors := result["model"]["errors"]):
            raise ArchitectureError(_refusal(errors[0], deleted))
        try:
            store.commit(snapshot, result["sources"], result["snapshots"], deleted=deleted)
            return WorkspaceResponse(model=result["model"], created=result["created"])
        except StorageConflictError:
            if attempt == _COMMIT_ATTEMPTS - 1:
                raise


@router.get("/{slug}", response_model=WorkspaceResponse)
async def read_workspace(slug: str, docs: DataStore = Depends(get_document_store), projects: ProjectStore = Depends(get_project_store)) -> WorkspaceResponse:
    """The whole model; a project without an architecture yet has an empty one."""
    _require_project(projects, slug)
    snapshot = ArchitectureStore.for_project(docs, slug).read(allow_empty=True)
    return WorkspaceResponse(model=(await c4_service.render(snapshot.sources, snapshot.snapshots))["model"])


@router.post("/{slug}/ops", response_model=WorkspaceResponse)
async def apply_operations(
    slug: str,
    req: OperationsRequest,
    docs: DataStore = Depends(get_document_store),
    projects: ProjectStore = Depends(get_project_store),
) -> WorkspaceResponse:
    """Model, view and layout edits (``{"op": "layout", ...}``), applied all or nothing. New files come
    from ``createPackage`` (``<id>/<id>.c4``: one system and its view; the first also declares the
    element kinds), ``createModule`` (beside its package, extending it with one container) and
    ``createView`` (an empty ``views/<id>.c4``); ``created`` holds their paths."""
    return await _write(slug, docs, req.ops, file=req.file, projects=projects)


@router.put("/{slug}/source", response_model=WorkspaceResponse)
async def write_source(slug: str, file: SourceFile, docs: DataStore = Depends(get_document_store)) -> WorkspaceResponse:
    """Hand edits must parse cleanly; the model is never saved in a state the diagram cannot show.
    Connections and view entries left naming an element the edit removed are dropped with it."""
    return await _write(slug, docs, [{"op": "pruneDangling"}], edit=lambda sources: {**sources, file.path: file.content}, must_parse=True)


@router.post("/{slug}/source", response_model=WorkspaceResponse)
async def create_source(
    slug: str,
    file: NewSourceFile,
    docs: DataStore = Depends(get_document_store),
    projects: ProjectStore = Depends(get_project_store),
) -> WorkspaceResponse:
    """Add a source file as written (empty by default); it must parse."""

    def add(sources: dict[str, str]) -> dict[str, str]:
        if file.path in sources:
            raise ArchitectureError(f"{file.path} already exists")
        return {**sources, file.path: file.content}

    return await _write(slug, docs, [], edit=add, must_parse=True, projects=projects)


@router.delete("/{slug}/source", response_model=WorkspaceResponse)
async def delete_source(slug: str, path: list[str] = Query(...), docs: DataStore = Depends(get_document_store)) -> WorkspaceResponse:
    """Deletes every ``path`` together (a package with its modules), refused while
    the remaining files still need them, e.g. they use their elements or kinds.

    The layouts of views they declared go with them. Deleting the last file
    leaves an empty model.
    """

    def drop(sources: dict[str, str]) -> dict[str, str]:
        if missing := sorted(set(path) - set(sources)):
            raise ArchitectureNotFound(f"Source file not found: {missing[0]}")
        return {name: text for name, text in sources.items() if name not in path}

    return await _write(slug, docs, [], edit=drop, must_parse=True)
