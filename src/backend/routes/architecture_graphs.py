"""HTTP boundary for live Architecture Graph YAML documents."""

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import BaseModel, Field

from backend.routes.deps import get_architecture_graph_store, get_project_store
from lib.architecture_graph import ArchitectureGraphNotFound, ArchitectureGraphStore
from lib.document_library import document_meta
from lib.project_store import ProjectStore

router = APIRouter(prefix="/api/architecture-graphs", tags=["architecture-graphs"])


class ArchitectureGraphContextReferenceModel(BaseModel):
    """A live reference to a canonical Architecture Graph, not a copied attachment."""

    path: str = Field(min_length=1)


class GraphContentRequest(BaseModel):
    content: str


class CreateGraphRequest(GraphContentRequest):
    name: str
    projectSlug: str | None = None


class GraphResponse(BaseModel):
    name: str
    path: str
    content: str
    hasDraft: bool
    revision: str


@contextmanager
def _project_lookup() -> Iterator[None]:
    """Restate ProjectStore's unknown-slug FileNotFoundError in the graph error vocabulary."""
    try:
        yield
    except ArchitectureGraphNotFound:
        raise
    except FileNotFoundError as exc:
        raise ArchitectureGraphNotFound(str(exc)) from exc


def _read_response(
    store: ArchitectureGraphStore, name: str, *, draft: bool = False, response: Response | None = None
) -> GraphResponse:
    content, revision = store.read_with_revision(name, draft=draft)
    if response is not None:
        response.headers["ETag"] = revision.token
    return GraphResponse(
        name=name,
        path=store.path_for(name, draft=draft),
        content=content,
        hasDraft=store.has_draft(name),
        revision=revision.token,
    )


def _expected_revision(store: ArchitectureGraphStore, name: str, if_match: str | None, *, draft: bool) -> str | None:
    """Resolve the revision a write must match, refusing a blind overwrite."""
    try:
        _, revision = store.read_with_revision(name, draft=draft)
    except ArchitectureGraphNotFound:
        return None
    if if_match is None:
        raise HTTPException(status_code=428, detail="If-Match is required to overwrite an existing Architecture Graph")
    return revision.token if if_match == "*" else if_match.strip('"')


@router.get("")
async def list_graphs(store: ArchitectureGraphStore = Depends(get_architecture_graph_store)) -> dict[str, list[dict[str, str]]]:
    return {"graphs": [document_meta(path) for path in store.list_paths()]}


@router.post("", response_model=GraphResponse)
async def create_graph(
    req: CreateGraphRequest,
    response: Response,
    store: ArchitectureGraphStore = Depends(get_architecture_graph_store),
    projects: ProjectStore = Depends(get_project_store),
) -> GraphResponse:
    with _project_lookup():
        # Reject an unknown slug before writing, so a bad association leaves no orphan file.
        if req.projectSlug:
            projects.list_files(req.projectSlug)
        path = store.write(req.name, req.content)
        if req.projectSlug:
            projects.add_file(req.projectSlug, path)
    return _read_response(store, req.name, response=response)


@router.get("/{name}", response_model=GraphResponse)
async def read_graph(
    name: str, response: Response, store: ArchitectureGraphStore = Depends(get_architecture_graph_store)
) -> GraphResponse:
    return _read_response(store, name, response=response)


@router.put("/{name}", response_model=GraphResponse)
async def write_graph(
    name: str,
    req: GraphContentRequest,
    response: Response,
    store: ArchitectureGraphStore = Depends(get_architecture_graph_store),
    if_match: str | None = Header(default=None, alias="If-Match"),
) -> GraphResponse:
    store.write(name, req.content, expected=_expected_revision(store, name, if_match, draft=False))
    return _read_response(store, name, response=response)


@router.post("/{name}/projects/{slug}")
async def associate_graph(
    name: str,
    slug: str,
    store: ArchitectureGraphStore = Depends(get_architecture_graph_store),
    projects: ProjectStore = Depends(get_project_store),
) -> dict[str, list[str]]:
    store.read(name)
    with _project_lookup():
        return {"files": projects.add_file(slug, store.path_for(name))}


@router.delete("/{name}/projects/{slug}")
async def unassociate_graph(
    name: str,
    slug: str,
    store: ArchitectureGraphStore = Depends(get_architecture_graph_store),
    projects: ProjectStore = Depends(get_project_store),
) -> dict[str, list[str]]:
    with _project_lookup():
        return {"files": projects.remove_file(slug, store.path_for(name))}


@router.get("/{name}/draft", response_model=GraphResponse)
async def read_draft(
    name: str, response: Response, store: ArchitectureGraphStore = Depends(get_architecture_graph_store)
) -> GraphResponse:
    return _read_response(store, name, draft=True, response=response)


@router.put("/{name}/draft", response_model=GraphResponse)
async def write_draft(
    name: str,
    req: GraphContentRequest,
    response: Response,
    store: ArchitectureGraphStore = Depends(get_architecture_graph_store),
    if_match: str | None = Header(default=None, alias="If-Match"),
) -> GraphResponse:
    store.write(name, req.content, draft=True, expected=_expected_revision(store, name, if_match, draft=True))
    return _read_response(store, name, draft=True, response=response)


@router.post("/{name}/draft/accept", response_model=GraphResponse)
async def accept_draft(name: str, store: ArchitectureGraphStore = Depends(get_architecture_graph_store)) -> GraphResponse:
    store.accept_draft(name)
    return _read_response(store, name)


@router.delete("/{name}/draft")
async def discard_draft(name: str, store: ArchitectureGraphStore = Depends(get_architecture_graph_store)) -> dict[str, str]:
    store.discard_draft(name)
    return {"status": "ok"}
