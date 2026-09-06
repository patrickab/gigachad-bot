"""HTTP boundary for live Architecture Graph YAML documents."""

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import APIRouter, Depends
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


@contextmanager
def _project_lookup() -> Iterator[None]:
    """Restate ProjectStore's unknown-slug FileNotFoundError in the graph error vocabulary."""
    try:
        yield
    except ArchitectureGraphNotFound:
        raise
    except FileNotFoundError as exc:
        raise ArchitectureGraphNotFound(str(exc)) from exc


def _read_response(store: ArchitectureGraphStore, name: str, *, draft: bool = False) -> GraphResponse:
    content = store.read(name, draft=draft)
    path = store.path_for(name, draft=draft)
    return GraphResponse(name=name, path=path, content=content, hasDraft=store.has_draft(name))


@router.get("")
async def list_graphs(store: ArchitectureGraphStore = Depends(get_architecture_graph_store)) -> dict[str, list[dict[str, str]]]:
    return {"graphs": [document_meta(path) for path in store.list_paths()]}


@router.post("", response_model=GraphResponse)
async def create_graph(
    req: CreateGraphRequest,
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
    return _read_response(store, req.name)


@router.get("/{name}", response_model=GraphResponse)
async def read_graph(name: str, store: ArchitectureGraphStore = Depends(get_architecture_graph_store)) -> GraphResponse:
    return _read_response(store, name)


@router.put("/{name}", response_model=GraphResponse)
async def write_graph(
    name: str, req: GraphContentRequest, store: ArchitectureGraphStore = Depends(get_architecture_graph_store)
) -> GraphResponse:
    store.write(name, req.content)
    return _read_response(store, name)


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
async def read_draft(name: str, store: ArchitectureGraphStore = Depends(get_architecture_graph_store)) -> GraphResponse:
    return _read_response(store, name, draft=True)


@router.put("/{name}/draft", response_model=GraphResponse)
async def write_draft(
    name: str, req: GraphContentRequest, store: ArchitectureGraphStore = Depends(get_architecture_graph_store)
) -> GraphResponse:
    store.write(name, req.content, draft=True)
    return _read_response(store, name, draft=True)


@router.post("/{name}/draft/accept", response_model=GraphResponse)
async def accept_draft(name: str, store: ArchitectureGraphStore = Depends(get_architecture_graph_store)) -> GraphResponse:
    store.accept_draft(name)
    return _read_response(store, name)


@router.delete("/{name}/draft")
async def discard_draft(name: str, store: ArchitectureGraphStore = Depends(get_architecture_graph_store)) -> dict[str, str]:
    store.discard_draft(name)
    return {"status": "ok"}
