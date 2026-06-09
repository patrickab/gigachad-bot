from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.routes.deps import get_project_store
from backend.routes.files import delete_chat_upload_dir
from lib.project_store import ProjectStore

router = APIRouter(prefix="/api/projects", tags=["projects"])


class KanbanCardModel(BaseModel):
    id: str
    title: str
    description: str = ""
    state: str = "backlog"


class ProjectTabModel(BaseModel):
    filename: str
    name: str | None = None
    title: str | None = None


class ProjectDataModel(BaseModel):
    name: str
    kanban: list[KanbanCardModel] = []
    tabs: list[ProjectTabModel] = []


class CreateProjectRequest(BaseModel):
    name: str


class AddCardRequest(BaseModel):
    title: str
    description: str = ""
    state: str = "backlog"


class MoveCardRequest(BaseModel):
    state: str


class UpdateCardRequest(BaseModel):
    title: str | None = None
    description: str | None = None


class SaveTabRequest(BaseModel):
    messages: list[dict[str, Any]] = []
    chat_id: str | None = None
    title: str | None = None
    usage: dict[str, int] | None = None
    parent_id: str | None = None
    branch_message_idx: int | None = None
    children: list[dict[str, Any]] | None = None
    filename: str
    tab_name: str | None = None


class UpdateProjectRequest(BaseModel):
    name: str | None = None


class ProjectStateModel(BaseModel):
    kanban: list[KanbanCardModel] = []
    tabs: list[ProjectTabModel] = []


def _not_found(detail: str = "Project not found") -> HTTPException:
    return HTTPException(status_code=404, detail=detail)


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _conflict(detail: str) -> HTTPException:
    return HTTPException(status_code=409, detail=detail)


@router.get("")
async def list_projects(store: ProjectStore = Depends(get_project_store)) -> dict[str, Any]:
    return {"projects": store.list_projects()}


@router.get("/{slug}")
async def get_project(slug: str, store: ProjectStore = Depends(get_project_store)) -> dict[str, Any]:
    try:
        return store.get_project(slug)
    except FileNotFoundError:
        raise _not_found()


@router.post("")
async def create_project(req: CreateProjectRequest, store: ProjectStore = Depends(get_project_store)) -> dict[str, Any]:
    try:
        return store.create_project(req.name)
    except ValueError as e:
        if "already exists" in str(e):
            raise _conflict(str(e))
        raise _bad_request(str(e))


@router.patch("/{slug}")
async def update_project(slug: str, req: UpdateProjectRequest, store: ProjectStore = Depends(get_project_store)) -> dict[str, Any]:
    try:
        return store.update_project(slug, name=req.name)
    except FileNotFoundError:
        raise _not_found()
    except ValueError as e:
        raise _bad_request(str(e))


@router.delete("/{slug}")
async def delete_project(slug: str, store: ProjectStore = Depends(get_project_store)) -> dict[str, str]:
    try:
        return store.delete_project(slug)
    except FileNotFoundError:
        raise _not_found()


@router.put("/{slug}/state")
async def update_project_state(
    slug: str,
    state: ProjectStateModel,
    store: ProjectStore = Depends(get_project_store),
) -> dict[str, Any]:
    try:
        return store.update_project_state(slug, [c.model_dump() for c in state.kanban], [t.model_dump() for t in state.tabs])
    except FileNotFoundError:
        raise _not_found()


@router.post("/{slug}/cards")
async def add_card(
    slug: str, req: AddCardRequest, store: ProjectStore = Depends(get_project_store)
) -> dict[str, Any]:
    try:
        return store.add_card(slug, req.title, req.description, req.state)
    except FileNotFoundError:
        raise _not_found()


@router.patch("/{slug}/cards/{card_id}")
async def move_card(
    slug: str,
    card_id: str,
    req: MoveCardRequest | UpdateCardRequest | None = None,
    store: ProjectStore = Depends(get_project_store),
) -> dict[str, Any]:
    try:
        updates: dict[str, Any] = {}
        if req is not None:
            if isinstance(req, MoveCardRequest):
                updates["state"] = req.state
            elif isinstance(req, UpdateCardRequest):
                if req.title is not None:
                    updates["title"] = req.title
                if req.description is not None:
                    updates["description"] = req.description
        return store.update_card(slug, card_id, updates)
    except FileNotFoundError as e:
        raise _not_found(str(e))


@router.delete("/{slug}/cards/{card_id}")
async def delete_card(slug: str, card_id: str, store: ProjectStore = Depends(get_project_store)) -> dict[str, str]:
    try:
        return store.delete_card(slug, card_id)
    except FileNotFoundError:
        raise _not_found()


@router.put("/{slug}/tabs/{filename:path}")
async def save_project_tab(
    slug: str,
    filename: str,
    data: SaveTabRequest,
    store: ProjectStore = Depends(get_project_store),
) -> dict[str, str]:
    try:
        return store.save_tab(slug, filename, data.model_dump())
    except FileNotFoundError:
        raise _not_found()
    except ValueError as e:
        raise _conflict(str(e))


@router.delete("/{slug}/tabs/{filename:path}")
async def delete_project_tab(slug: str, filename: str, store: ProjectStore = Depends(get_project_store)) -> dict[str, str]:
    try:
        return store.delete_tab(slug, filename, cleanup_uploads_fn=delete_chat_upload_dir)
    except FileNotFoundError:
        raise _not_found()