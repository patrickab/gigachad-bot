import asyncio
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.http_plumbing import http_errors
from backend.routes.deps import get_chat_cleanup, get_project_store, get_sandbox_service
from backend.routes.schemas import ChatSnapshot
from lib.chat_cleanup import ChatCleanup
from lib.data_store import StorageConflictError
from lib.project_store import ProjectExists, ProjectStore
from lib.sandbox_service import SandboxService

router = APIRouter(prefix="/api/projects", tags=["projects"])

_PROJECT_NOT_FOUND = (FileNotFoundError, 404, "Project not found")


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


class SaveTabRequest(ChatSnapshot):
    filename: str
    tab_name: str | None = None


class UpdateProjectRequest(BaseModel):
    name: str | None = None
    archived: bool | None = None


# Store calls block on Postgres, so handlers are sync (FastAPI's threadpool) or move the call to a thread:
# a blocking call on the event loop would stall every concurrent request, SSE streams included.
@router.get("")
def list_projects(store: ProjectStore = Depends(get_project_store)) -> dict[str, Any]:
    return {"projects": store.list_projects()}


@router.get("/{slug}")
def get_project(slug: str, store: ProjectStore = Depends(get_project_store)) -> dict[str, Any]:
    with http_errors(_PROJECT_NOT_FOUND):
        return store.get_project(slug)


@router.post("")
def create_project(req: CreateProjectRequest, store: ProjectStore = Depends(get_project_store)) -> dict[str, Any]:
    with http_errors((ProjectExists, 409, None), (ValueError, 400, None)):
        return store.create_project(req.name)


@router.patch("/{slug}")
def update_project(slug: str, req: UpdateProjectRequest, store: ProjectStore = Depends(get_project_store)) -> dict[str, Any]:
    with http_errors((StorageConflictError, 409, None), _PROJECT_NOT_FOUND, (ValueError, 400, None)):
        return store.update_project(slug, name=req.name, archived=req.archived)


@router.delete("/{slug}")
async def delete_project(
    slug: str,
    store: ProjectStore = Depends(get_project_store),
    cleanup: ChatCleanup = Depends(get_chat_cleanup),
) -> dict[str, str]:
    with http_errors(_PROJECT_NOT_FOUND):
        return await cleanup.after(lambda released: store.delete_project(slug, released=released))


@router.post("/{slug}/cards")
def add_card(slug: str, req: AddCardRequest, store: ProjectStore = Depends(get_project_store)) -> dict[str, Any]:
    with http_errors(_PROJECT_NOT_FOUND):
        return store.add_card(slug, req.title, req.description, req.state)


@router.patch("/{slug}/cards/{card_id}")
def move_card(
    slug: str,
    card_id: str,
    req: MoveCardRequest | UpdateCardRequest | None = None,
    store: ProjectStore = Depends(get_project_store),
) -> dict[str, Any]:
    """Move a card (body `state`) or edit it (body `title`/`description`), omitted fields stay unchanged."""
    with http_errors((FileNotFoundError, 404, None)):
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


@router.delete("/{slug}/cards/{card_id}")
def delete_card(slug: str, card_id: str, store: ProjectStore = Depends(get_project_store)) -> dict[str, str]:
    with http_errors(_PROJECT_NOT_FOUND):
        return store.delete_card(slug, card_id)


@router.put("/{slug}/tabs/{filename:path}")
async def save_project_tab(
    slug: str,
    filename: str,
    data: SaveTabRequest,
    store: ProjectStore = Depends(get_project_store),
    sandbox_service: SandboxService = Depends(get_sandbox_service),
) -> dict[str, str]:
    with http_errors(_PROJECT_NOT_FOUND, (ValueError, 409, None)):
        result = await asyncio.to_thread(store.save_tab, slug, filename, data.model_dump())
        if data.chat_id:
            await sandbox_service.checkpoint(chat_id=data.chat_id)
        return result
