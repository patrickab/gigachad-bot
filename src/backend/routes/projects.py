import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import DIRECTORY_CHAT_HISTORIES
from backend.routes.files import delete_chat_upload_dir
from backend.routes.histories import SaveRequest, _load_chat_file
from lib.json_io import safe_read_json
from lib.payload import _build_payload
from lib.chat_index import invalidate_chat_id_index

router = APIRouter(prefix="/api/projects", tags=["projects"])

PROJECT_JSON = "project.json"
META_JSON = "projects-meta.json"


def _slugify(name: str) -> str:
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-") or "project"
    return slug[:64]


def _meta_path() -> Path:
    return DIRECTORY_CHAT_HISTORIES / META_JSON


def _read_meta() -> dict[str, Any]:
    return safe_read_json(_meta_path(), {"projects": []})


def _write_meta(meta: dict[str, Any]) -> None:
    DIRECTORY_CHAT_HISTORIES.mkdir(parents=True, exist_ok=True)
    _meta_path().write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_entry(meta: dict[str, Any], slug: str) -> dict[str, Any] | None:
    for p in meta.get("projects", []):
        if p.get("slug") == slug:
            return p
    return None


def _unique_slug(meta: dict[str, Any], base: str) -> str:
    existing = {p.get("slug") for p in meta.get("projects", [])}
    slug = base
    i = 2
    while slug in existing:
        slug = f"{base}-{i}"
        i += 1
    return slug


def _resolve_project_dir(slug: str) -> Path:
    project_dir = (DIRECTORY_CHAT_HISTORIES / slug).resolve()
    if not str(project_dir).startswith(str(DIRECTORY_CHAT_HISTORIES.resolve())):
        raise HTTPException(status_code=400, detail="Invalid project slug")
    return project_dir


def _read_project(project_dir: Path) -> dict[str, Any]:
    return safe_read_json(
        project_dir / PROJECT_JSON,
        {"name": project_dir.name, "kanban": [], "tabs": []},
    )


def _write_project(project_dir: Path, data: dict[str, Any]) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / PROJECT_JSON).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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


class SaveTabRequest(SaveRequest):
    filename: str
    tab_name: str | None = None


class UpdateProjectRequest(BaseModel):
    name: str | None = None


class ProjectStateModel(BaseModel):
    kanban: list[KanbanCardModel] = []
    tabs: list[ProjectTabModel] = []


@router.get("")
async def list_projects() -> dict[str, Any]:
    meta = _read_meta()
    projects: list[dict[str, Any]] = []
    for p in meta.get("projects", []):
        project_dir = _resolve_project_dir(p["slug"])
        data = _read_project(project_dir)
        projects.append(
            {
                "name": p["name"],
                "slug": p["slug"],
                "tabs": data.get("tabs", []),
            }
        )
    return {"projects": projects}


@router.get("/{slug}")
async def get_project(slug: str) -> dict[str, Any]:
    meta = _read_meta()
    entry = _find_entry(meta, slug)
    if not entry:
        raise HTTPException(status_code=404, detail="Project not found")
    project_dir = _resolve_project_dir(slug)
    data = _read_project(project_dir)
    return {"name": entry["name"], "slug": entry["slug"], "kanban": data.get("kanban", []), "tabs": data.get("tabs", [])}


@router.post("")
async def create_project(req: CreateProjectRequest) -> dict[str, Any]:
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name required")
    meta = _read_meta()
    slug = _unique_slug(meta, _slugify(name))
    project_dir = _resolve_project_dir(slug)
    if project_dir.exists():
        raise HTTPException(status_code=409, detail="Project already exists")
    now = _now_iso()
    entry = {"name": name, "slug": slug, "createdAt": now, "updatedAt": now}
    meta.setdefault("projects", []).append(entry)
    _write_meta(meta)
    _write_project(project_dir, {"name": name, "kanban": [], "tabs": []})
    return {"name": name, "slug": slug, "kanban": [], "tabs": []}


@router.patch("/{slug}")
async def update_project(slug: str, req: UpdateProjectRequest) -> dict[str, Any]:
    meta = _read_meta()
    entry = _find_entry(meta, slug)
    if not entry:
        raise HTTPException(status_code=404, detail="Project not found")
    if req.name is not None:
        name = req.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Project name required")
        entry["name"] = name
        entry["updatedAt"] = _now_iso()
        project_dir = _resolve_project_dir(slug)
        data = _read_project(project_dir)
        data["name"] = name
        _write_project(project_dir, data)
    _write_meta(meta)
    return {"name": entry["name"], "slug": entry["slug"]}


@router.delete("/{slug}")
async def delete_project(slug: str) -> dict[str, str]:
    meta = _read_meta()
    entry = _find_entry(meta, slug)
    if not entry:
        raise HTTPException(status_code=404, detail="Project not found")
    project_dir = _resolve_project_dir(slug)
    if project_dir.exists():
        shutil.rmtree(project_dir)
    meta["projects"] = [p for p in meta.get("projects", []) if p.get("slug") != slug]
    _write_meta(meta)
    return {"status": "ok"}


@router.put("/{slug}/state")
async def update_project_state(slug: str, state: ProjectStateModel) -> dict[str, Any]:
    meta = _read_meta()
    entry = _find_entry(meta, slug)
    if not entry:
        raise HTTPException(status_code=404, detail="Project not found")
    project_dir = _resolve_project_dir(slug)
    data = _read_project(project_dir)
    data["kanban"] = [c.model_dump() for c in state.kanban]
    seen = set()
    deduped = []
    for t in state.tabs:
        d = t.model_dump()
        if d.get("filename") not in seen:
            seen.add(d.get("filename"))
            deduped.append(d)
    data["tabs"] = deduped
    _write_project(project_dir, data)
    return {"name": entry["name"], "slug": entry["slug"], "kanban": data["kanban"], "tabs": data["tabs"]}


@router.post("/{slug}/cards")
async def add_card(slug: str, req: AddCardRequest) -> dict[str, Any]:
    meta = _read_meta()
    if not _find_entry(meta, slug):
        raise HTTPException(status_code=404, detail="Project not found")
    project_dir = _resolve_project_dir(slug)
    data = _read_project(project_dir)
    card = {"id": str(uuid.uuid4()), "title": req.title, "description": req.description, "state": req.state}
    data.setdefault("kanban", []).append(card)
    _write_project(project_dir, data)
    return card


@router.patch("/{slug}/cards/{card_id}")
async def move_card(slug: str, card_id: str, req: MoveCardRequest | UpdateCardRequest | None = None) -> dict[str, Any]:
    meta = _read_meta()
    if not _find_entry(meta, slug):
        raise HTTPException(status_code=404, detail="Project not found")
    project_dir = _resolve_project_dir(slug)
    data = _read_project(project_dir)
    for card in data.get("kanban", []):
        if card["id"] == card_id:
            if req is not None:
                if isinstance(req, MoveCardRequest):
                    card["state"] = req.state
                elif isinstance(req, UpdateCardRequest):
                    if req.title is not None:
                        card["title"] = req.title
                    if req.description is not None:
                        card["description"] = req.description
            _write_project(project_dir, data)
            return card
    raise HTTPException(status_code=404, detail="Card not found")


@router.delete("/{slug}/cards/{card_id}")
async def delete_card(slug: str, card_id: str) -> dict[str, str]:
    meta = _read_meta()
    if not _find_entry(meta, slug):
        raise HTTPException(status_code=404, detail="Project not found")
    project_dir = _resolve_project_dir(slug)
    data = _read_project(project_dir)
    data["kanban"] = [c for c in data.get("kanban", []) if c["id"] != card_id]
    _write_project(project_dir, data)
    return {"status": "ok"}


@router.put("/{slug}/tabs/{filename:path}")
async def save_project_tab(slug: str, filename: str, data: SaveTabRequest) -> dict[str, str]:
    meta = _read_meta()
    if not _find_entry(meta, slug):
        raise HTTPException(status_code=404, detail="Project not found")
    project_dir = _resolve_project_dir(slug)
    tab_path = project_dir / filename
    tab_path.parent.mkdir(parents=True, exist_ok=True)

    # Prevent cross-chat overwrites: if the file exists and has a chat_id,
    # the incoming chat_id must match.
    existing = _load_chat_file(tab_path) if tab_path.exists() else None
    if existing and existing.get("chat_id") and data.chat_id:
        if existing["chat_id"] != data.chat_id:
            raise HTTPException(
                status_code=409,
                detail="Chat ID mismatch: cannot overwrite an existing project tab with a different chat_id",
            )

    payload = _build_payload(data.model_dump() if data else None, existing)
    tab_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    invalidate_chat_id_index()
    project_data = _read_project(project_dir)
    tabs = project_data.get("tabs", [])
    existing = next((t for t in tabs if t["filename"] == filename), None)
    if existing:
        if data.tab_name is not None:
            existing["name"] = data.tab_name
        if data.title is not None:
            existing["title"] = data.title
    else:
        tabs.append({"filename": filename, "name": data.tab_name, "title": data.title})
        project_data["tabs"] = tabs
    _write_project(project_dir, project_data)
    return {"status": "ok"}


@router.delete("/{slug}/tabs/{filename:path}")
async def delete_project_tab(slug: str, filename: str) -> dict[str, str]:
    meta = _read_meta()
    if not _find_entry(meta, slug):
        raise HTTPException(status_code=404, detail="Project not found")
    project_dir = _resolve_project_dir(slug)
    tab_path = project_dir / filename
    chat_id_to_clean: str | None = None
    if tab_path.exists():
        raw = safe_read_json(tab_path, {})
        cid = raw.get("chat_id")
        if isinstance(cid, str):
            chat_id_to_clean = cid
        tab_path.unlink()
    invalidate_chat_id_index()
    if chat_id_to_clean:
        delete_chat_upload_dir(chat_id_to_clean, slug)
    project_data = _read_project(project_dir)
    project_data["tabs"] = [t for t in project_data.get("tabs", []) if t["filename"] != filename]
    _write_project(project_dir, project_data)
    return {"status": "ok"}
