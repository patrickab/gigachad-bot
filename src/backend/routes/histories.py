import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import DIRECTORY_CHAT_HISTORIES, chat_upload_dir
from backend.routes.files import delete_chat_upload_dir
from lib.json_io import safe_read_json

router = APIRouter(prefix="/api/chat-histories", tags=["histories"])


class SaveRequest(BaseModel):
    messages: list[dict[str, Any]] = []
    chat_id: str | None = None
    title: str | None = None


class RenameRequest(BaseModel):
    old_path: str
    new_title: str


PROJECT_JSON = "project.json"
META_JSON = "projects-meta.json"


def _sanitize_title(title: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._\- ]+", "_", title).strip()
    return sanitized or "untitled"


def _unique_filename(directory: Path, base: str) -> str:
    """Return a unique filename in `directory` for a given base like 'foo.json'.
    Disambiguates with '-2', '-3', ... suffixes.
    """
    target = directory / base
    if not target.exists():
        return base
    stem = Path(base).stem
    suffix = Path(base).suffix
    n = 2
    while True:
        candidate = f"{stem}-{n}{suffix}"
        if not (directory / candidate).exists():
            return candidate
        n += 1


def _project_dir_for_path(path: Path) -> Path | None:
    """Return the project directory containing `path`, or None if at root."""
    try:
        rel = path.relative_to(DIRECTORY_CHAT_HISTORIES.resolve())
    except ValueError:
        return None
    if len(rel.parts) < 2:
        return None
    candidate = DIRECTORY_CHAT_HISTORIES / rel.parts[0]
    if candidate.is_dir() and (candidate / PROJECT_JSON).exists():
        return candidate
    return None


def _load_chat_file(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(raw, list):
        return {"messages": raw, "chat_id": None, "title": None}
    return {
        "messages": raw.get("messages", []),
        "chat_id": raw.get("chat_id"),
        "title": raw.get("title"),
    }


@router.get("")
async def list_chat_histories() -> dict[str, Any]:
    if not DIRECTORY_CHAT_HISTORIES.exists():
        return {"histories": {}}

    histories: dict[str, list[str]] = {}

    if DIRECTORY_CHAT_HISTORIES.is_dir():
        for f in sorted(DIRECTORY_CHAT_HISTORIES.iterdir()):
            if f.name in (PROJECT_JSON, META_JSON):
                continue
            if f.name == "_uploads" and f.is_dir():
                continue
            if f.is_file() and f.suffix == ".json" and f.name != "archived":
                histories.setdefault("root", []).append(f.name)
            elif f.is_dir() and (f / PROJECT_JSON).is_file() and f.name not in ("_uploads", "archived"):
                for sf in sorted(f.iterdir()):
                    if sf.is_file() and sf.suffix == ".json" and sf.name != PROJECT_JSON:
                        histories.setdefault(f.name, []).append(sf.name)

    return {"histories": histories}


@router.get("/{filename:path}")
async def load_chat_history(filename: str) -> dict[str, Any]:
    filepath = _resolve_history_path(filename)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat history not found")
    data = _load_chat_file(filepath)
    if data is None:
        raise HTTPException(status_code=500, detail="Failed to read chat history")
    return {**data, "filename": filename}


@router.put("/{filename:path}")
async def save_chat_history(filename: str, data: SaveRequest | None = None) -> dict[str, str]:
    DIRECTORY_CHAT_HISTORIES.mkdir(parents=True, exist_ok=True)
    filepath = _resolve_history_path(filename)
    payload: dict[str, Any] = {"messages": data.messages if data else []}
    if data and data.chat_id:
        payload["chat_id"] = data.chat_id
    if data and data.title:
        payload["title"] = data.title
    filepath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"status": "ok", "filename": str(filename)}


@router.delete("/{filename:path}")
async def delete_chat_history(filename: str) -> dict[str, str]:
    filepath = _resolve_history_path(filename)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat history not found")
    data = _load_chat_file(filepath)
    project_dir = _project_dir_for_path(filepath)
    slug = project_dir.name if project_dir else None
    if data and data.get("chat_id"):
        delete_chat_upload_dir(data["chat_id"], slug)
    filepath.unlink()
    if project_dir is not None:
        proj_data_path = project_dir / PROJECT_JSON
        proj_data = safe_read_json(proj_data_path, {"name": slug, "kanban": [], "tabs": []})
        proj_data["tabs"] = [t for t in proj_data.get("tabs", []) if t.get("filename") != filepath.name]
        proj_data_path.write_text(json.dumps(proj_data, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": "ok"}


@router.post("/rename")
async def rename_chat_history(req: RenameRequest) -> dict[str, str]:
    old_path = _resolve_history_path(req.old_path)
    if not old_path.exists():
        raise HTTPException(status_code=404, detail="Chat history not found")
    project_dir = _project_dir_for_path(old_path)
    new_title = _sanitize_title(req.new_title)
    new_filename = f"{new_title}.json"
    if project_dir is not None:
        new_dir = project_dir
    else:
        new_dir = DIRECTORY_CHAT_HISTORIES
    unique_name = _unique_filename(new_dir, new_filename)
    new_path = new_dir / unique_name
    data = _load_chat_file(old_path) or {}
    new_payload: dict[str, Any] = {"messages": data.get("messages", []), "title": new_title}
    if data.get("chat_id"):
        new_payload["chat_id"] = data["chat_id"]
    new_path.write_text(json.dumps(new_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    old_path.unlink()
    if project_dir is not None:
        proj_data_path = project_dir / PROJECT_JSON
        proj_data = safe_read_json(proj_data_path, {"name": project_dir.name, "kanban": [], "tabs": []})
        for t in proj_data.get("tabs", []):
            if t.get("filename") == old_path.name:
                t["filename"] = unique_name
                t["name"] = new_title
                t["title"] = new_title
        proj_data_path.write_text(json.dumps(proj_data, indent=2, ensure_ascii=False), encoding="utf-8")
    rel = new_path.relative_to(DIRECTORY_CHAT_HISTORIES)
    return {"status": "ok", "new_path": str(rel), "filename": unique_name}


@router.put("/{filename:path}/archive")
async def archive_chat_history(filename: str) -> dict[str, str]:
    filepath = _resolve_history_path(filename)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat history not found")
    archive_dir = DIRECTORY_CHAT_HISTORIES / "archived"
    archive_dir.mkdir(parents=True, exist_ok=True)
    filepath.rename(archive_dir / filepath.name)
    return {"status": "ok"}


@router.post("/{filename:path}/unarchive")
async def unarchive_chat_history(filename: str) -> dict[str, str]:
    archive_dir = DIRECTORY_CHAT_HISTORIES / "archived"
    filepath = (archive_dir / filename).resolve()
    if not str(filepath).startswith(str(archive_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Unreachable history")
    filepath.rename(DIRECTORY_CHAT_HISTORIES / filepath.name)
    return {"status": "ok"}


def _resolve_history_path(filename: str) -> Path:
    resolved = (DIRECTORY_CHAT_HISTORIES / filename).resolve()
    if not str(resolved).startswith(str(DIRECTORY_CHAT_HISTORIES.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    return resolved
