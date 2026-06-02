import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import DIRECTORY_CHAT_HISTORIES
from backend.routes.files import delete_chat_upload_dir

router = APIRouter(prefix="/api/chat-histories", tags=["histories"])


class SaveRequest(BaseModel):
    filename: str | None = None
    messages: list[dict[str, Any]] = []
    chat_id: str | None = None


@router.get("")
async def list_chat_histories() -> dict[str, Any]:
    if not DIRECTORY_CHAT_HISTORIES.exists():
        return {"histories": {}}

    histories: dict[str, list[str]] = {}
    for f in sorted(DIRECTORY_CHAT_HISTORIES.rglob("*.json")):
        if f.is_file() and "archived" not in f.parts:
            dir_key = str(f.parent.relative_to(DIRECTORY_CHAT_HISTORIES)) if f.parent != DIRECTORY_CHAT_HISTORIES else "root"
            histories.setdefault(dir_key, []).append(f.name)
    return {"histories": histories}


@router.get("/{filename:path}")
async def load_chat_history(filename: str) -> dict[str, Any]:
    filepath = _resolve_history_path(filename)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat history not found")
    raw = json.loads(filepath.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {"messages": raw, "filename": filename, "chat_id": None}
    return {"messages": raw.get("messages", []), "filename": filename, "chat_id": raw.get("chat_id")}


@router.put("/{filename:path}")
async def save_chat_history(filename: str, data: SaveRequest | None = None) -> dict[str, str]:
    DIRECTORY_CHAT_HISTORIES.mkdir(parents=True, exist_ok=True)
    name = filename if not data or not data.filename else data.filename
    filepath = _resolve_history_path(name)
    if data and data.chat_id:
        payload = {"chat_id": data.chat_id, "messages": data.messages}
    else:
        payload = data.messages if data else []
    filepath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"status": "ok", "filename": str(name)}


@router.delete("/{filename:path}")
async def delete_chat_history(filename: str) -> dict[str, str]:
    filepath = _resolve_history_path(filename)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat history not found")
    try:
        raw = json.loads(filepath.read_text(encoding="utf-8"))
        chat_id = raw.get("chat_id") if isinstance(raw, dict) else None
    except Exception:
        chat_id = None
    filepath.unlink()
    if chat_id:
        delete_chat_upload_dir(chat_id)
    return {"status": "ok"}


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