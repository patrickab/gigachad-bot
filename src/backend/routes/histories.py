import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import DIRECTORY_CHAT_HISTORIES

router = APIRouter(prefix="/api/chat-histories", tags=["histories"])


class SaveRequest(BaseModel):
    filename: str | None = None
    messages: list[dict[str, Any]] = []


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
    messages = json.loads(filepath.read_text(encoding="utf-8"))
    return {"messages": messages, "filename": filename}


@router.put("/{filename:path}")
async def save_chat_history(filename: str, data: SaveRequest | None = None) -> dict[str, str]:
    DIRECTORY_CHAT_HISTORIES.mkdir(parents=True, exist_ok=True)
    name = filename if not data or not data.filename else data.filename
    filepath = DIRECTORY_CHAT_HISTORIES / name
    filepath.write_text(json.dumps(data.messages if data else [], indent=2), encoding="utf-8")
    return {"status": "ok", "filename": str(name)}


@router.delete("/{filename:path}")
async def delete_chat_history(filename: str) -> dict[str, str]:
    filepath = _resolve_history_path(filename)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat history not found")
    filepath.unlink()
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
    filepath = archive_dir / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Unreachable history")
    filepath.rename(DIRECTORY_CHAT_HISTORIES / filepath.name)
    return {"status": "ok"}


def _resolve_history_path(filename: str) -> Path:
    return DIRECTORY_CHAT_HISTORIES / filename