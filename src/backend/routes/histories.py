from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.routes.deps import get_chat_store
from backend.routes.files import delete_chat_upload_dir
from lib.chat_store import ChatStore

router = APIRouter(prefix="/api/chat-histories", tags=["histories"])


class SaveRequest(BaseModel):
    messages: list[dict[str, Any]] = []
    chat_id: str | None = None
    title: str | None = None
    usage: dict[str, int] | None = None
    parent_id: str | None = None
    branch_message_idx: int | None = None
    children: list[dict[str, Any]] | None = None


class RenameRequest(BaseModel):
    old_path: str
    new_title: str


class MkdirRequest(BaseModel):
    parent_path: str
    name: str


class MoveRequest(BaseModel):
    filename: str
    target_dir: str


class BranchRequest(BaseModel):
    parent_file: str
    branch_message_idx: int


class MergeRequest(BaseModel):
    child_file: str


@router.get("")
async def list_chat_histories(store: ChatStore = Depends(get_chat_store)) -> dict[str, Any]:
    return store.list_histories()


@router.get("/branch-meta")
async def get_branch_meta(dirs: str | None = None, store: ChatStore = Depends(get_chat_store)) -> dict[str, dict[str, Any]]:
    return store.get_branch_meta(dirs)


@router.get("/{filename:path}")
async def load_chat_history(filename: str, store: ChatStore = Depends(get_chat_store)) -> dict[str, Any]:
    data = store.load(filename)
    if data is None:
        raise HTTPException(status_code=404, detail="Chat history not found")
    return {**data, "filename": filename}


@router.put("/{filename:path}")
async def save_chat_history(
    filename: str,
    data: SaveRequest | None = None,
    store: ChatStore = Depends(get_chat_store),
) -> dict[str, str]:
    try:
        return store.save(filename, data.model_dump() if data else None)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/cascade/{filename:path}")
async def cascade_delete(filename: str, store: ChatStore = Depends(get_chat_store)) -> dict[str, Any]:
    try:
        return store.cascade_delete(filename, cleanup_uploads_fn=delete_chat_upload_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chat history not found")


@router.delete("/orphan/{filename:path}")
async def orphan_children(filename: str, store: ChatStore = Depends(get_chat_store)) -> dict[str, Any]:
    try:
        return store.orphan(filename, cleanup_uploads_fn=delete_chat_upload_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chat history not found")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{filename:path}")
async def delete_chat_history(filename: str, store: ChatStore = Depends(get_chat_store)) -> dict[str, str]:
    try:
        return store.delete(filename, cleanup_uploads_fn=delete_chat_upload_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chat history not found")


@router.post("/rename")
async def rename_chat_history(req: RenameRequest, store: ChatStore = Depends(get_chat_store)) -> dict[str, str]:
    try:
        return store.rename(req.old_path, req.new_title)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chat history not found")


@router.post("/mkdir")
async def create_directory(req: MkdirRequest, store: ChatStore = Depends(get_chat_store)) -> dict[str, str]:
    try:
        return store.create_directory(req.parent_path, req.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/move")
async def move_history(req: MoveRequest, store: ChatStore = Depends(get_chat_store)) -> dict[str, str]:
    try:
        return store.move(req.filename, req.target_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/branch")
async def create_branch(req: BranchRequest, store: ChatStore = Depends(get_chat_store)) -> dict[str, Any]:
    try:
        return store.branch(req.parent_file, req.branch_message_idx)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Parent chat history not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/merge")
async def merge_branch(req: MergeRequest, store: ChatStore = Depends(get_chat_store)) -> dict[str, Any]:
    try:
        return store.merge(req.child_file, cleanup_uploads_fn=delete_chat_upload_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Child chat history not found")
    except ValueError as e:
        status = 409 if "diverged" in str(e) else 400
        raise HTTPException(status_code=status, detail=str(e))
