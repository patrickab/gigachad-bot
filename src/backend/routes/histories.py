import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.http_plumbing import http_errors
from backend.routes.deps import get_chat_cleanup, get_chat_store
from backend.routes.schemas import ChatSnapshot
from lib.chat_cleanup import ChatCleanup
from lib.chat_store import ChatStore, ParentDiverged
from lib.data_store import StorageConflictError

router = APIRouter(prefix="/api/chat-histories", tags=["histories"])


class SaveRequest(ChatSnapshot):
    expected_revision: str | None = None


class MoveRequest(BaseModel):
    filename: str
    target_dir: str


class BranchRequest(BaseModel):
    parent_file: str
    branch_message_idx: int


class MergeRequest(BaseModel):
    child_file: str


# Sync handlers: FastAPI runs them in its threadpool, so blocking store I/O for one request
# (e.g. the branch-meta scan) cannot stall a concurrent chat load on the event loop.
@router.get("")
def list_chat_histories(store: ChatStore = Depends(get_chat_store)) -> dict[str, list[str]]:
    return store.list_histories()


@router.get("/branch-meta")
# Registered before `/{filename:path}` so the literal path is not captured as a filename
def get_branch_meta(store: ChatStore = Depends(get_chat_store)) -> dict[str, dict[str, Any]]:
    return store.get_branch_meta()


@router.get("/all")
# Registered before `/{filename:path}`, like `/branch-meta`
def list_all_chats(store: ChatStore = Depends(get_chat_store)) -> dict[str, list[dict[str, Any]]]:
    """Every chat of the login, project chats included, newest first."""
    return {"chats": store.list_all()}


@router.get("/search")
# Registered before `/{filename:path}`, like `/branch-meta`
def search_chats(
    q: str = Query("", max_length=200), limit: int = Query(20, ge=1, le=100), store: ChatStore = Depends(get_chat_store)
) -> dict[str, list[dict[str, Any]]]:
    """Chats whose question or answer text holds *q*, case-insensitive, at most *limit* of them, newest first."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Empty search query")
    return {"hits": store.search(q, limit)}


@router.get("/{filename:path}")
def load_chat_history(filename: str, store: ChatStore = Depends(get_chat_store)) -> dict[str, Any]:
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
    """Save a chat, 409 when `expected_revision` is stale."""
    with http_errors(((StorageConflictError, ValueError), 409, None)):
        payload = data.model_dump() if data else None
        expected_revision = payload.pop("expected_revision", None) if payload else None
        return await asyncio.to_thread(store.save, filename, payload, expected_revision=expected_revision)


@router.delete("/cascade/{filename:path}")
async def cascade_delete(
    filename: str,
    store: ChatStore = Depends(get_chat_store),
    cleanup: ChatCleanup = Depends(get_chat_cleanup),
) -> dict[str, Any]:
    """Delete a chat and all its descendant branches, purging uploads and sandbox state."""
    with http_errors((FileNotFoundError, 404, "Chat history not found")):
        return await cleanup.after(lambda released: store.cascade_delete(filename, released=released))


@router.delete("/{filename:path}")
async def delete_chat_history(
    filename: str,
    store: ChatStore = Depends(get_chat_store),
    cleanup: ChatCleanup = Depends(get_chat_cleanup),
) -> dict[str, str]:
    with http_errors((FileNotFoundError, 404, "Chat history not found")):
        return await cleanup.after(lambda released: store.delete(filename, released=released))


@router.post("/move")
def move_history(req: MoveRequest, store: ChatStore = Depends(get_chat_store)) -> dict[str, str]:
    with http_errors((FileNotFoundError, 404, "Source not found"), (ValueError, 400, None)):
        return store.move(req.filename, req.target_dir)


@router.post("/branch")
def create_branch(req: BranchRequest, store: ChatStore = Depends(get_chat_store)) -> dict[str, Any]:
    with http_errors((FileNotFoundError, 404, "Parent chat history not found"), (ValueError, 400, None)):
        return store.branch(req.parent_file, req.branch_message_idx)


@router.post("/merge")
async def merge_branch(
    req: MergeRequest,
    store: ChatStore = Depends(get_chat_store),
    cleanup: ChatCleanup = Depends(get_chat_cleanup),
) -> dict[str, Any]:
    """Merge a branch back into its parent and delete the child, 409 when the parent diverged."""
    with http_errors((FileNotFoundError, 404, "Child chat history not found"), (ParentDiverged, 409, None), (ValueError, 400, None)):
        return await cleanup.after(lambda released: store.merge(req.child_file, released=released))
