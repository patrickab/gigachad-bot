import asyncio
from collections.abc import Callable
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.routes.deps import get_asset_store, get_chat_store, get_sandbox_service, http_errors
from backend.routes.files import delete_chat_upload_dir
from backend.routes.schemas import ChatSnapshot
from lib.asset_store import AssetStore
from lib.chat_store import ChatStore
from lib.data_store import StorageConflictError
from lib.sandbox_service import SandboxService

router = APIRouter(prefix="/api/chat-histories", tags=["histories"])
logger = logging.getLogger(__name__)


async def cleanup_after_delete[T](
    assets: AssetStore,
    sandbox_service: SandboxService,
    run: Callable[[Callable[[str, str | None], None]], T],
) -> T:
    """Delete sandbox state after chat deletion succeeds.

    Log cleanup failures because the chat is already gone.
    """
    deleted_chat_ids: list[str] = []

    def cleanup(chat_id: str, slug: str | None = None) -> None:
        delete_chat_upload_dir(chat_id, slug, assets)
        # ChatStore's cleanup callback cannot await, so sandbox deletion waits for the IDs recorded here.
        deleted_chat_ids.append(chat_id)

    result = await asyncio.to_thread(run, cleanup)
    for chat_id in deleted_chat_ids:
        try:
            await sandbox_service.delete_chat_state(chat_id=chat_id)
        except Exception:
            logger.exception("sandbox cleanup failed for chat %s after history deletion", chat_id)
    return result


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
def get_branch_meta(dirs: str | None = None, store: ChatStore = Depends(get_chat_store)) -> dict[str, dict[str, Any]]:
    return store.get_branch_meta(dirs)


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
    sandbox_service: SandboxService = Depends(get_sandbox_service),
) -> dict[str, str]:
    """Save a chat, 409 when `expected_revision` is stale, then checkpoint its sandbox."""
    with http_errors(((StorageConflictError, ValueError), 409, None)):
        payload = data.model_dump() if data else None
        expected_revision = payload.pop("expected_revision", None) if payload else None
        result = await asyncio.to_thread(store.save, filename, payload, expected_revision=expected_revision)
        if data is not None and data.chat_id:
            await sandbox_service.checkpoint(chat_id=data.chat_id)
        return result


@router.delete("/cascade/{filename:path}")
async def cascade_delete(
    filename: str,
    store: ChatStore = Depends(get_chat_store),
    assets: AssetStore = Depends(get_asset_store),
    sandbox_service: SandboxService = Depends(get_sandbox_service),
) -> dict[str, Any]:
    """Delete a chat and all its descendant branches, purging uploads and sandbox state."""
    with http_errors((FileNotFoundError, 404, "Chat history not found")):
        return await cleanup_after_delete(
            assets, sandbox_service, lambda cleanup: store.cascade_delete(filename, cleanup_uploads_fn=cleanup)
        )


@router.delete("/{filename:path}")
async def delete_chat_history(
    filename: str,
    store: ChatStore = Depends(get_chat_store),
    assets: AssetStore = Depends(get_asset_store),
    sandbox_service: SandboxService = Depends(get_sandbox_service),
) -> dict[str, str]:
    with http_errors((FileNotFoundError, 404, "Chat history not found")):
        return await cleanup_after_delete(assets, sandbox_service, lambda cleanup: store.delete(filename, cleanup_uploads_fn=cleanup))


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
    assets: AssetStore = Depends(get_asset_store),
    sandbox_service: SandboxService = Depends(get_sandbox_service),
) -> dict[str, Any]:
    """Merge a branch back into its parent and delete the child, 409 when the parent diverged."""
    with http_errors((FileNotFoundError, 404, "Child chat history not found")):
        try:
            return await cleanup_after_delete(
                assets, sandbox_service, lambda cleanup: store.merge(req.child_file, cleanup_uploads_fn=cleanup)
            )
        except ValueError as e:
            raise HTTPException(status_code=409 if "diverged" in str(e) else 400, detail=str(e)) from e
