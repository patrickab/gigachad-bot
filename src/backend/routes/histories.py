from collections.abc import Callable
import logging
from typing import Any, TypeVar

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.routes.architecture_graphs import ArchitectureGraphContextReferenceModel
from backend.routes.deps import get_asset_store, get_chat_store, get_sandbox_service
from backend.routes.files import delete_chat_upload_dir
from lib.asset_store import AssetStore
from lib.chat_store import ChatStore
from lib.data_store import StorageConflictError
from lib.sandbox_service import SandboxService

router = APIRouter(prefix="/api/chat-histories", tags=["histories"])
logger = logging.getLogger(__name__)


def _upload_cleanup(assets: AssetStore) -> Callable[[str, str | None], None]:
    """Bind the request's asset store so chat deletion also purges stored uploads."""
    return lambda chat_id, slug=None: delete_chat_upload_dir(chat_id, slug, assets)


def _sandbox_cleanup_recorder() -> tuple[Callable[[str, str | None], None], list[str]]:
    """Record chat IDs because ChatStore's cleanup callback cannot await sandbox deletion."""
    seen: list[str] = []
    return (lambda chat_id, _slug=None: seen.append(chat_id)), seen


def _composed_cleanup(assets: AssetStore, recorder: Callable[[str, str | None], None]) -> Callable[[str, str | None], None]:
    upload_cleanup = _upload_cleanup(assets)

    def cleanup(chat_id: str, slug: str | None = None) -> None:
        upload_cleanup(chat_id, slug)
        recorder(chat_id, slug)

    return cleanup

T = TypeVar("T")


async def cleanup_after_delete(
    assets: AssetStore,
    sandbox_service: SandboxService,
    run: Callable[[Callable[[str, str | None], None]], T],
) -> T:
    """Delete sandbox state after chat deletion succeeds.

    Log cleanup failures because the chat is already gone.
    """
    recorder, deleted_chat_ids = _sandbox_cleanup_recorder()
    result = run(_composed_cleanup(assets, recorder))
    for chat_id in deleted_chat_ids:
        try:
            await sandbox_service.delete_chat_state(chat_id=chat_id)
        except Exception:
            logger.exception("sandbox cleanup failed for chat %s after history deletion", chat_id)
    return result

class SaveRequest(BaseModel):
    messages: list[dict[str, Any]] = []
    chat_id: str | None = None
    title: str | None = None
    usage: dict[str, int] | None = None
    parent_id: str | None = None
    branch_message_idx: int | None = None
    children: list[dict[str, Any]] | None = None
    architecture_graph_contexts: list[ArchitectureGraphContextReferenceModel] | None = None
    expected_revision: str | None = None


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
    return {**data, "filename": filename, "revision": store.revision(filename)}


@router.put("/{filename:path}")
async def save_chat_history(
    filename: str,
    data: SaveRequest | None = None,
    store: ChatStore = Depends(get_chat_store),
    sandbox_service: SandboxService = Depends(get_sandbox_service),
) -> dict[str, str]:
    try:
        payload = data.model_dump() if data else None
        expected_revision = payload.pop("expected_revision", None) if payload else None
        result = store.save(filename, payload, expected_revision=expected_revision)
        if data is not None and data.chat_id:
            await sandbox_service.checkpoint(chat_id=data.chat_id)
        return result
    except StorageConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/cascade/{filename:path}")
async def cascade_delete(
    filename: str,
    store: ChatStore = Depends(get_chat_store),
    assets: AssetStore = Depends(get_asset_store),
    sandbox_service: SandboxService = Depends(get_sandbox_service),
) -> dict[str, Any]:
    try:
        return await cleanup_after_delete(
            assets, sandbox_service, lambda cleanup: store.cascade_delete(filename, cleanup_uploads_fn=cleanup)
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chat history not found")


@router.delete("/orphan/{filename:path}")
async def orphan_children(
    filename: str,
    store: ChatStore = Depends(get_chat_store),
    assets: AssetStore = Depends(get_asset_store),
    sandbox_service: SandboxService = Depends(get_sandbox_service),
) -> dict[str, Any]:
    try:
        return await cleanup_after_delete(
            assets, sandbox_service, lambda cleanup: store.orphan(filename, cleanup_uploads_fn=cleanup)
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chat history not found")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{filename:path}")
async def delete_chat_history(
    filename: str,
    store: ChatStore = Depends(get_chat_store),
    assets: AssetStore = Depends(get_asset_store),
    sandbox_service: SandboxService = Depends(get_sandbox_service),
) -> dict[str, str]:
    try:
        return await cleanup_after_delete(assets, sandbox_service, lambda cleanup: store.delete(filename, cleanup_uploads_fn=cleanup))
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
async def merge_branch(
    req: MergeRequest,
    store: ChatStore = Depends(get_chat_store),
    assets: AssetStore = Depends(get_asset_store),
    sandbox_service: SandboxService = Depends(get_sandbox_service),
) -> dict[str, Any]:
    try:
        return await cleanup_after_delete(
            assets, sandbox_service, lambda cleanup: store.merge(req.child_file, cleanup_uploads_fn=cleanup)
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Child chat history not found")
    except ValueError as e:
        status = 409 if "diverged" in str(e) else 400
        raise HTTPException(status_code=status, detail=str(e))
