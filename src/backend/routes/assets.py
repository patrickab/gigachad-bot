"""Authenticated byte serving for stored assets.

In Postgres mode the per-chat ``_uploads`` directories no longer exist on disk,
so the StaticFiles mounts cannot serve attachments. This route replaces them:
one asset at a time, always scoped to the requesting user's store.
"""

from fastapi import APIRouter, Depends, HTTPException, Response

from backend.routes.deps import get_asset_store
from config import DOCUMENTS
from lib.asset_store import AssetStore
from lib.data_store import InvalidStorageKey, LocalDataStore, StorageNotFoundError
from lib.document_library import mime_for

router = APIRouter(prefix="/api/assets", tags=["assets"])

# Local mode still keeps these bytes on disk. Only the chat tree is reachable here,
# so this route can never serve prompts, graphs, or anything else under Documents.
_LOCAL_PREFIX = "chat_history/"


def _local_asset(logical_path: str) -> Response:
    if not logical_path.startswith(_LOCAL_PREFIX):
        raise HTTPException(status_code=404, detail="Asset not found")
    try:
        content, revision = LocalDataStore(DOCUMENTS).read_bytes(logical_path)
    except (InvalidStorageKey, StorageNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Asset not found") from exc
    return Response(content=content, media_type=mime_for(logical_path), headers={"ETag": f'"{revision.token}"'})


@router.get("/{logical_path:path}")
def get_asset(logical_path: str, assets: AssetStore | None = Depends(get_asset_store)) -> Response:
    if assets is None:
        return _local_asset(logical_path)
    try:
        asset = assets.read(logical_path)
    except (InvalidStorageKey, StorageNotFoundError) as exc:
        # The store is scoped to this user, so another user's asset is simply
        # absent here — answer 404 so it stays undiscoverable.
        raise HTTPException(status_code=404, detail="Asset not found") from exc
    # ponytail: version is a monotonic counter, so a plain strong ETag is enough;
    # add Last-Modified/If-None-Match handling only if a client asks for it.
    return Response(content=asset.content, media_type=asset.mime, headers={"ETag": f'"{asset.version}"'})
