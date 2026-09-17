"""Authenticated byte serving for stored assets.

The per-chat ``_uploads`` directories no longer exist on disk, so the StaticFiles
mounts cannot serve attachments. This route replaces them: one asset at a time,
always scoped to the requesting user's store.
"""

from fastapi import APIRouter, Depends, HTTPException, Response

from backend.routes.deps import get_asset_store
from lib.asset_store import AssetStore
from lib.data_store import InvalidStorageKey, StorageNotFoundError

router = APIRouter(prefix="/api/assets", tags=["assets"])


@router.get("/{logical_path:path}")
def get_asset(logical_path: str, assets: AssetStore = Depends(get_asset_store)) -> Response:
    try:
        asset = assets.read(logical_path)
    except (InvalidStorageKey, StorageNotFoundError) as exc:
        # The store is scoped to this user, so another user's asset is simply
        # absent here — answer 404 so it stays undiscoverable.
        raise HTTPException(status_code=404, detail="Asset not found") from exc
    # ponytail: version is a monotonic counter, so a plain strong ETag is enough;
    # add Last-Modified/If-None-Match handling only if a client asks for it.
    return Response(content=asset.content, media_type=asset.mime, headers={"ETag": f'"{asset.version}"'})
