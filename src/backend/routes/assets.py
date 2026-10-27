"""Authenticated byte serving for stored assets, always scoped to the requesting user's store."""

from fastapi import APIRouter, Depends, Response

from backend.http_plumbing import http_errors
from backend.routes.deps import get_asset_store
from lib.asset_store import AssetStore
from lib.data_store import InvalidStorageKey, StorageNotFoundError

router = APIRouter(prefix="/api/assets", tags=["assets"])


@router.get("/{logical_path:path}")
def get_asset(logical_path: str, assets: AssetStore = Depends(get_asset_store)) -> Response:
    # The store is scoped to this user, so another user's asset is absent here.
    # Answer 404 so it stays undiscoverable.
    with http_errors(((InvalidStorageKey, StorageNotFoundError), 404, "Asset not found")):
        asset = assets.read(logical_path)
    # ponytail: version is a monotonic counter, so a plain strong ETag is enough;
    # add Last-Modified/If-None-Match handling only if a client asks for it.
    return Response(content=asset.content, media_type=asset.mime, headers={"ETag": f'"{asset.version}"'})
