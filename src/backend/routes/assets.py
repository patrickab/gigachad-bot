"""Authenticated byte serving for stored assets, always scoped to the requesting user's store."""

from fastapi import APIRouter, Depends, Response

from backend.http_plumbing import http_errors
from backend.identity import RequestIdentity, get_request_identity
from backend.routes.deps import get_asset_store
from lib.asset_store import Asset, AssetStore
from lib.attachment_materialize import restore_library_image
from lib.data_store import InvalidStorageKey, StorageNotFoundError

router = APIRouter(prefix="/api/assets", tags=["assets"])


def _read_or_restore(assets: AssetStore, logical_path: str, *, host_access: bool) -> Asset:
    """The stored asset, else a MinerU image restored from the host's Nextcloud copy, for host-access logins only."""
    try:
        return assets.read(logical_path)
    except StorageNotFoundError:
        restored = restore_library_image(assets, logical_path) if host_access else None
        if restored is None:
            raise
        return restored


@router.get("/{logical_path:path}")
def get_asset(
    logical_path: str,
    assets: AssetStore = Depends(get_asset_store),
    identity: RequestIdentity = Depends(get_request_identity),
) -> Response:
    # The store is scoped to this user, so another user's asset is absent here.
    # Answer 404 so it stays undiscoverable.
    with http_errors(((InvalidStorageKey, StorageNotFoundError), 404, "Asset not found")):
        asset = _read_or_restore(assets, logical_path, host_access=identity.has_host_access)
    # ponytail: version is a monotonic counter, so a plain strong ETag is enough;
    # add Last-Modified/If-None-Match handling only if a client asks for it.
    return Response(content=asset.content, media_type=asset.mime, headers={"ETag": f'"{asset.version}"'})
