"""MinerU images an older extraction never stored are restored from the Nextcloud copy on first open."""

import pytest

from backend.identity import get_request_identity
from backend.routes import assets as assets_route
from tests.conftest import request_identity, route_client

pytestmark = pytest.mark.usefixtures("clean_database")


def _client(asset_store, *, has_host_access: bool):
    return route_client(
        assets_route.router,
        overrides={assets_route.get_asset_store: asset_store, get_request_identity: request_identity(has_host_access)},
    )


@pytest.mark.parametrize("copy", ["images/paper/paper-1.png", "images/paper-1.png"])
def test_a_missing_mineru_image_is_restored_from_either_disk_layout(asset_store, documents_root, copy):
    disk = documents_root / "Mineru" / copy
    disk.parent.mkdir(parents=True)
    disk.write_bytes(b"\x89PNG restored")

    served = _client(asset_store, has_host_access=True).get("/api/assets/Mineru/images/paper/paper-1.png")

    assert served.status_code == 200
    assert served.content == b"\x89PNG restored"
    disk.unlink()
    assert asset_store.read("Mineru/images/paper/paper-1.png").content == b"\x89PNG restored"


def test_only_host_logins_restore_from_disk_and_nothing_outside_mineru_images(asset_store, documents_root):
    for copy in ("Mineru/images/paper/paper-1.png", "PDFs/paper.pdf"):
        (documents_root / copy).parent.mkdir(parents=True, exist_ok=True)
        (documents_root / copy).write_bytes(b"on disk")

    assert _client(asset_store, has_host_access=False).get("/api/assets/Mineru/images/paper/paper-1.png").status_code == 404
    assert _client(asset_store, has_host_access=True).get("/api/assets/PDFs/paper.pdf").status_code == 404
    assert asset_store.list() == []
