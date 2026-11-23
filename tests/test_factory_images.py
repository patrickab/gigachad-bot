"""`POST /api/factory/images`: images attached to a factory task land on the host and answer their path."""

from pathlib import Path

from backend.identity import get_request_identity
from backend.routes.factory import router
from lib import factory_images
from tests.conftest import request_identity, route_client

PNG = b"\x89PNG\r\n\x1a\n"


def _client(tmp_path: Path, monkeypatch, has_host_access: bool = True):
    monkeypatch.setattr(factory_images, "IMAGES_DIRECTORY", tmp_path / "images")
    return route_client(router, overrides={get_request_identity: request_identity(has_host_access)})


def test_an_image_is_stored_under_a_safe_unique_name(tmp_path, monkeypatch):
    answer = _client(tmp_path, monkeypatch).post("/api/factory/images", files={"file": ("../my shot (1).PNG", PNG, "image/png")})

    path = Path(answer.json()["path"])
    assert answer.status_code == 200
    assert path.parent == tmp_path / "images"
    assert path.name.endswith("-my_shot_1.png")
    assert path.read_bytes() == PNG


def test_the_suffix_falls_back_to_the_content_type(tmp_path, monkeypatch):
    answer = _client(tmp_path, monkeypatch).post("/api/factory/images", files={"file": ("blob", PNG, "image/webp")})

    assert answer.json()["path"].endswith(".webp")


def test_anything_but_an_image_is_refused(tmp_path, monkeypatch):
    answer = _client(tmp_path, monkeypatch).post("/api/factory/images", files={"file": ("notes.txt", b"x", "text/plain")})

    assert answer.status_code == 400


def test_an_oversized_image_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(factory_images, "MAX_IMAGE_BYTES", 4)

    answer = _client(tmp_path, monkeypatch).post("/api/factory/images", files={"file": ("a.png", PNG, "image/png")})

    assert answer.status_code == 413


def test_a_login_without_host_access_is_refused(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, has_host_access=False)

    answer = client.post("/api/factory/images", files={"file": ("a.png", PNG, "image/png")})

    assert answer.status_code == 403
