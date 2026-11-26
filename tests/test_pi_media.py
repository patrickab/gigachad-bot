"""Tool-result images: the media routes cache by content hash, thumbnails are 480px WebP, pi-host is asked once per hash."""

import hashlib
import io

from fastapi.testclient import TestClient
import httpx
from PIL import Image
import pytest

from backend.identity import get_request_identity
from backend.routes.pi import router
from backend.websocket_origin import require_trusted_request
from lib import pi_host, pi_media
from tests.conftest import request_identity, route_client

TOKEN = "secret-token"


@pytest.fixture(autouse=True)
def token_file(tmp_path, monkeypatch):
    path = tmp_path / "pi-host.token"
    path.write_text(f"{TOKEN}\n")
    monkeypatch.setenv("PI_HOST_TOKEN_FILE", str(path))
    monkeypatch.setattr(pi_media, "_cache", pi_media.OrderedDict())
    monkeypatch.setattr(pi_media, "_cache_bytes", 0)


def _png(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(out, format="PNG")
    return out.getvalue()


def _serve(monkeypatch, stored: dict[str, bytes]) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        data = stored.get(request.url.path.removeprefix("/media/"))
        if data is None:
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, content=data, headers={"content-type": "image/png"})

    monkeypatch.setattr(pi_host, "_transport", httpx.MockTransport(handle))
    return requests


def _client() -> TestClient:
    return route_client(router, overrides={get_request_identity: request_identity(True)})


def test_thumbnail_is_a_webp_of_at_most_480px_and_small_images_keep_their_size():
    big = Image.open(io.BytesIO(pi_media.make_thumbnail(_png(1920, 1080))))
    small = Image.open(io.BytesIO(pi_media.make_thumbnail(_png(100, 50))))

    assert big.format == "WEBP" and big.size == (480, 270)
    assert small.format == "WEBP" and small.size == (100, 50)
    with pytest.raises(ValueError):
        pi_media.make_thumbnail(b"not an image")


def test_the_original_is_served_immutable_with_the_hash_as_etag(monkeypatch):
    data = _png(8, 8)
    digest = hashlib.sha256(data).hexdigest()
    requests = _serve(monkeypatch, {digest: data})

    response = _client().get(f"/api/pi/media/{digest}")

    assert response.status_code == 200 and response.content == data
    assert response.headers["content-type"] == "image/png"
    assert response.headers["etag"] == f'"{digest}"'
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert requests[0].headers["authorization"] == f"Bearer {TOKEN}"


def test_a_repeat_with_the_etag_is_a_304_without_asking_pi_host(monkeypatch):
    requests = _serve(monkeypatch, {})
    digest = "a" * 64

    response = _client().get(f"/api/pi/media/{digest}", headers={"If-None-Match": f'"{digest}"'})
    thumb = _client().get(f"/api/pi/media/{digest}/thumb", headers={"If-None-Match": f'"{digest}-480w"'})

    assert response.status_code == thumb.status_code == 304
    assert thumb.headers["cache-control"].endswith("immutable")
    assert requests == []


def test_the_thumbnail_is_made_once_and_cached(monkeypatch):
    data = _png(1200, 800)
    digest = hashlib.sha256(data).hexdigest()
    requests = _serve(monkeypatch, {digest: data})
    client = _client()

    first = client.get(f"/api/pi/media/{digest}/thumb")
    second = client.get(f"/api/pi/media/{digest}/thumb")

    assert first.headers["content-type"] == "image/webp"
    assert first.headers["etag"] == f'"{digest}-480w"'
    assert first.content == second.content
    assert Image.open(io.BytesIO(first.content)).size == (480, 320)
    assert len(requests) == 1


def test_an_unreadable_image_falls_back_to_the_original(monkeypatch):
    digest = "b" * 64
    _serve(monkeypatch, {digest: b"not an image"})

    response = _client().get(f"/api/pi/media/{digest}/thumb")

    assert response.status_code == 200 and response.content == b"not an image"


def test_unknown_and_malformed_hashes(monkeypatch):
    _serve(monkeypatch, {})
    client = _client()

    assert client.get(f"/api/pi/media/{'c' * 64}").status_code == 404
    assert client.get("/api/pi/media/not-a-hash").status_code == 422
    assert client.get("/api/pi/media/not-a-hash/thumb").status_code == 422


def test_media_needs_host_access(monkeypatch):
    _serve(monkeypatch, {})
    client = route_client(router, overrides={get_request_identity: request_identity(False)})

    assert client.get(f"/api/pi/media/{'d' * 64}").status_code in (401, 403)


def test_a_delete_forgets_the_thumbnail_of_a_hash_only_in_not_removed(monkeypatch):
    data = _png(1200, 800)
    digest = hashlib.sha256(data).hexdigest()
    _serve(monkeypatch, {digest: data})

    async def deleted(*_args):
        return pi_host.TurnDeleted(removed_entries=2, removed_media=[], not_removed=[digest, "paste-1"])

    monkeypatch.setattr(pi_host, "delete_turn", deleted)
    client = route_client(
        router,
        overrides={get_request_identity: request_identity(True), require_trusted_request: None},
    )
    client.get(f"/api/pi/media/{digest}/thumb")
    assert digest in pi_media._cache

    assert client.post("/api/pi/sessions/s1/turns/e1/delete", json={"through": "turn"}).status_code == 200

    assert digest not in pi_media._cache
    assert pi_media._cache_bytes == 0
