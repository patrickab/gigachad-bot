"""`POST /api/pi/pastes`: videos pasted into a Pi composer land in the session folder's `.pi/pastes/`."""

import re

from fastapi.testclient import TestClient
import pytest

from backend.identity import get_request_identity
from backend.routes.deps import get_file_roots
from backend.routes.pi import router
from lib import pi_host, pi_pastes
from tests.conftest import mounted_roots, request_identity, route_client

VIDEO = b"\x00\x00\x00\x18ftypmp42"


@pytest.fixture
def git_root(tmp_path, monkeypatch):
    root = tmp_path / "git"
    (root / "app" / ".git").mkdir(parents=True)
    monkeypatch.setattr(pi_host, "REPOSITORIES_ROOT", root)
    return root


def _client(*mounts, has_host_access: bool = True) -> TestClient:
    overrides = {get_request_identity: request_identity(has_host_access), get_file_roots: mounted_roots(*mounts)}
    return route_client(router, overrides=overrides)


def _paste(client: TestClient, params: dict[str, str], name: str = "my clip (1).mp4", content_type: str = "video/mp4", data: bytes = VIDEO):
    return client.post("/api/pi/pastes", params=params, files={"file": (name, data, content_type)})


def test_a_video_lands_in_the_repo_pastes_folder_under_a_safe_timestamped_name(git_root):
    answer = _paste(_client(git_root), {"repo": "app"})

    assert answer.status_code == 200
    mention = answer.json()["mention"]
    assert re.fullmatch(r"\.pi/pastes/\d{8}-\d{6}-my_clip_1\.mp4", mention)
    stored = git_root / "app" / mention
    assert answer.json()["path"] == str(stored)
    assert stored.read_bytes() == VIDEO


def test_an_absolute_cwd_inside_the_file_roots_takes_a_named_or_typed_video(tmp_path):
    vault = tmp_path / "vault"
    (vault / "docs").mkdir(parents=True)
    client = _client(vault)

    assert _paste(client, {"cwd": str(vault / "docs")}, name="../../clip.webm", content_type="video/webm").json()["mention"].endswith("-clip.webm")
    assert _paste(client, {"cwd": str(vault / "docs")}, name="blob", content_type="video/quicktime").json()["mention"].endswith("-blob.mov")
    assert sorted(path.suffix for path in (vault / "docs" / ".pi" / "pastes").iterdir()) == [".mov", ".webm"]


def test_folders_outside_the_file_roots_or_relative_are_unknown(tmp_path):
    vault, outside = tmp_path / "vault", tmp_path / "outside"
    vault.mkdir()
    outside.mkdir()
    client = _client(vault)

    assert _paste(client, {"cwd": str(outside)}).status_code == 404
    assert _paste(client, {"cwd": str(vault / ".." / "outside")}).status_code == 404
    assert _paste(client, {"cwd": "vault"}).status_code == 404
    assert _paste(client, {"cwd": str(vault / "missing")}).status_code == 404
    assert _paste(client, {"repo": "../etc"}).status_code == 404
    assert not (outside / ".pi").exists()


def test_only_previewable_videos_up_to_the_size_limit_are_taken(tmp_path, monkeypatch):
    client = _client(tmp_path)
    params = {"cwd": str(tmp_path)}

    assert _paste(client, params, name="notes.txt", content_type="text/plain").status_code == 400
    assert _paste(client, params, name="clip.avi", content_type="video/x-msvideo").status_code == 400
    monkeypatch.setattr(pi_pastes, "MAX_PASTE_BYTES", len(VIDEO) - 1)
    assert _paste(client, params).status_code == 413
    assert _paste(client, {}).status_code == 400
    assert _paste(client, {"cwd": str(tmp_path), "repo": "app"}).status_code == 400
    assert not (tmp_path / ".pi").exists()


def test_a_symlinked_pastes_folder_is_refused(tmp_path):
    vault, elsewhere = tmp_path / "vault", tmp_path / "elsewhere"
    vault.mkdir()
    elsewhere.mkdir()
    (vault / ".pi").symlink_to(elsewhere)

    assert _paste(_client(vault), {"cwd": str(vault)}).status_code == 400
    assert list(elsewhere.iterdir()) == []


def test_logins_without_host_access_are_refused(tmp_path):
    assert _paste(_client(tmp_path, has_host_access=False), {"cwd": str(tmp_path)}).status_code == 403
