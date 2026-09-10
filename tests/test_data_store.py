from email.message import Message
from urllib.error import HTTPError, URLError

import pytest

from lib.data_store import LocalDataStore, StorageConflictError, StorageError, WebDavDataStore


def test_local_store_rejects_a_stale_revision_and_traversal(tmp_path):
    store = LocalDataStore(tmp_path)
    first = store.write_bytes("chat_history/chat.json", b"one")
    store.write_bytes("chat_history/chat.json", b"two", expected=first)

    with pytest.raises(StorageConflictError):
        store.write_bytes("chat_history/chat.json", b"three", expected=first)
    with pytest.raises(ValueError):
        store.read_bytes("../outside")


class _Response:
    def __init__(self, body: bytes, etag: str) -> None:
        self._body = body
        self.headers = Message()
        self.headers["ETag"] = etag

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._body


def test_webdav_write_uses_the_revision_from_read(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if request.method == "GET":
            return _Response(b"one", '"v1"')
        return _Response(b"", '"v2"')

    monkeypatch.setattr("lib.data_store.urlopen", fake_urlopen)
    store = WebDavDataStore("https://cloud.example/Documents", "user", "password")
    _, revision = store.read_bytes("chat_history/chat.json")
    updated = store.write_bytes("chat_history/chat.json", b"two", expected=revision)

    assert requests[1].get_header("If-match") == '"v1"'
    assert updated.token == '"v2"'


def test_webdav_precondition_failure_is_a_conflict(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 412, "Precondition Failed", None, None)

    monkeypatch.setattr("lib.data_store.urlopen", fake_urlopen)
    store = WebDavDataStore("https://cloud.example/Documents", "user", "password")

    with pytest.raises(StorageConflictError):
        store.write_bytes("chat_history/chat.json", b"two", expected=type("R", (), {"token": '"v1"'})())


def test_webdav_uses_cached_reads_without_contacting_remote(monkeypatch, tmp_path):
    fallback = LocalDataStore(tmp_path)
    fallback.write_bytes("chat_history/chat.json", b"cached")

    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("cache hit should not contact WebDAV")

    monkeypatch.setattr("lib.data_store.urlopen", unexpected_request)
    store = WebDavDataStore("https://cloud.example/Documents", "user", "password", fallback=fallback)

    assert store.read_bytes("chat_history/chat.json")[0] == b"cached"
    assert store.exists("chat_history/chat.json")
    assert [entry.key for entry in store.list("chat_history")] == ["chat_history/chat.json"]


def test_webdav_cached_revision_is_preflighted_before_direct_write(monkeypatch, tmp_path):
    fallback = LocalDataStore(tmp_path)
    local_revision = fallback.write_bytes("chat_history/chat.json", b"old")
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if request.method == "GET":
            return _Response(b"old", '"remote-v1"')
        return _Response(b"", '"remote-v2"')

    monkeypatch.setattr("lib.data_store.urlopen", fake_urlopen)
    store = WebDavDataStore("https://cloud.example/Documents", "user", "password", fallback=fallback)

    revision = store.write_bytes("chat_history/chat.json", b"new", expected=local_revision)

    assert requests[1].get_header("If-match") == '"remote-v1"'
    assert revision.token == '"remote-v2"'
    assert fallback.read_bytes("chat_history/chat.json")[0] == b"old"


def test_webdav_uses_local_fallback_only_when_unavailable(monkeypatch, tmp_path):
    fallback = LocalDataStore(tmp_path)
    revision = fallback.write_bytes("chat_history/chat.json", b"old")

    def offline(*_args, **_kwargs):
        raise URLError("offline")

    monkeypatch.setattr("lib.data_store.urlopen", offline)
    store = WebDavDataStore("https://cloud.example/Documents", "user", "password", fallback=fallback)

    store.write_bytes("chat_history/chat.json", b"new", expected=revision)

    assert fallback.read_bytes("chat_history/chat.json")[0] == b"new"


def test_webdav_remote_revision_can_fall_back_to_a_new_local_copy(monkeypatch, tmp_path):
    fallback = LocalDataStore(tmp_path)
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _Response(b"remote", '"remote-v1"')
        raise URLError("offline")

    monkeypatch.setattr("lib.data_store.urlopen", fake_urlopen)
    store = WebDavDataStore("https://cloud.example/Documents", "user", "password", fallback=fallback)
    _, remote_revision = store.read_bytes("chat_history/chat.json")

    store.write_bytes("chat_history/chat.json", b"offline update", expected=remote_revision)

    assert fallback.read_bytes("chat_history/chat.json")[0] == b"offline update"


def test_webdav_auth_failure_never_writes_to_local_fallback(monkeypatch, tmp_path):
    fallback = LocalDataStore(tmp_path)
    revision = fallback.write_bytes("chat_history/chat.json", b"old")

    def unauthorized(request, timeout):
        raise HTTPError(request.full_url, 401, "Unauthorized", None, None)

    monkeypatch.setattr("lib.data_store.urlopen", unauthorized)
    store = WebDavDataStore("https://cloud.example/Documents", "user", "password", fallback=fallback)

    with pytest.raises(StorageError):
        store.write_bytes("chat_history/chat.json", b"new", expected=revision)

    assert fallback.read_bytes("chat_history/chat.json")[0] == b"old"


def test_webdav_configuration_requires_all_credentials(monkeypatch):
    for name in ("GIGACHAD_WEBDAV_URL", "GIGACHAD_WEBDAV_USER", "GIGACHAD_WEBDAV_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    assert not WebDavDataStore.configured()

    monkeypatch.setenv("GIGACHAD_WEBDAV_URL", "https://cloud.example/Documents")
    monkeypatch.setenv("GIGACHAD_WEBDAV_USER", "user")
    monkeypatch.setenv("GIGACHAD_WEBDAV_PASSWORD", "password")
    assert WebDavDataStore.configured()
