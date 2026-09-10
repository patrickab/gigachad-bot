from email.message import Message
from urllib.error import HTTPError

import pytest

from lib.data_store import LocalDataStore, StorageConflictError, WebDavDataStore


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
