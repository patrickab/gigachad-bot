import pytest

from lib.data_store import LocalDataStore, StorageConflictError


def test_local_store_rejects_a_stale_revision_and_traversal(tmp_path):
    store = LocalDataStore(tmp_path)
    first = store.write_bytes("chat_history/chat.json", b"one")
    store.write_bytes("chat_history/chat.json", b"two", expected=first)

    with pytest.raises(StorageConflictError):
        store.write_bytes("chat_history/chat.json", b"three", expected=first)
    with pytest.raises(ValueError):
        store.read_bytes("../outside")
