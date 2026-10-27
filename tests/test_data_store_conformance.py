import pytest

from lib.changes import UserScope
from lib.data_store import (
    DataStore,
    InvalidStorageKey,
    StorageConflictError,
    StorageNotFoundError,
)
from lib.postgres_data_store import PostgresDataStore
from tests.conftest import create_user


def entries_by_key(data_store: DataStore, prefix: str = "", *, recursive: bool = False):
    return {entry.key: entry for entry in data_store.list(prefix, recursive=recursive)}


def test_missing_read_raises_storage_not_found(data_store: DataStore):
    with pytest.raises(StorageNotFoundError):
        data_store.read_bytes("missing.json")


def test_write_reads_back_and_enforces_revision(data_store: DataStore):
    first = data_store.write_bytes("state.json", b"one")
    content, read_revision = data_store.read_bytes("state.json")

    assert content == b"one"
    assert read_revision == first

    second = data_store.write_bytes("state.json", b"two", expected=first)
    assert second != first

    with pytest.raises(StorageConflictError):
        data_store.write_bytes("state.json", b"three", expected=first)


def test_stale_write_rejects_deleted_key(data_store: DataStore):
    revision = data_store.write_bytes("state.json", b"one")
    data_store.delete("state.json")

    with pytest.raises(StorageConflictError):
        data_store.write_bytes("state.json", b"two", expected=revision)


def test_apply_is_all_or_nothing(data_store: DataStore):
    first = data_store.write_bytes("ws/a.c4", b"a1")
    second = data_store.write_bytes("ws/b.c4", b"b1")
    data_store.write_bytes("ws/b.c4", b"b2", expected=second)
    data_store.write_bytes("ws/gone.c4", b"g")

    with pytest.raises(StorageConflictError):
        data_store.apply([("ws/a.c4", b"a2", first), ("ws/new.json", b"{}", None), ("ws/b.c4", b"b3", second)], ["ws/gone.c4"])

    assert data_store.read_bytes("ws/a.c4")[0] == b"a1"
    assert not data_store.exists("ws/new.json")
    assert data_store.exists("ws/gone.c4")

    current = data_store.read_bytes("ws/b.c4")[1]
    data_store.apply([("ws/a.c4", b"a2", first), ("ws/b.c4", b"b3", current)], ["ws/gone.c4"])
    assert [data_store.read_bytes(key)[0] for key in ("ws/a.c4", "ws/b.c4")] == [b"a2", b"b3"]
    assert not data_store.exists("ws/gone.c4")


def test_list_reports_direct_children_and_recursive_descendants(data_store: DataStore):
    data_store.write_bytes("parent/one.txt", b"1")
    data_store.write_bytes("parent/nested/two.txt", b"22")

    direct = entries_by_key(data_store, "parent")
    recursive = entries_by_key(data_store, "parent", recursive=True)

    assert set(direct) == {"parent/one.txt", "parent/nested"}
    assert direct["parent/one.txt"].is_dir is False
    assert direct["parent/one.txt"].size == 1
    assert direct["parent/nested"].is_dir is True
    assert set(recursive) == {"parent/one.txt", "parent/nested", "parent/nested/two.txt"}
    assert recursive["parent/nested/two.txt"].size == 2


def test_explicit_empty_directory_exists_and_is_listed(data_store: DataStore):
    data_store.mkdir("empty")

    assert data_store.exists("empty")
    entry = entries_by_key(data_store)["empty"]
    assert entry.is_dir is True
    assert entry.size is None
    assert entry.revision is None


def test_prefix_with_children_exists_without_explicit_directory(data_store: DataStore):
    data_store.write_bytes("implied/child.json", b"{}")

    assert data_store.exists("implied")


def test_delete_handles_single_keys_and_recursive_prefixes(data_store: DataStore):
    data_store.write_bytes("single.txt", b"one")
    data_store.write_bytes("tree/first.txt", b"one")
    data_store.write_bytes("tree/nested/second.txt", b"two")

    data_store.delete("single.txt")
    data_store.delete("tree", recursive=True)

    assert data_store.exists("single.txt") is False
    assert data_store.exists("tree") is False
    assert data_store.list("tree", recursive=True) == []


def test_non_recursive_delete_rejects_non_empty_directory(data_store: DataStore):
    data_store.write_bytes("tree/child.txt", b"one")

    with pytest.raises(OSError):
        data_store.delete("tree")


def test_move_handles_single_key_and_prefix(data_store: DataStore):
    data_store.write_bytes("source.txt", b"single")
    data_store.write_bytes("tree/first.txt", b"one")
    data_store.write_bytes("tree/nested/second.txt", b"two")

    data_store.move("source.txt", "destination.txt")
    data_store.move("tree", "moved")

    assert data_store.exists("source.txt") is False
    assert data_store.read_bytes("destination.txt")[0] == b"single"
    assert data_store.exists("tree") is False
    assert data_store.read_bytes("moved/first.txt")[0] == b"one"
    assert data_store.read_bytes("moved/nested/second.txt")[0] == b"two"


def test_postgres_store_cannot_read_another_users_key(postgres_pool, clean_database):
    first_user = create_user(postgres_pool, "first@example.test")
    second_user = create_user(postgres_pool, "second@example.test")

    PostgresDataStore(UserScope(postgres_pool, first_user)).write_bytes("private.json", b"first")
    second_store = PostgresDataStore(UserScope(postgres_pool, second_user))

    assert second_store.exists("private.json") is False
    assert second_store.list(recursive=True) == []
    with pytest.raises(StorageNotFoundError):
        second_store.read_bytes("private.json")


def test_move_missing_source_raises_storage_not_found(data_store: DataStore):
    with pytest.raises(StorageNotFoundError):
        data_store.move("missing", "destination")


@pytest.mark.parametrize("key", ["", "/absolute", "../outside", "parent/../../outside"])
def test_invalid_storage_keys_are_rejected(data_store: DataStore, key: str):
    operations = [
        lambda: data_store.read_bytes(key),
        lambda: data_store.write_bytes(key, b"content"),
        lambda: data_store.exists(key),
        lambda: data_store.mkdir(key),
        lambda: data_store.delete(key),
        lambda: data_store.move(key, "destination"),
        lambda: data_store.move("source", key),
    ]
    if key:
        operations.append(lambda: data_store.list(key))

    for operation in operations:
        with pytest.raises(InvalidStorageKey):
            operation()
