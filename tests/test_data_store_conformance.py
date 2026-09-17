import os
from uuid import uuid4

from psycopg_pool import ConnectionPool

from lib.db_schema import upgrade
from lib.postgres_data_store import PostgresDataStore


import pytest

from lib.data_store import (
    DataStore,
    InvalidStorageKey,
    LocalDataStore,
    StorageConflictError,
    StorageNotFoundError,
)


@pytest.fixture(scope="session")
def postgres_pool():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for Postgres DataStore conformance tests")
    upgrade(url)
    pool = ConnectionPool(url, min_size=1, max_size=4)
    yield pool
    pool.close()


@pytest.fixture(params=("local", "postgres"))
def store(request, tmp_path) -> DataStore:
    if request.param == "local":
        return LocalDataStore(tmp_path)

    postgres_pool = request.getfixturevalue("postgres_pool")
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute("TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE")
        user_id = connection.execute(
            "INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", (f"test-{uuid4()}@example.test",)
        ).fetchone()[0]
    return PostgresDataStore(postgres_pool, user_id)


def entries_by_key(store: DataStore, prefix: str = "", *, recursive: bool = False):
    return {entry.key: entry for entry in store.list(prefix, recursive=recursive)}


def test_missing_read_raises_storage_not_found(store: DataStore):
    with pytest.raises(StorageNotFoundError):
        store.read_bytes("missing.json")


def test_write_reads_back_and_enforces_revision(store: DataStore):
    first = store.write_bytes("state.json", b"one")
    content, read_revision = store.read_bytes("state.json")

    assert content == b"one"
    assert read_revision == first

    second = store.write_bytes("state.json", b"two", expected=first)
    assert second != first

    with pytest.raises(StorageConflictError):
        store.write_bytes("state.json", b"three", expected=first)


def test_stale_write_rejects_deleted_key(store: DataStore):
    revision = store.write_bytes("state.json", b"one")
    store.delete("state.json")

    with pytest.raises(StorageConflictError):
        store.write_bytes("state.json", b"two", expected=revision)


def test_list_reports_direct_children_and_recursive_descendants(store: DataStore):
    store.write_bytes("parent/one.txt", b"1")
    store.write_bytes("parent/nested/two.txt", b"22")

    direct = entries_by_key(store, "parent")
    recursive = entries_by_key(store, "parent", recursive=True)

    assert set(direct) == {"parent/one.txt", "parent/nested"}
    assert direct["parent/one.txt"].is_dir is False
    assert direct["parent/one.txt"].size == 1
    assert direct["parent/nested"].is_dir is True
    assert set(recursive) == {"parent/one.txt", "parent/nested", "parent/nested/two.txt"}
    assert recursive["parent/nested/two.txt"].size == 2


def test_explicit_empty_directory_exists_and_is_listed(store: DataStore):
    store.mkdir("empty")

    assert store.exists("empty")
    entry = entries_by_key(store)["empty"]
    assert entry.is_dir is True
    assert entry.size is None
    assert entry.revision is None


def test_prefix_with_children_exists_without_explicit_directory(store: DataStore):
    store.write_bytes("implied/child.json", b"{}")

    assert store.exists("implied")


def test_delete_handles_single_keys_and_recursive_prefixes(store: DataStore):
    store.write_bytes("single.txt", b"one")
    store.write_bytes("tree/first.txt", b"one")
    store.write_bytes("tree/nested/second.txt", b"two")

    store.delete("single.txt")
    store.delete("tree", recursive=True)

    assert store.exists("single.txt") is False
    assert store.exists("tree") is False
    assert store.list("tree", recursive=True) == []


def test_non_recursive_delete_rejects_non_empty_directory(store: DataStore):
    store.write_bytes("tree/child.txt", b"one")

    with pytest.raises(OSError):
        store.delete("tree")


def test_move_handles_single_key_and_prefix(store: DataStore):
    store.write_bytes("source.txt", b"single")
    store.write_bytes("tree/first.txt", b"one")
    store.write_bytes("tree/nested/second.txt", b"two")

    store.move("source.txt", "destination.txt")
    store.move("tree", "moved")

    assert store.exists("source.txt") is False
    assert store.read_bytes("destination.txt")[0] == b"single"
    assert store.exists("tree") is False
    assert store.read_bytes("moved/first.txt")[0] == b"one"
    assert store.read_bytes("moved/nested/second.txt")[0] == b"two"


def test_postgres_store_cannot_read_another_users_key(postgres_pool):
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute("TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE")
        first_user = connection.execute(
            "INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", ("first@example.test",)
        ).fetchone()[0]
        second_user = connection.execute(
            "INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", ("second@example.test",)
        ).fetchone()[0]

    PostgresDataStore(postgres_pool, first_user).write_bytes("private.json", b"first")
    second_store = PostgresDataStore(postgres_pool, second_user)

    assert second_store.exists("private.json") is False
    assert second_store.list(recursive=True) == []
    with pytest.raises(StorageNotFoundError):
        second_store.read_bytes("private.json")


def test_move_missing_source_raises_storage_not_found(store: DataStore):
    with pytest.raises(StorageNotFoundError):
        store.move("missing", "destination")


@pytest.mark.parametrize("key", ["", "/absolute", "../outside", "parent/../../outside"])
def test_invalid_storage_keys_are_rejected(store: DataStore, key: str):
    operations = [
        lambda: store.read_bytes(key),
        lambda: store.write_bytes(key, b"content"),
        lambda: store.exists(key),
        lambda: store.mkdir(key),
        lambda: store.delete(key),
        lambda: store.move(key, "destination"),
        lambda: store.move("source", key),
    ]
    if key:
        operations.append(lambda: store.list(key))

    for operation in operations:
        with pytest.raises(InvalidStorageKey):
            operation()
