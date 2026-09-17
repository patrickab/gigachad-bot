import os
from uuid import uuid4

from psycopg_pool import ConnectionPool
import pytest

from lib.chat_store import ChatStore
from lib.db_schema import upgrade
from lib.json_io import safe_write_json
from lib.postgres_data_store import PostgresDataStore


@pytest.fixture(scope="module")
def postgres_pool():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for Postgres-backed chat store tests")
    upgrade(url)
    pool = ConnectionPool(url, min_size=1, max_size=4)
    yield pool
    pool.close()


@pytest.fixture
def chat_store(postgres_pool) -> ChatStore:
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute("TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE")
        user_id = connection.execute(
            "INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", (f"test-{uuid4()}@example.test",)
        ).fetchone()[0]
    return ChatStore(data_store=PostgresDataStore(postgres_pool, user_id))


def test_history_listing_sorts_storage_backed_paths_by_name(chat_store: ChatStore):
    store = chat_store
    for filename in ("z.json", "a.json", "folder/b.json"):
        safe_write_json(store.resolve_path(filename), {"chat_id": filename, "messages": []})

    assert store.list_histories() == {
        "files": ["a.json", "z.json"],
        "histories": {"folder": ["b.json"]},
    }
    assert list(store.get_branch_meta()) == ["a.json", "folder/b.json", "z.json"]
