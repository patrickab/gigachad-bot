import os
from uuid import uuid4

from psycopg_pool import ConnectionPool
import pytest

from lib.chat_store import ChatStore
from lib.db_schema import upgrade
from lib.postgres_data_store import PostgresDataStore

CONTEXTS = [{"path": "graph/checkout.architecture.yaml"}]


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


def test_architecture_graph_contexts_round_trip_and_survive_ordinary_save(chat_store: ChatStore) -> None:
    store = chat_store
    store.save("chat.json", {"messages": [], "chat_id": "chat-1", "architecture_graph_contexts": CONTEXTS})

    assert store.load("chat.json")["architecture_graph_contexts"] == CONTEXTS

    # Existing callers that have no graph-context UI yet must not erase it.
    store.save("chat.json", {"messages": [{"role": "user", "content": "Hello"}], "chat_id": "chat-1"})

    assert store.load("chat.json")["architecture_graph_contexts"] == CONTEXTS


def test_architecture_graph_contexts_follow_branch_and_move(chat_store: ChatStore) -> None:
    store = chat_store
    store.save(
        "chat.json",
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_id": "chat-1",
            "architecture_graph_contexts": CONTEXTS,
        },
    )

    branch = store.branch("chat.json", 0)
    assert store.load(branch["child_file"])["architecture_graph_contexts"] == CONTEXTS

    moved = store.move("chat.json", "archive")
    assert store.load(moved["new_path"])["architecture_graph_contexts"] == CONTEXTS
