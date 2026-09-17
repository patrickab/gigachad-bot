import os
from uuid import uuid4

from psycopg_pool import ConnectionPool
import pytest

from lib.chat_store import ChatStore
from lib.db_schema import upgrade
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


def test_save_with_empty_messages_does_not_erase_existing_messages(chat_store: ChatStore) -> None:
    """Regression test: a save payload missing/empty `messages` (e.g. from a stale
    client-side closure racing a history load) must not wipe stored conversation
    content, matching the fallback-to-existing behavior every other field already has.
    """
    store = chat_store
    real_messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]
    store.save("chat.json", {"messages": real_messages, "chat_id": "chat-1", "usage": {"total_tokens": 44800}})

    assert store.load("chat.json")["messages"] == real_messages

    # A save arriving with empty messages but no usage (usage falls back to existing,
    # as observed in the bug report: tokens survive, messages vanish) must preserve messages.
    store.save("chat.json", {"messages": [], "chat_id": "chat-1"})

    loaded = store.load("chat.json")
    assert loaded["messages"] == real_messages
    assert loaded["usage"] == {"total_tokens": 44800}


def test_save_with_no_data_still_preserves_existing_messages(chat_store: ChatStore) -> None:
    store = chat_store
    real_messages = [{"role": "user", "content": "Hello"}]
    store.save("chat.json", {"messages": real_messages, "chat_id": "chat-1"})

    store.save("chat.json", {"chat_id": "chat-1"})

    assert store.load("chat.json")["messages"] == real_messages


def test_save_on_new_file_with_no_messages_is_empty(chat_store: ChatStore) -> None:
    store = chat_store
    store.save("new.json", {"chat_id": "chat-2"})

    assert store.load("new.json")["messages"] == []
