import os
from uuid import uuid4

from psycopg_pool import ConnectionPool
import pytest

import config
from lib.db_schema import upgrade
from lib.postgres_data_store import PostgresDataStore


@pytest.fixture(scope="module")
def postgres_pool():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for Postgres-backed config tests")
    upgrade(url)
    pool = ConnectionPool(url, min_size=1, max_size=4)
    yield pool
    pool.close()


@pytest.fixture
def postgres_store(postgres_pool):
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute("TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE")
        user_id = connection.execute(
            "INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", (f"test-{uuid4()}@example.test",)
        ).fetchone()[0]
    return PostgresDataStore(postgres_pool, user_id)


def test_runtime_paths_are_outside_checkout():
    assert config.DIRECTORY_OUTPUT_MINERU == config.REMOTE_ROOT / "Documents/Mineru"
    for name, value in vars(config).items():
        if name.startswith("DIRECTORY_"):
            assert value.is_absolute()
            assert value.is_relative_to(config.REMOTE_ROOT)


def test_prompt_seed_preserves_edits_and_deletions(postgres_store):
    config.seed_prompts(postgres_store)
    prompt_key = next(
        entry.key for entry in postgres_store.list("prompt", recursive=True) if entry.key.endswith(".md")
    )

    _, revision = postgres_store.read_bytes(prompt_key)
    postgres_store.write_bytes(prompt_key, b"User edit", expected=revision)
    config.seed_prompts(postgres_store)
    assert postgres_store.read_bytes(prompt_key)[0] == b"User edit"

    postgres_store.delete(prompt_key)
    config.seed_prompts(postgres_store)
    assert not postgres_store.exists(prompt_key)
