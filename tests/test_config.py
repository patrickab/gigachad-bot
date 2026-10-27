import pytest

import config
from lib.postgres_data_store import PostgresDataStore
from tests.conftest import create_user


@pytest.fixture
def postgres_store(postgres_pool, clean_database):
    return PostgresDataStore(postgres_pool, create_user(postgres_pool))


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
