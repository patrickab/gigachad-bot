import os
from uuid import uuid4

from psycopg_pool import ConnectionPool
import pytest

from lib.data_store import StorageError
from lib.db_schema import upgrade
from lib.model_provider_store import (
    DEFAULT_MODEL_DEFAULTS,
    MODEL_DEFAULTS_FILE,
    MODEL_PROVIDERS_FILE,
    OPENAI_PROVIDER,
    ModelProviderStore,
    providers_for_ui,
)
from lib.postgres_data_store import PostgresDataStore


@pytest.fixture(scope="module")
def postgres_pool():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for Postgres-backed model provider store tests")
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


def test_provider_catalog_round_trips_yaml_and_rejects_duplicates(postgres_store):
    catalog = ModelProviderStore(postgres_store)
    providers = {"Together": {"litellm_id": "together_ai", "models": ["meta-llama/Llama-3.3-70B"]}}

    assert catalog.save(providers) == providers
    assert catalog.load() == providers
    content, _ = postgres_store.read_bytes(MODEL_PROVIDERS_FILE)
    assert "Together:" in content.decode()

    try:
        catalog.save({"Together": {"litellm_id": "together_ai", "models": ["a", "a"]}})
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate model names must be rejected")


def test_provider_catalog_propagates_primary_store_errors():
    class UnavailableStore:
        def read_bytes(self, key):
            raise StorageError("offline")

        def write_bytes(self, key, content, *, expected=None):
            raise StorageError("offline")

    catalog = ModelProviderStore(UnavailableStore())

    with pytest.raises(StorageError, match="offline"):
        catalog.load()


def test_openai_provider_is_visible_only_with_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "OpenAI" not in providers_for_ui({})

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert providers_for_ui({})["OpenAI"] == OPENAI_PROVIDER


def test_model_defaults_file_with_unknown_keys_is_reset(postgres_store):
    """Dev setting: a mismatched file is replaced, never migrated."""
    postgres_store.write_bytes(MODEL_DEFAULTS_FILE, b"default_model: provider/only-key\n")

    defaults = ModelProviderStore(postgres_store).load_defaults()

    assert defaults == DEFAULT_MODEL_DEFAULTS
