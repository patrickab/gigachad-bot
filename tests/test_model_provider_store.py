import pytest

from lib.data_store import LocalDataStore, StorageError
from lib.model_provider_store import MODEL_PROVIDERS_FILE, ModelProviderStore, providers_for_ui


def test_provider_catalog_round_trips_yaml_and_rejects_duplicates(tmp_path):
    store = LocalDataStore(tmp_path)
    catalog = ModelProviderStore(store)
    providers = {"Together": {"litellm_id": "together_ai", "models": ["meta-llama/Llama-3.3-70B"]}}

    assert catalog.save(providers) == providers
    assert catalog.load() == providers
    assert "Together:" in (tmp_path / MODEL_PROVIDERS_FILE).read_text()

    try:
        catalog.save({"Together": {"litellm_id": "together_ai", "models": ["a", "a"]}})
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate model names must be rejected")


def test_provider_catalog_propagates_primary_store_errors():
    class UnavailableStore:
        def read_bytes(self, key):
            from lib.data_store import StorageError
            raise StorageError("offline")

        def write_bytes(self, key, content, *, expected=None):
            from lib.data_store import StorageError
            raise StorageError("offline")

    catalog = ModelProviderStore(UnavailableStore())

    with pytest.raises(StorageError, match="offline"):
        catalog.load()


def test_openai_provider_is_visible_only_with_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "OpenAI" not in providers_for_ui({})

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert providers_for_ui({})["OpenAI"] == {"litellm_id": "openai", "models": ["gpt-5.6"]}


def test_model_defaults_round_trip(tmp_path):
    catalog = ModelProviderStore(LocalDataStore(tmp_path))
    defaults = {
        "default_model": "openai/gpt-5.6",
        "small_model": "ollama/qwen3",
        "vision_model": "gemini/gemini-3.1-pro",
        "memory_model": "openai/gpt-5.6",
    }

    assert catalog.save_defaults(defaults) == defaults
    assert catalog.load_defaults() == defaults
