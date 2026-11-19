import pytest

from lib.data_store import StorageError
from lib.model_provider_store import (
    DEFAULT_MODEL_DEFAULTS,
    MODEL_DEFAULTS_FILE,
    MODEL_PROVIDERS_FILE,
    MODEL_TAB_ORDER_FILE,
    OPENAI_PROVIDER,
    ModelProviderStore,
    providers_for_ui,
)
from lib.storage_namespace import MODEL


def test_provider_catalog_round_trips_yaml_and_rejects_duplicates(data_store):
    catalog = ModelProviderStore(data_store)
    providers = {"Together": {"litellm_id": "together_ai", "models": ["meta-llama/Llama-3.3-70B"]}}

    assert catalog.save(providers) == providers
    assert catalog.load() == providers
    content, _ = data_store.read_bytes(f"{MODEL}/{MODEL_PROVIDERS_FILE}")
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


def test_model_defaults_file_with_unknown_keys_is_reset(data_store):
    """Dev setting: a mismatched file is replaced, never migrated."""
    data_store.write_bytes(f"{MODEL}/{MODEL_DEFAULTS_FILE}", b"default_model: provider/only-key\n")

    defaults = ModelProviderStore(data_store).load_defaults()

    assert defaults == DEFAULT_MODEL_DEFAULTS


def test_legacy_gateway_rows_migrate_once_into_the_pi_row(data_store):
    """Pre-Pi `omp` rows, per login or already folded, become one `Pi` row; stored selectors keep their spelling."""
    catalog = ModelProviderStore(data_store)
    catalog.save({
        "OMP · Anthropic": {"litellm_id": "litellm_proxy", "models": ["anthropic/claude-opus-5"], "source": "omp"},
        "Gemini": {"litellm_id": "gemini", "models": ["gemini-3.1-pro"]},
        "OMP · OpenAI Codex": {
            "litellm_id": "litellm_proxy",
            "models": ["openai-codex/gpt-5.2", "anthropic/claude-opus-5"],
            "source": "omp",
        },
    })

    providers = catalog.load()

    # Merged in row order at the first legacy row's position, deduped, the other provider untouched.
    assert list(providers) == ["Pi", "Gemini"]
    assert providers["Pi"] == {
        "litellm_id": "litellm_proxy",
        "models": ["anthropic/claude-opus-5", "openai-codex/gpt-5.2"],
        "source": "pi",
    }
    content, _ = data_store.read_bytes(f"{MODEL}/{MODEL_PROVIDERS_FILE}")
    assert "omp" not in content.decode()
    assert catalog.load() == providers


def test_legacy_tab_order_entry_is_renamed_once(data_store):
    data_store.write_bytes(f"{MODEL}/{MODEL_TAB_ORDER_FILE}", b"- OMP\n- OpenAI\n")
    catalog = ModelProviderStore(data_store)

    assert catalog.load_tab_order() == ["Pi", "OpenAI"]
    content, _ = data_store.read_bytes(f"{MODEL}/{MODEL_TAB_ORDER_FILE}")
    assert content.decode() == "- Pi\n- OpenAI\n"
