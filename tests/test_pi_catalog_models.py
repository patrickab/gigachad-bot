"""Hand-entered catalog models Pi lacks, synced into Pi's `models.json` against a fake gateway list."""

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend.identity import get_request_identity
from backend.routes import models as models_route
from backend.routes.deps import get_model_provider_store
from lib import ollama_models, pi_catalog_models, pi_models
from tests.conftest import request_identity

# Providers pi-host has keys for, each with a model Pi's registry knows.
REGISTRY = {"openrouter/openai/gpt-5", "google/gemini-3.5-flash", "anthropic/claude-opus-5"}
OLLAMA = {"baseUrl": "http://localhost:11434/v1", "api": "openai-completions", "apiKey": "ollama", "models": [{"id": "qwen3:0.6b"}]}


def catalog(**rows: list[str]) -> dict[str, dict[str, Any]]:
    """A stored catalog with one hand-entered row per provider id, plus the Pi row."""
    providers: dict[str, dict[str, Any]] = {label: {"provider_id": label, "models": models} for label, models in rows.items()}
    providers["Pi"] = {"provider_id": "litellm_proxy", "models": ["anthropic/claude-made-up"], "source": "pi"}
    return providers


def models_json(agent_dir: Path) -> dict[str, Any]:
    return json.loads((agent_dir / "models.json").read_text())


def write_models_json(agent_dir: Path, config: dict[str, Any]) -> None:
    (agent_dir / "models.json").write_text(json.dumps(config))


def served_now(agent_dir: Path) -> set[str]:
    """What the gateway serves once it reloads `models.json`: the registry plus its non-Ollama entries."""
    path = agent_dir / "models.json"
    providers = json.loads(path.read_text())["providers"] if path.exists() else {}
    added = {
        f"{provider}/{model['id']}"
        for provider, config in providers.items()
        if provider != "ollama"
        for model in config.get("models", [])
    }
    return REGISTRY | added


def test_adds_a_typed_model_the_registry_lacks_and_keeps_the_rest(scratch_pi_agent_dir: Path) -> None:
    write_models_json(scratch_pi_agent_dir, {"$schema": "https://pi.dev/schemas/models.schema.json", "providers": {"ollama": OLLAMA}})

    assert pi_catalog_models.sync(catalog(openrouter=["acme/model-x", "openai/gpt-5"]), served_now(scratch_pi_agent_dir)) is True

    written = models_json(scratch_pi_agent_dir)
    assert written["$schema"] == "https://pi.dev/schemas/models.schema.json"
    assert written["providers"] == {"ollama": OLLAMA, "openrouter": {"models": [{"id": "acme/model-x"}]}}


def test_an_unchanged_catalog_rewrites_nothing(scratch_pi_agent_dir: Path) -> None:
    providers = catalog(openrouter=["acme/model-x"])
    pi_catalog_models.sync(providers, served_now(scratch_pi_agent_dir))
    before = (scratch_pi_agent_dir / "models.json").stat()

    # The gateway now serves the added model: it stays, nothing is written
    assert pi_catalog_models.sync(providers, served_now(scratch_pi_agent_dir)) is False

    after = (scratch_pi_agent_dir / "models.json").stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert models_json(scratch_pi_agent_dir)["providers"]["openrouter"]["models"] == [{"id": "acme/model-x"}]


def test_registry_models_keyless_providers_ollama_and_the_pi_row_are_left_alone(scratch_pi_agent_dir: Path) -> None:
    providers = catalog(openrouter=["openai/gpt-5"], deepseek=["deepseek-made-up"], ollama=["llama-typed"])

    assert pi_catalog_models.sync(providers, served_now(scratch_pi_agent_dir)) is False
    assert not (scratch_pi_agent_dir / "models.json").exists()


def test_a_stored_selector_maps_to_the_gateway_provider(scratch_pi_agent_dir: Path) -> None:
    assert pi_catalog_models.sync(catalog(gemini=["gemini-made-up"]), served_now(scratch_pi_agent_dir)) is True
    assert models_json(scratch_pi_agent_dir)["providers"] == {"google": {"models": [{"id": "gemini-made-up"}]}}


def test_a_removed_model_goes_again_and_user_entries_stay(scratch_pi_agent_dir: Path) -> None:
    mine = {"id": "acme/own", "name": "Own", "contextWindow": 32000}
    write_models_json(scratch_pi_agent_dir, {"providers": {"openrouter": {"compat": {"supportsStore": False}, "models": [mine]}}})
    pi_catalog_models.sync(catalog(openrouter=["acme/model-x", "acme/model-y"]), served_now(scratch_pi_agent_dir))

    assert pi_catalog_models.sync(catalog(openrouter=["acme/model-y"]), served_now(scratch_pi_agent_dir)) is True
    assert models_json(scratch_pi_agent_dir)["providers"]["openrouter"] == {
        "compat": {"supportsStore": False},
        "models": [mine, {"id": "acme/model-y"}],
    }

    assert pi_catalog_models.sync(catalog(openrouter=[]), served_now(scratch_pi_agent_dir)) is True
    assert models_json(scratch_pi_agent_dir)["providers"]["openrouter"] == {"compat": {"supportsStore": False}, "models": [mine]}


def test_a_provider_only_the_sync_created_is_removed_with_its_last_model(scratch_pi_agent_dir: Path) -> None:
    write_models_json(scratch_pi_agent_dir, {"providers": {"ollama": OLLAMA}})
    pi_catalog_models.sync(catalog(openrouter=["acme/model-x"]), served_now(scratch_pi_agent_dir))

    assert pi_catalog_models.sync(catalog(), served_now(scratch_pi_agent_dir)) is True
    assert models_json(scratch_pi_agent_dir)["providers"] == {"ollama": OLLAMA}


def test_an_invalid_file_is_never_overwritten(scratch_pi_agent_dir: Path) -> None:
    (scratch_pi_agent_dir / "models.json").write_text("{not json")
    assert pi_catalog_models.sync(catalog(openrouter=["acme/model-x"]), REGISTRY) is False
    assert (scratch_pi_agent_dir / "models.json").read_text() == "{not json"


class OpenRouterStore:
    """The model provider store `/api/models` reads: an OpenRouter row with a typed model."""

    def load(self) -> dict:
        return catalog(openrouter=["acme/model-x"])

    def load_defaults(self) -> dict:
        return {}

    def load_tab_order(self) -> list[str]:
        return []


def test_models_route_adds_the_model_and_lists_it_available(scratch_pi_agent_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama_models, "sync_pi_provider", lambda: False)
    monkeypatch.setattr(ollama_models, "installed_chat_models", lambda: [])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(pi_models, "served_ladders", lambda: {key: [] for key in served_now(scratch_pi_agent_dir)})
    app = FastAPI()
    app.include_router(models_route.router)
    app.dependency_overrides[get_model_provider_store] = OpenRouterStore
    app.dependency_overrides[get_request_identity] = request_identity

    response = TestClient(app).get("/api/models")

    assert response.status_code == 200
    assert models_json(scratch_pi_agent_dir)["providers"]["openrouter"]["models"] == [{"id": "acme/model-x"}]
    assert response.json()["unavailable"] == ["litellm_proxy/anthropic/claude-made-up"]
