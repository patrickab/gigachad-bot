"""User-managed LiteLLM model providers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from llm_baseclient.config import MODELS_DEEPSEEK, MODELS_GEMINI

from config import MEMORY_MODEL, SMALL_MODEL, VISION_MODEL
from lib.data_store import DataStore, LocalDataStore, StorageError, StorageNotFoundError, read_text, write_text

MODEL_PROVIDERS_FILE = "model-providers.yaml"
MODEL_DEFAULTS_FILE = "model-defaults.yaml"

# This is only used to create a first editable catalog. Ollama remains dynamic.
DEFAULT_PROVIDERS = {
    "Gemini": {"litellm_id": "gemini", "models": MODELS_GEMINI},
    "DeepSeek": {"litellm_id": "deepseek", "models": MODELS_DEEPSEEK},
    "OpenRouter": {"litellm_id": "openrouter", "models": []},
}
OPENAI_PROVIDER = {"litellm_id": "openai", "models": ["gpt-5.6"]}
DEFAULT_MODEL_DEFAULTS = {
    "default_model": SMALL_MODEL,
    "small_model": SMALL_MODEL,
    "vision_model": VISION_MODEL,
    "memory_model": MEMORY_MODEL,
}


def providers_for_ui(providers: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Add the OpenAI default only when its LiteLLM credential is available."""
    if os.environ.get("OPENAI_API_KEY") and "OpenAI" not in providers:
        return {**providers, "OpenAI": OPENAI_PROVIDER}
    return providers


class ModelProviderStore:
    """Persists the editable provider catalog in the configured storage root."""

    def __init__(self, store: DataStore, *, fallback_dir: Path | None = None) -> None:
        self._store = store
        self._fallback = LocalDataStore(fallback_dir or Path.cwd())

    @staticmethod
    def _validate(data: object) -> dict[str, dict[str, Any]]:
        if not isinstance(data, dict):
            raise ValueError("Provider catalog must be a mapping of labels to providers")
        result: dict[str, dict[str, Any]] = {}
        for label, provider in data.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError("Provider labels must be non-empty strings")
            if not isinstance(provider, dict):
                raise ValueError(f"Provider '{label}' must be a mapping")
            litellm_id = provider.get("litellm_id")
            models = provider.get("models")
            if not isinstance(litellm_id, str) or not litellm_id.strip() or "/" in litellm_id:
                raise ValueError(f"Provider '{label}' needs a LiteLLM prefix without '/'")
            if not isinstance(models, list) or any(not isinstance(model, str) or not model.strip() for model in models):
                raise ValueError(f"Provider '{label}' needs a list of non-empty model names")
            cleaned = [model.strip().lstrip("/") for model in models]
            if len(cleaned) != len(set(cleaned)):
                raise ValueError(f"Provider '{label}' contains duplicate models")
            result[label.strip()] = {"litellm_id": litellm_id.strip(), "models": cleaned}
        return result

    def _read(self, store: DataStore) -> dict[str, dict[str, Any]]:
        raw, _ = read_text(store, MODEL_PROVIDERS_FILE)
        try:
            data = yaml.safe_load(raw) or {}
        except yaml.YAMLError as exc:
            raise ValueError("model-providers.yaml is not valid YAML") from exc
        return self._validate(data)

    def _write(self, store: DataStore, providers: dict[str, dict[str, Any]]) -> None:
        try:
            _, revision = read_text(store, MODEL_PROVIDERS_FILE)
        except StorageNotFoundError:
            revision = None
        write_text(store, MODEL_PROVIDERS_FILE, yaml.safe_dump(providers, allow_unicode=True, sort_keys=False), expected=revision)

    def load(self) -> dict[str, dict[str, Any]]:
        try:
            return self._read(self._store)
        except StorageNotFoundError:
            providers = self._validate(DEFAULT_PROVIDERS)
            try:
                self._write(self._store, providers)
            except StorageError:
                self._write(self._fallback, providers)
            return providers
        except StorageError:
            try:
                return self._read(self._fallback)
            except StorageNotFoundError:
                providers = self._validate(DEFAULT_PROVIDERS)
                self._write(self._fallback, providers)
                return providers

    def save(self, providers: object) -> dict[str, dict[str, Any]]:
        cleaned = self._validate(providers)
        try:
            self._write(self._store, cleaned)
        except StorageError:
            self._write(self._fallback, cleaned)
        return cleaned

    def load_defaults(self) -> dict[str, str]:
        try:
            raw, _ = read_text(self._store, MODEL_DEFAULTS_FILE)
        except StorageNotFoundError:
            defaults = dict(DEFAULT_MODEL_DEFAULTS)
            self._write_defaults(defaults)
            return defaults
        except StorageError:
            try:
                raw, _ = read_text(self._fallback, MODEL_DEFAULTS_FILE)
            except StorageNotFoundError:
                defaults = dict(DEFAULT_MODEL_DEFAULTS)
                write_text(self._fallback, MODEL_DEFAULTS_FILE, yaml.safe_dump(defaults, sort_keys=False))
                return defaults
        try:
            defaults = yaml.safe_load(raw) or {}
        except yaml.YAMLError as exc:
            raise ValueError("model-defaults.yaml is not valid YAML") from exc
        if set(defaults) != set(DEFAULT_MODEL_DEFAULTS) or any(not isinstance(value, str) or not value.strip() for value in defaults.values()):
            raise ValueError("model-defaults.yaml must define each default model as a non-empty string")
        return defaults

    def _write_defaults(self, defaults: dict[str, str]) -> None:
        try:
            _, revision = read_text(self._store, MODEL_DEFAULTS_FILE)
        except StorageNotFoundError:
            revision = None
        try:
            write_text(self._store, MODEL_DEFAULTS_FILE, yaml.safe_dump(defaults, sort_keys=False), expected=revision)
        except StorageError:
            write_text(self._fallback, MODEL_DEFAULTS_FILE, yaml.safe_dump(defaults, sort_keys=False))

    def save_defaults(self, defaults: object) -> dict[str, str]:
        if not isinstance(defaults, dict) or set(defaults) != set(DEFAULT_MODEL_DEFAULTS):
            raise ValueError("All default model values are required")
        if any(not isinstance(value, str) or not value.strip() for value in defaults.values()):
            raise ValueError("Default models must be non-empty strings")
        cleaned = {key: value.strip() for key, value in defaults.items()}
        self._write_defaults(cleaned)
        return cleaned
