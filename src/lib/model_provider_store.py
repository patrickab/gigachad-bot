"""User-managed LiteLLM model providers."""

from __future__ import annotations

import os
from typing import Any

import yaml

from config import MODEL_DEFAULT_KEYS, TEST_MODEL
from lib import omp_source
from lib.data_store import DataStore, StorageNotFoundError, overwrite_text, read_text

MODEL_PROVIDERS_FILE = "model-providers.yaml"
MODEL_DEFAULTS_FILE = "model-defaults.yaml"
MODEL_TAB_ORDER_FILE = "model-tab-order.yaml"

# Seeds a first editable catalog of provider shells only. Models are the
# user's choice: an empty list keeps a provider out of the model selector
# while still offering it under Providers / Models. Ollama remains dynamic.
DEFAULT_PROVIDERS = {
    "Gemini": {"litellm_id": "gemini", "models": []},
    "DeepSeek": {"litellm_id": "deepseek", "models": []},
    "OpenRouter": {"litellm_id": "openrouter", "models": []},
}
OPENAI_PROVIDER = {"litellm_id": "openai", "models": ["gpt-5.6"]}
# Seed values only: the user picks real models in the model selector, and those are authoritative.
DEFAULT_MODEL_DEFAULTS = dict.fromkeys(MODEL_DEFAULT_KEYS, TEST_MODEL)


def providers_for_ui(providers: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Shape the stored catalog for the browser.

    Folds legacy per-login OMP rows into one, and adds the OpenAI default only
    when its LiteLLM credential is available.
    """
    visible = omp_source.fold_provider_rows(providers)
    if os.environ.get("OPENAI_API_KEY") and "OpenAI" not in visible:
        return {**visible, "OpenAI": OPENAI_PROVIDER}
    return visible


class ModelProviderStore:
    """Persists the editable provider catalog in the configured storage root."""

    def __init__(self, store: DataStore, *, prefix: str = "") -> None:
        self._store = store
        self._prefix = prefix

    def _key(self, filename: str) -> str:
        return f"{self._prefix}/{filename}" if self._prefix else filename

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
            entry: dict[str, Any] = {"litellm_id": litellm_id.strip(), "models": cleaned}
            # An optional origin marker (today only "omp") so the browser can
            # group provider rows by where their models came from. Absent for
            # hand-entered providers, so existing catalogs round-trip byte-wise.
            source = provider.get("source")
            if source is not None and (not isinstance(source, str) or not source.strip()):
                raise ValueError(f"Provider '{label}' has an invalid source")
            if source:
                entry["source"] = source.strip()
            result[label.strip()] = entry
        return result

    def _load_yaml(self, filename: str, empty: object) -> object:
        """Parse a stored YAML file, `empty` for blank content. Raises StorageNotFoundError when absent."""
        raw, _ = read_text(self._store, self._key(filename))
        try:
            return yaml.safe_load(raw) or empty
        except yaml.YAMLError as exc:
            raise ValueError(f"{filename} is not valid YAML") from exc

    def _dump_yaml(self, filename: str, data: object, **options: bool) -> None:
        overwrite_text(self._store, self._key(filename), yaml.safe_dump(data, sort_keys=False, **options))

    def load(self) -> dict[str, dict[str, Any]]:
        """Return the catalog, seeding `DEFAULT_PROVIDERS` when none is stored."""
        try:
            return self._validate(self._load_yaml(MODEL_PROVIDERS_FILE, {}))
        except StorageNotFoundError:
            providers = self._validate(DEFAULT_PROVIDERS)
            self._dump_yaml(MODEL_PROVIDERS_FILE, providers, allow_unicode=True)
            return providers

    def save(self, providers: object) -> dict[str, dict[str, Any]]:
        cleaned = self._validate(providers)
        self._dump_yaml(MODEL_PROVIDERS_FILE, cleaned, allow_unicode=True)
        return cleaned

    def load_defaults(self) -> dict[str, str]:
        """Return the stored defaults, resetting any file that does not match the current keys."""
        try:
            defaults = self._load_yaml(MODEL_DEFAULTS_FILE, {})
        except StorageNotFoundError:
            defaults = {}
        usable = (
            isinstance(defaults, dict)
            and set(defaults) == set(DEFAULT_MODEL_DEFAULTS)
            and all(isinstance(value, str) and value.strip() for value in defaults.values())
        )
        if usable:
            return defaults
        defaults = dict(DEFAULT_MODEL_DEFAULTS)
        self._dump_yaml(MODEL_DEFAULTS_FILE, defaults)
        return defaults

    def save_defaults(self, defaults: object) -> dict[str, str]:
        if not isinstance(defaults, dict) or set(defaults) != set(DEFAULT_MODEL_DEFAULTS):
            raise ValueError("All default model values are required")
        if any(not isinstance(value, str) or not value.strip() for value in defaults.values()):
            raise ValueError("Default models must be non-empty strings")
        cleaned = {key: value.strip() for key, value in defaults.items()}
        self._dump_yaml(MODEL_DEFAULTS_FILE, cleaned)
        return cleaned

    @staticmethod
    def _clean_labels(order: object, message: str) -> list[str]:
        if not isinstance(order, list) or any(not isinstance(label, str) or not label.strip() for label in order):
            raise ValueError(message)
        return [label.strip() for label in order]

    def load_tab_order(self) -> list[str]:
        """Persisted selector-tab order, e.g. `["Ollama", "OMP", "OpenAI"]`.

        Entries no longer present in the catalog are harmless leftovers; the
        caller ranks known tabs by position here and appends any unlisted
        tab alphabetically, so a stale or empty order never hides a tab.
        """
        try:
            order = self._load_yaml(MODEL_TAB_ORDER_FILE, [])
        except StorageNotFoundError:
            return []
        return self._clean_labels(order, "model-tab-order.yaml must be a list of non-empty tab labels")

    def save_tab_order(self, order: object) -> list[str]:
        cleaned = self._clean_labels(order, "Tab order must be a list of non-empty labels")
        self._dump_yaml(MODEL_TAB_ORDER_FILE, cleaned, allow_unicode=True)
        return cleaned
