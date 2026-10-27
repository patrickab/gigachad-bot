"""User-managed LiteLLM model providers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any

import yaml

from config import MODEL_DEFAULT_KEYS, TEST_MODEL
from lib import omp_source
from lib.data_store import DataStore, StorageNotFoundError, overwrite_text, read_text
from lib.storage_namespace import MODEL

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


def _nonempty(value: object, message: str) -> str:
    """*value* stripped, ValueError(*message*) unless it is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(message)
    return value.strip()


def _labels(order: object, message: str) -> list[str]:
    if not isinstance(order, list):
        raise ValueError(message)
    return [_nonempty(label, message) for label in order]


def _checked_defaults(defaults: object) -> dict[str, str]:
    if not isinstance(defaults, dict) or set(defaults) != set(DEFAULT_MODEL_DEFAULTS):
        raise ValueError("All default model values are required")
    return {key: _nonempty(value, "Default models must be non-empty strings") for key, value in defaults.items()}


@dataclass
class Provider:
    """One catalog entry: a LiteLLM prefix, its model names, and an optional origin marker."""

    litellm_id: str
    models: list[str]
    # Where the models came from (today only "omp"), so the browser can group provider rows.
    # Absent for hand-entered providers, so existing catalogs round-trip byte-wise.
    source: str | None = None

    @classmethod
    def from_raw(cls, label: str, raw: object) -> Provider:
        if not isinstance(raw, dict):
            raise ValueError(f"Provider '{label}' must be a mapping")
        litellm_id = _nonempty(raw.get("litellm_id"), f"Provider '{label}' needs a LiteLLM prefix without '/'")
        if "/" in litellm_id:
            raise ValueError(f"Provider '{label}' needs a LiteLLM prefix without '/'")
        models = raw.get("models")
        if not isinstance(models, list):
            raise ValueError(f"Provider '{label}' needs a list of non-empty model names")
        cleaned = [_nonempty(model, f"Provider '{label}' needs a list of non-empty model names").lstrip("/") for model in models]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError(f"Provider '{label}' contains duplicate models")
        source = raw.get("source")
        return cls(litellm_id, cleaned, None if source is None else _nonempty(source, f"Provider '{label}' has an invalid source"))

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if key != "source" or value}


def _validate(data: object) -> dict[str, dict[str, Any]]:
    if not isinstance(data, dict):
        raise ValueError("Provider catalog must be a mapping of labels to providers")
    result: dict[str, dict[str, Any]] = {}
    for label, provider in data.items():
        key = _nonempty(label, "Provider labels must be non-empty strings")
        result[key] = Provider.from_raw(label, provider).to_dict()
    return result


class ModelProviderStore:
    """Persists the editable provider catalog, defaults and tab order in the ``model`` namespace."""

    def __init__(self, store: DataStore) -> None:
        self._store = store

    def _load_yaml(self, filename: str, empty: object) -> object:
        """Parse a stored YAML file, `empty` for blank content. Raises StorageNotFoundError when absent."""
        raw, _ = read_text(self._store, f"{MODEL}/{filename}")
        try:
            return yaml.safe_load(raw) or empty
        except yaml.YAMLError as exc:
            raise ValueError(f"{filename} is not valid YAML") from exc

    def _dump_yaml(self, filename: str, data: object, **options: bool) -> None:
        overwrite_text(self._store, f"{MODEL}/{filename}", yaml.safe_dump(data, sort_keys=False, **options))

    def _load_or_seed(self, filename: str, empty: object, seed: object, **options: bool) -> object:
        """The stored YAML, or *seed* written in its place when the file is missing."""
        try:
            return self._load_yaml(filename, empty)
        except StorageNotFoundError:
            self._dump_yaml(filename, seed, **options)
            return seed

    def load(self) -> dict[str, dict[str, Any]]:
        """Return the catalog, seeding `DEFAULT_PROVIDERS` when none is stored."""
        return _validate(self._load_or_seed(MODEL_PROVIDERS_FILE, {}, DEFAULT_PROVIDERS, allow_unicode=True))

    def save(self, providers: object) -> dict[str, dict[str, Any]]:
        cleaned = _validate(providers)
        self._dump_yaml(MODEL_PROVIDERS_FILE, cleaned, allow_unicode=True)
        return cleaned

    def load_defaults(self) -> dict[str, str]:
        """Return the stored defaults, resetting any file that does not match the current keys."""
        defaults = self._load_or_seed(MODEL_DEFAULTS_FILE, {}, dict(DEFAULT_MODEL_DEFAULTS))
        try:
            _checked_defaults(defaults)
        except ValueError:
            defaults = dict(DEFAULT_MODEL_DEFAULTS)
            self._dump_yaml(MODEL_DEFAULTS_FILE, defaults)
        return defaults

    def save_defaults(self, defaults: object) -> dict[str, str]:
        cleaned = _checked_defaults(defaults)
        self._dump_yaml(MODEL_DEFAULTS_FILE, cleaned)
        return cleaned

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
        return _labels(order, "model-tab-order.yaml must be a list of non-empty tab labels")

    def save_tab_order(self, order: object) -> list[str]:
        cleaned = _labels(order, "Tab order must be a list of non-empty labels")
        self._dump_yaml(MODEL_TAB_ORDER_FILE, cleaned, allow_unicode=True)
        return cleaned
