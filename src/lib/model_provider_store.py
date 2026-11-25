"""User-managed model providers, each a selector prefix and its model names."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any

import yaml

from config import MODEL_DEFAULT_KEYS, TEST_MODEL
from lib import pi_models
from lib.data_store import DataStore, StorageNotFoundError, overwrite_text, read_text
from lib.storage_namespace import MODEL

MODEL_PROVIDERS_FILE = "model-providers.yaml"
MODEL_DEFAULTS_FILE = "model-defaults.yaml"
MODEL_TAB_ORDER_FILE = "model-tab-order.yaml"

# Seeds a first editable catalog of provider shells only. Models are the
# user's choice: an empty list keeps a provider out of the model selector
# while still offering it under Providers / Models. Ollama remains dynamic.
DEFAULT_PROVIDERS = {
    "Gemini": {"provider_id": "gemini", "models": []},
    "DeepSeek": {"provider_id": "deepseek", "models": []},
    "OpenRouter": {"provider_id": "openrouter", "models": []},
}
OPENAI_PROVIDER = {"provider_id": "openai", "models": ["gpt-5.6"]}
# Seed values only: the user picks real models in the model selector, and those are authoritative.
DEFAULT_MODEL_DEFAULTS = dict.fromkeys(MODEL_DEFAULT_KEYS, TEST_MODEL)

# Before Pi the gateway row was labelled `OMP` and sourced `omp`, in older catalogs one row per login
# ("OMP · Anthropic"). The store rewrites such rows once into the single Pi row; their selectors stay as stored.
LEGACY_SOURCE = "omp"
LEGACY_LABEL = "OMP"
# Before Pi the sandbox model default was stored as `omp_model`. Loading rewrites it once as `sandbox_model`.
LEGACY_SANDBOX_MODEL_KEY = "omp_model"
SANDBOX_MODEL_KEY = "sandbox_model"
# The old memory system's extraction model. Loading drops it once, keeping the other defaults.
LEGACY_MEMORY_MODEL_KEY = "memory_model"
# Before Pi a provider's selector prefix was stored as `litellm_id`. Loading rewrites it once as `provider_id`.
LEGACY_PROVIDER_ID_KEY = "litellm_id"
PROVIDER_ID_KEY = "provider_id"


def providers_for_ui(providers: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The stored catalog plus the OpenAI default when `OPENAI_API_KEY` is set."""
    if os.environ.get("OPENAI_API_KEY") and "OpenAI" not in providers:
        return {**providers, "OpenAI": OPENAI_PROVIDER}
    return providers


def _migrated_providers(providers: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]] | None:
    """*providers* with every legacy gateway row merged into the Pi row, in place of the first; None without one."""
    legacy = [provider for provider in providers.values() if provider.get("source") == LEGACY_SOURCE]
    if not legacy:
        return None
    models: list[str] = []
    for provider in legacy:
        models.extend(model for model in provider["models"] if model not in models)
    pi_row = {"provider_id": legacy[0]["provider_id"], "models": models, "source": pi_models.SOURCE}
    migrated: dict[str, dict[str, Any]] = {}
    for label, provider in providers.items():
        if provider.get("source") != LEGACY_SOURCE:
            migrated[label] = provider
        elif pi_models.PROVIDER_LABEL not in migrated:
            migrated[pi_models.PROVIDER_LABEL] = pi_row
    return migrated


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


def _migrated_defaults(defaults: object) -> dict[str, Any] | None:
    """*defaults* with the legacy sandbox key renamed in place and the memory key dropped, None when there is nothing to migrate."""
    if not isinstance(defaults, dict):
        return None
    rename = LEGACY_SANDBOX_MODEL_KEY in defaults and SANDBOX_MODEL_KEY not in defaults
    if not rename and LEGACY_MEMORY_MODEL_KEY not in defaults:
        return None
    return {
        SANDBOX_MODEL_KEY if rename and key == LEGACY_SANDBOX_MODEL_KEY else key: value
        for key, value in defaults.items()
        if key != LEGACY_MEMORY_MODEL_KEY
    }


def _renamed_provider_ids(data: object) -> dict[str, Any] | None:
    """*data* with every legacy `litellm_id` key renamed in place, None when there is nothing to migrate."""
    if not isinstance(data, dict) or not any(isinstance(raw, dict) and LEGACY_PROVIDER_ID_KEY in raw for raw in data.values()):
        return None
    return {
        label: {PROVIDER_ID_KEY if key == LEGACY_PROVIDER_ID_KEY else key: value for key, value in raw.items()}
        if isinstance(raw, dict) and PROVIDER_ID_KEY not in raw
        else raw
        for label, raw in data.items()
    }


@dataclass
class Provider:
    """One catalog entry: a selector prefix, its model names, and an optional origin marker."""

    provider_id: str
    models: list[str]
    # Where the models came from (today only `pi_models.SOURCE`, the gateway row), so the browser can group provider rows.
    # Absent for hand-entered providers, so existing catalogs round-trip byte-wise.
    source: str | None = None

    @classmethod
    def from_raw(cls, label: str, raw: object) -> Provider:
        if not isinstance(raw, dict):
            raise ValueError(f"Provider '{label}' must be a mapping")
        provider_id = _nonempty(raw.get(PROVIDER_ID_KEY), f"Provider '{label}' needs a provider id without '/'")
        if "/" in provider_id:
            raise ValueError(f"Provider '{label}' needs a provider id without '/'")
        models = raw.get("models")
        if not isinstance(models, list):
            raise ValueError(f"Provider '{label}' needs a list of non-empty model names")
        cleaned = [_nonempty(model, f"Provider '{label}' needs a list of non-empty model names").lstrip("/") for model in models]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError(f"Provider '{label}' contains duplicate models")
        source = raw.get("source")
        return cls(provider_id, cleaned, None if source is None else _nonempty(source, f"Provider '{label}' has an invalid source"))

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
        """Return the catalog, seeding `DEFAULT_PROVIDERS` when none is stored and migrating legacy keys and gateway rows once."""
        stored = self._load_or_seed(MODEL_PROVIDERS_FILE, {}, DEFAULT_PROVIDERS, allow_unicode=True)
        renamed = _renamed_provider_ids(stored)
        providers = _validate(stored if renamed is None else renamed)
        migrated = _migrated_providers(providers)
        if migrated is not None:
            return self.save(migrated)
        return providers if renamed is None else self.save(providers)

    def save(self, providers: object) -> dict[str, dict[str, Any]]:
        cleaned = _validate(providers)
        self._dump_yaml(MODEL_PROVIDERS_FILE, cleaned, allow_unicode=True)
        return cleaned

    def load_defaults(self) -> dict[str, str]:
        """Return the stored defaults, migrating a legacy key once and resetting any file that does not match the current keys."""
        defaults = self._load_or_seed(MODEL_DEFAULTS_FILE, {}, dict(DEFAULT_MODEL_DEFAULTS))
        migrated = _migrated_defaults(defaults)
        try:
            _checked_defaults(defaults if migrated is None else migrated)
        except ValueError:
            defaults = dict(DEFAULT_MODEL_DEFAULTS)
            self._dump_yaml(MODEL_DEFAULTS_FILE, defaults)
            return defaults
        if migrated is None:
            return defaults
        self._dump_yaml(MODEL_DEFAULTS_FILE, migrated)
        return migrated

    def save_defaults(self, defaults: object) -> dict[str, str]:
        cleaned = _checked_defaults(defaults)
        self._dump_yaml(MODEL_DEFAULTS_FILE, cleaned)
        return cleaned

    def load_tab_order(self) -> list[str]:
        """Persisted selector-tab order, e.g. `["Ollama", "Pi", "OpenAI"]`, a legacy `OMP` entry renamed once.

        Entries no longer present in the catalog are harmless leftovers; the
        caller ranks known tabs by position here and appends any unlisted
        tab alphabetically, so a stale or empty order never hides a tab.
        """
        try:
            order = self._load_yaml(MODEL_TAB_ORDER_FILE, [])
        except StorageNotFoundError:
            return []
        labels = _labels(order, "model-tab-order.yaml must be a list of non-empty tab labels")
        if LEGACY_LABEL not in labels or pi_models.PROVIDER_LABEL in labels:
            return labels
        return self.save_tab_order([pi_models.PROVIDER_LABEL if label == LEGACY_LABEL else label for label in labels])

    def save_tab_order(self, order: object) -> list[str]:
        cleaned = _labels(order, "Tab order must be a list of non-empty labels")
        self._dump_yaml(MODEL_TAB_ORDER_FILE, cleaned, allow_unicode=True)
        return cleaned
