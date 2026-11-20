"""Hand-entered catalog models Pi's registry lacks, added to their Pi provider in `models.json` so Pi runs them.

- Consumes the stored model catalog (`ModelProviderStore.load()`) and the models pi-host's gateway serves
  (`pi_models.served_ladders()`)
- Provides `sync`: a model typed into a provider row (`acme/model-x` under OpenRouter) whose Pi provider the gateway
  serves but whose model it does not becomes `{"id": "acme/model-x"}` in `providers.openrouter.models` (Pi gives it the
  provider's API, base URL and key, its default window and no thinking). Removing it from the catalog removes it again

The Pi row (picked from what the gateway serves) and Ollama (`ollama_models`) never go through here. A provider the gateway
does not list gets nothing: pi-host has no key for it (Pi could not run the model) or Pi does not know it (a model id gives
no base URL or API).

`gigachad-models.json` beside `models.json` lists the entries this module added, so it never removes one the user wrote.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from lib import ollama_models, pi_models, pi_models_json

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

log = logging.getLogger(__name__)

OWNED_FILE = "gigachad-models.json"

# Pi provider -> ids of models in that provider's `models.json` entry.
Added = dict[str, list[str]]


def _owned_path() -> Path:
    return pi_models_json.agent_dir() / OWNED_FILE


def _owned() -> Added:
    """The entries an earlier sync added, none when the record is missing or unreadable (they then stay in `models.json`)."""
    path = _owned_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        log.warning("Could not read %s, catalog models added to Pi before stay", path)
        return {}
    providers = data.get("providers") if isinstance(data, dict) else None
    if not isinstance(providers, dict):
        return {}
    return {
        provider: [model for model in ids if isinstance(model, str)] for provider, ids in providers.items() if isinstance(ids, list)
    }


def _wanted(catalog: dict[str, dict[str, Any]], served: Collection[str], owned: Added) -> Added:
    """The entries the catalog needs: hand-entered models the gateway does not serve, or serves only through an earlier sync."""
    served_providers = {model_id.partition("/")[0] for model_id in served}
    result: Added = {}
    for provider in catalog.values():
        if provider.get("source"):
            continue
        for model in provider["models"]:
            gateway_id = pi_models.gateway_model_id(f"{provider['provider_id']}/{model}")
            pi_provider, _, model_id = gateway_id.partition("/")
            if not model_id or pi_provider == ollama_models.PROVIDER_ID or pi_provider not in served_providers:
                continue
            if gateway_id in served and model_id not in owned.get(pi_provider, []):
                continue
            ids = result.setdefault(pi_provider, [])
            if model_id not in ids:
                ids.append(model_id)
    return result


def _with_entries(providers: dict[str, Any], owned: Added, entries: Added) -> dict[str, Any] | None:
    """*providers* with the owned entries no longer wanted removed and the missing wanted ones added, None when unchanged."""
    result = dict(providers)
    for pi_provider in {*owned, *entries}:
        current = providers.get(pi_provider, {})
        models = current.get("models", []) if isinstance(current, dict) else None
        if not isinstance(models, list):
            log.warning("Pi's provider %s in models.json is not a mapping with a models list, left as it is", pi_provider)
            continue
        dropped = set(owned.get(pi_provider, [])) - set(entries.get(pi_provider, []))
        kept = [model for model in models if not (isinstance(model, dict) and model.get("id") in dropped)]
        present = {model.get("id") for model in kept if isinstance(model, dict)}
        updated = kept + [{"id": model_id} for model_id in entries.get(pi_provider, []) if model_id not in present]
        if updated == models:
            continue
        provider = {**current, "models": updated}
        if not updated:
            del provider["models"]
        if provider:
            result[pi_provider] = provider
        else:
            del result[pi_provider]
    return None if result == providers else result


def sync(catalog: dict[str, dict[str, Any]], served: Collection[str]) -> bool:
    """Bring Pi's `models.json` in line with the hand-entered models of *catalog*, True when it was rewritten.

    *served* is every `<provider>/<id>` the gateway serves now. Other providers, keys and entries stay as they are.
    """
    with pi_models_json.lock:
        owned = _owned()
        entries = _wanted(catalog, served, owned)
        try:
            written = pi_models_json.update(lambda providers: _with_entries(providers, owned, entries))
        except pi_models_json.ModelsJsonError:
            log.exception("Catalog models are not synced into Pi")
            return False
        if entries != owned:
            try:
                pi_models_json.write_json(_owned_path(), {"providers": entries})
            except OSError:
                log.exception("Could not record the catalog models added to Pi in %s", _owned_path())
    if written:
        log.info("Synced %d catalog models into Pi's %s", sum(map(len, entries.values())), pi_models_json.models_json_path())
    return written
