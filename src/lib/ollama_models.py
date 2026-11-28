"""Installed Ollama chat models, served to chats through Pi as the custom provider `ollama`.

- Consumes Ollama's API at `config.OLLAMA_BASE_URL`: `/api/tags` (installed models) and `/api/show` (capabilities,
  context window), never starts Ollama
- Provides `sync_pi_provider`, which writes the installed chat models into Pi's `models.json` (`pi_models_json`) so
  pi-host's gateway serves `ollama/<tag>`, and returns their tags

pi-host reloads `models.json` when it changes, so a newly pulled model is served after the next sync.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config import OLLAMA_BASE_URL
from lib import pi_models_json
from lib.upstream import tls_context

log = logging.getLogger(__name__)

PROVIDER_ID = "ollama"
# Short: the backend start and `/api/models` wait on it, and a local Ollama answers in milliseconds.
_TIMEOUT_SECONDS = 3.0
# `/api/show` by model digest: a digest's details never change, so repeated syncs only list the tags.
_details_by_digest: dict[str, dict[str, Any]] = {}


def _show(client: httpx.Client, tag: str, digest: str) -> dict[str, Any] | None:
    """`/api/show` of *tag*, cached by digest, None when Ollama does not answer it."""
    if digest in _details_by_digest:
        return _details_by_digest[digest]
    try:
        response = client.post("/api/show", json={"model": tag})
        response.raise_for_status()
    except httpx.HTTPError:
        log.warning("Ollama did not describe %s, leaving it out", tag)
        return None
    details = response.json()
    if digest:
        _details_by_digest[digest] = details
    return details


def _installed_chat_details() -> dict[str, dict[str, Any]] | None:
    """`/api/show` details of each installed chat model by tag, None when Ollama is not running."""
    try:
        with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=_TIMEOUT_SECONDS, verify=tls_context()) as client:
            response = client.get("/api/tags")
            response.raise_for_status()
            listed = response.json().get("models") or []
            shown = {model["name"]: _show(client, model["name"], model.get("digest", "")) for model in listed}
    except httpx.HTTPError:
        return None
    # Embedding models (`nomic-embed-text`, `bge-m3`) lack the `completion` capability.
    return {tag: details for tag, details in shown.items() if details and "completion" in details.get("capabilities", [])}


def _context_window(details: dict[str, Any]) -> int | None:
    """`/api/show` reports the window as `<architecture>.context_length` in `model_info`."""
    info = details.get("model_info") or {}
    context = info.get(f"{info.get('general.architecture')}.context_length")
    return context if isinstance(context, int) and context > 0 else None


def _pi_model(tag: str, details: dict[str, Any]) -> dict[str, Any]:
    """Pi's `models.json` entry of one Ollama model."""
    capabilities = details.get("capabilities", [])
    model: dict[str, Any] = {
        "id": tag,
        "input": ["text", "image"] if "vision" in capabilities else ["text"],
        "reasoning": "thinking" in capabilities,
    }
    if model["reasoning"]:
        # Ollama thinks unless told `none` and knows only low/medium/high, Pi sends nothing for `off` by default.
        model["thinkingLevelMap"] = {"off": "none", "minimal": None}
    if context := _context_window(details):
        model["contextWindow"] = context
    return model


def sync_pi_provider() -> list[str]:
    """Merge the installed chat models into Pi's provider `ollama` and return their tags, none when Ollama is not running."""
    # Other providers and keys stay. Ollama down, an unreadable file or an unchanged provider writes nothing.
    details = _installed_chat_details()
    if details is None:
        return []

    def with_installed(providers: dict[str, Any]) -> dict[str, Any] | None:
        if not details and PROVIDER_ID not in providers:
            return None
        provider = {
            **providers.get(PROVIDER_ID, {}),
            "baseUrl": f"{OLLAMA_BASE_URL.rstrip('/')}/v1",
            "api": "openai-completions",
            "apiKey": "ollama",  # Ollama ignores it, Pi lists a provider only with credentials.
            "models": [_pi_model(tag, model_details) for tag, model_details in details.items()],
        }
        return None if providers.get(PROVIDER_ID) == provider else {**providers, PROVIDER_ID: provider}

    try:
        written = pi_models_json.update(with_installed)
    except pi_models_json.ModelsJsonError:
        log.exception("Ollama models are not synced")
        return list(details)
    if written:
        log.info("Synced %d Ollama models into Pi's %s", len(details), pi_models_json.models_json_path())
    return list(details)
