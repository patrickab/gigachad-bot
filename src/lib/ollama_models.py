"""Installed Ollama chat models.

- Consumes Ollama's API at `config.OLLAMA_BASE_URL`: `/api/tags` (installed models) and `/api/show` (capabilities,
  context window), never starts Ollama
- Provides `installed_chat_models`, the tags of the installed chat models

Pi runs an Ollama model only when the user's own `models.json` has a provider `ollama` that lists it: this app never
writes Pi's files.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config import OLLAMA_BASE_URL

log = logging.getLogger(__name__)

# Short: the backend start and `/api/models` wait on it, and a local Ollama answers in milliseconds.
_TIMEOUT_SECONDS = 3.0
# `/api/show` by model digest: a digest's details never change, so repeated listings only read the tags.
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
        with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=_TIMEOUT_SECONDS) as client:
            response = client.get("/api/tags")
            response.raise_for_status()
            listed = response.json().get("models") or []
            shown = {model["name"]: _show(client, model["name"], model.get("digest", "")) for model in listed}
    except httpx.HTTPError:
        return None
    # Embedding models (`nomic-embed-text`, `bge-m3`) lack the `completion` capability.
    return {tag: details for tag, details in shown.items() if details and "completion" in details.get("capabilities", [])}


def installed_chat_models() -> list[str]:
    """Tags of the installed Ollama chat models, [] when Ollama is not running."""
    return list(_installed_chat_details() or {})
