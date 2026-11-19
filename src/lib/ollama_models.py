"""Installed Ollama chat models, served to chats through Pi as the custom provider `ollama`.

- Consumes Ollama's API at `config.OLLAMA_BASE_URL`: `/api/tags` (installed models) and `/api/show` (capabilities,
  context window), never starts Ollama
- Provides `installed_chat_models` (the tags of the installed chat models) and `sync_pi_provider`, which writes them
  into Pi's `models.json` (`$PI_CODING_AGENT_DIR`, default `~/.pi/agent`) so pi-host's gateway serves `ollama/<tag>`

pi-host reloads `models.json` when it changes, so a newly pulled model is served after the next sync.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

import httpx

from config import OLLAMA_BASE_URL

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


def models_json_path() -> Path:
    """Pi's `models.json`, in `$PI_CODING_AGENT_DIR` (default `~/.pi/agent`) like Pi resolves it."""
    agent_dir = os.environ.get("PI_CODING_AGENT_DIR")
    return Path(agent_dir).expanduser() / "models.json" if agent_dir else Path.home() / ".pi" / "agent" / "models.json"


def _write_atomically(path: Path, config: dict[str, Any]) -> None:
    """Replace *path* with *config* in one rename, keeping an existing file's mode (new files are owner-only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staging = tempfile.mkstemp(dir=path.parent, prefix=".models-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(config, file, indent=2, ensure_ascii=False)
            file.write("\n")
        if path.exists():
            os.chmod(staging, path.stat().st_mode & 0o777)
        os.replace(staging, path)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise


def sync_pi_provider() -> bool:
    """Merge the installed chat models into Pi's provider `ollama`, True when `models.json` was rewritten."""
    # Other providers and keys stay. Ollama down, an unreadable file or an unchanged provider writes nothing.
    details = _installed_chat_details()
    if details is None:
        return False
    path = models_json_path()
    try:
        config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        log.exception("Could not read Pi's %s, Ollama models are not synced", path)
        return False
    if not isinstance(config, dict) or not isinstance(config.get("providers", {}), dict):
        log.error("Pi's %s has no providers object, Ollama models are not synced", path)
        return False
    providers = config.get("providers", {})
    if not details and PROVIDER_ID not in providers:
        return False
    provider = {
        **providers.get(PROVIDER_ID, {}),
        "baseUrl": f"{OLLAMA_BASE_URL.rstrip('/')}/v1",
        "api": "openai-completions",
        "apiKey": "ollama",  # Ollama ignores it, Pi lists a provider only with credentials.
        "models": [_pi_model(tag, model_details) for tag, model_details in details.items()],
    }
    if providers.get(PROVIDER_ID) == provider:
        return False
    try:
        _write_atomically(path, {**config, "providers": {**providers, PROVIDER_ID: provider}})
    except OSError:
        log.exception("Could not write Pi's %s, Ollama models are not synced", path)
        return False
    log.info("Synced %d Ollama models into Pi's %s", len(details), path)
    return True
