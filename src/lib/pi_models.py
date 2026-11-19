"""Pi as a model source: chats reach Pi's logins through pi-host's completion gateway.

- Consumes pi-host's OpenAI-compatible gateway (`pi_host.gateway_url()`, bearer `pi_host.token()`): `/v1/models`
  lists Pi's providers (Anthropic subscription and API keys, D13) with each model's `thinking_levels`
- Provides the LiteLLM routing of the `litellm_proxy/` prefix, the gateway catalog for the settings panel, and each
  model's effort ladder for `reasoning_levels` and LearnHarness

A Pi model is an ordinary selector string on every call path, `litellm_proxy/anthropic/claude-opus-4-5`. LiteLLM
resolves that prefix from `LITELLM_PROXY_API_BASE`/`_KEY`, so the override reaches gateway models only and every other
provider keeps its own credentials.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from lib import pi_host
from lib.ttl_cache import TtlCache

log = logging.getLogger(__name__)

# The LiteLLM provider prefix of a gateway model. Stored selectors carry it, so it never changes.
LITELLM_ID = "litellm_proxy"
# Stored selectors keep LiteLLM's provider ids, the gateway names these providers differently.
_GATEWAY_PROVIDER_ALIASES = {"gemini": "google"}


def gateway_model_id(selector: str) -> str:
    """The gateway's `<provider>/<id>` of a stored selector: `litellm_proxy/x` -> `x`, `gemini/m` -> `google/m`.

    Mirrored by the frontend's `gatewayModelId` (`src/frontend/lib/models.ts`), keep both in step.
    """
    if selector.startswith(f"{LITELLM_ID}/"):
        return selector.removeprefix(f"{LITELLM_ID}/")
    provider, separator, model = selector.partition("/")
    if not separator:
        return selector
    return f"{_GATEWAY_PROVIDER_ALIASES.get(provider, provider)}/{model}"


# Every gateway pick shares one provider row, labelled and sourced by these two values (the browser mirrors them).
PROVIDER_LABEL = "Pi"
SOURCE = "pi"

_CATALOG_TTL_SECONDS = 60.0
_REQUEST_TIMEOUT_SECONDS = 3.0

_PROVIDER_LABELS = {"openai": "OpenAI", "openrouter": "OpenRouter", "xai": "xAI", "zai": "Z.AI"}

# A catalog caches its failure too (a stopped pi-host is a normal state), thinking levels only a successful read.
_catalog = TtlCache[dict[str, Any]](_CATALOG_TTL_SECONDS)
_thinking = TtlCache[dict[str, list[str]]](_CATALOG_TTL_SECONDS)


class GatewayUnavailable(RuntimeError):
    """pi-host's gateway does not answer, has no token yet, or sent an unreadable model list."""


def configure_litellm_proxy() -> None:
    """Route the `litellm_proxy/` prefix at pi-host's gateway, keeping an operator's own wiring (setdefault)."""
    os.environ.setdefault("LITELLM_PROXY_API_BASE", pi_host.gateway_url())
    try:
        os.environ.setdefault("LITELLM_PROXY_API_KEY", pi_host.token())
    except pi_host.PiHostUnavailable as exc:
        log.warning("Gateway models are unreachable until the backend restarts after pi-host: %s", exc)


def provider_label(provider_id: str) -> str:
    known = _PROVIDER_LABELS.get(provider_id)
    return known if known else provider_id.replace("-", " ").replace("_", " ").title()


def catalog_entries() -> list[dict[str, Any]]:
    """The gateway's raw `/v1/models` rows, uncached. Raises `GatewayUnavailable`."""
    try:
        response = httpx.get(
            f"{pi_host.gateway_url()}/models",
            headers={"Authorization": f"Bearer {pi_host.token()}"},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except (pi_host.PiHostUnavailable, httpx.HTTPError, ValueError) as exc:
        raise GatewayUnavailable(f"no Pi gateway at {pi_host.gateway_url()} ({exc})") from exc
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise GatewayUnavailable(f"the Pi gateway at {pi_host.gateway_url()} sent no model list")
    return [entry for entry in entries if isinstance(entry, dict)]


def entry_thinking_levels(entry: dict[str, Any]) -> list[str]:
    """Pi's thinking ladder of a `/v1/models` row: `off` only when thinking can be off, empty when the model cannot reason."""
    levels = entry.get("thinking_levels")
    return [level for level in levels if isinstance(level, str)] if isinstance(levels, list) else []


def group_by_provider(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold the flat gateway model list into its providers."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        model_id = str(entry.get("id") or "").strip().lstrip("/")
        if not model_id:
            continue
        provider_id = str(entry.get("owned_by") or "").strip() or model_id.split("/")[0]
        modalities = entry.get("input")
        grouped.setdefault(provider_id, []).append(
            {
                "id": model_id,
                "name": str(entry.get("name") or model_id),
                "vision": isinstance(modalities, list) and "image" in modalities,
                "context_window": entry.get("context_window"),
                "thinking_levels": entry_thinking_levels(entry),
            }
        )
    return [
        {"id": provider_id, "label": provider_label(provider_id), "models": sorted(models, key=lambda model: model["id"])}
        for provider_id, models in sorted(grouped.items())
    ]


def _installed() -> bool:
    """True once pi-host has started on this machine (it writes its token on first start)."""
    try:
        pi_host.token()
    except pi_host.PiHostUnavailable:
        return False
    return True


def _read_catalog() -> dict[str, Any]:
    result: dict[str, Any] = {
        "installed": _installed(),
        "online": False,
        "gateway": pi_host.gateway_url(),
        "provider_id": LITELLM_ID,
        "providers": [],
        "error": None,
    }
    if not result["installed"]:
        result["error"] = "pi-host has not started on this machine yet"
        return result
    try:
        result["providers"] = group_by_provider(catalog_entries())
        result["online"] = True
    except GatewayUnavailable as exc:
        result["error"] = str(exc)
    return result


def catalog(*, refresh: bool = False) -> dict[str, Any]:
    """Pi's providers and their models, or why they are missing (`online: false` plus a reason, never a failure)."""
    return _catalog.get(_read_catalog, refresh=refresh)


def reset_cache() -> None:
    _catalog.clear()
    _thinking.clear()


def thinking_registry() -> dict[str, list[str]]:
    """`<provider>/<model>` -> its effort ladder, empty for non-reasoning models, uncached. Raises `GatewayUnavailable`."""
    return {str(entry.get("id")): entry_thinking_levels(entry) for entry in catalog_entries() if entry.get("id")}


def served_ladders() -> dict[str, list[str]] | None:
    """Every model the gateway serves -> its effort ladder, cached, None when the gateway is unreachable."""
    try:
        return _thinking.get(thinking_registry)
    except GatewayUnavailable:
        return None


def thinking_levels(model_id: str) -> list[str] | None:
    """Effort levels the gateway accepts for `<provider>/<model>`, None when it is unreachable or lacks the model."""
    ladders = served_ladders()
    return None if ladders is None else ladders.get(model_id)
