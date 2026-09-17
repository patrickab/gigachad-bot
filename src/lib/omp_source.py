"""OMP (Oh My Pi) as a model source.

OMP owns its own provider logins, including OAuth subscriptions such as Claude
Pro/Max and the Codex plan, and exposes every credential it holds through
``omp auth-gateway`` — an OpenAI-compatible forward proxy that dispatches with
OMP's own provider shaping. Pointing LiteLLM's ``litellm_proxy/`` prefix at
that gateway is enough to reach those subscriptions from chat, so an OMP model
is an ordinary ``provider/model`` string on every existing call path:

    litellm_proxy/anthropic/claude-opus-5

LiteLLM resolves the base URL for that prefix from ``LITELLM_PROXY_API_BASE``,
which scopes the override to OMP models only and leaves every other provider
on its own credentials. No route, client, or dispatch code changes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.request

# The LiteLLM provider prefix that carries an OMP model. Kept in one place
# because the browser needs the same value to build selector strings.
LITELLM_ID = "litellm_proxy"

# Every OMP pick shares one provider row, so the model selector shows a single
# OMP tab listing `<omp-provider>/<model>` rather than a tab per login.
PROVIDER_LABEL = "OMP"
SOURCE = "omp"

DEFAULT_GATEWAY_URL = "http://127.0.0.1:4000/v1"
# A --no-auth gateway ignores the bearer, but LiteLLM refuses an empty key.
FALLBACK_TOKEN = "omp-gateway"

_CATALOG_TTL_SECONDS = 60.0
_REQUEST_TIMEOUT_SECONDS = 3.0

# Provider ids the gateway reports verbatim; only the ones whose casing cannot
# be derived from the id are listed.
_PROVIDER_LABELS = {
    "ai-gateway": "Vercel AI Gateway",
    "github-copilot": "GitHub Copilot",
    "google": "Google",
    "google-antigravity": "Antigravity",
    "google-gemini-cli": "Gemini CLI",
    "google-vertex": "Vertex AI",
    "lm-studio": "LM Studio",
    "openai": "OpenAI",
    "openai-codex": "OpenAI Codex",
    "openrouter": "OpenRouter",
    "xai": "xAI",
    "zai": "Z.AI",
    "zai-coding-plan": "Z.AI Coding Plan",
}

_cache: tuple[float, dict[str, Any]] | None = None


def omp_home() -> Path:
    """OMP's config directory, honoring its own PI_CONFIG_DIR override."""
    override = os.environ.get("PI_CONFIG_DIR")
    return Path(override) if override else Path.home() / ".omp"


def installed() -> bool:
    """True when this machine has an OMP credential store to serve from."""
    return (omp_home() / "agent" / "agent.db").exists()


def gateway_url() -> str:
    return (os.environ.get("GIGACHAD_OMP_GATEWAY_URL") or DEFAULT_GATEWAY_URL).rstrip("/")


def gateway_token() -> str:
    explicit = os.environ.get("GIGACHAD_OMP_GATEWAY_TOKEN")
    if explicit:
        return explicit
    try:
        return (omp_home() / "auth-gateway.token").read_text(encoding="utf-8").strip() or FALLBACK_TOKEN
    except OSError:
        return FALLBACK_TOKEN


def configure_litellm_proxy() -> None:
    """Route the ``litellm_proxy/`` prefix at the OMP gateway.

    Uses setdefault so an operator who already points that prefix somewhere
    else keeps their own wiring.
    """
    os.environ.setdefault("LITELLM_PROXY_API_BASE", gateway_url())
    os.environ.setdefault("LITELLM_PROXY_API_KEY", gateway_token())


def provider_label(provider_id: str) -> str:
    known = _PROVIDER_LABELS.get(provider_id)
    return known if known else provider_id.replace("-", " ").replace("_", " ").title()


def model_selector(model_id: str) -> str:
    """The selector gigachad stores and sends for an OMP model id."""
    return f"{LITELLM_ID}/{model_id}"


def _fetch_catalog_entries() -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"{gateway_url()}/models",
        headers={"Authorization": f"Bearer {gateway_token()}"},
    )
    with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))
    entries = payload.get("data") if isinstance(payload, dict) else None
    return [entry for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []


def group_by_provider(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold a flat OpenAI model list into the gateway's authenticated providers."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        model_id = str(entry.get("id") or "").strip().lstrip("/")
        if not model_id:
            continue
        owner = str(entry.get("owned_by") or "").strip()
        provider_id = owner or model_id.split("/")[0]
        modalities = entry.get("input_modalities")
        grouped.setdefault(provider_id, []).append(
            {
                "id": model_id,
                "name": str(entry.get("display_name") or model_id),
                "vision": isinstance(modalities, list) and "image" in modalities,
                "context_window": entry.get("context_length"),
            }
        )
    return [
        {
            "id": provider_id,
            "label": provider_label(provider_id),
            "models": sorted(models, key=lambda model: model["id"]),
        }
        for provider_id, models in sorted(grouped.items())
    ]


def catalog(*, refresh: bool = False) -> dict[str, Any]:
    """Authenticated OMP providers and their models, or why they are missing.

    An unreachable gateway is a normal state, not an error: OMP may be
    installed while its services are down, so the browser gets
    ``online: false`` plus a reason instead of a failed request.
    """
    global _cache
    now = time.monotonic()
    if not refresh and _cache is not None and now - _cache[0] < _CATALOG_TTL_SECONDS:
        return _cache[1]

    result: dict[str, Any] = {
        "installed": installed(),
        "online": False,
        "gateway": gateway_url(),
        "litellm_id": LITELLM_ID,
        "providers": [],
        "error": None,
    }
    if result["installed"]:
        try:
            result["providers"] = group_by_provider(_fetch_catalog_entries())
            result["online"] = True
        except (urllib.error.URLError, OSError, ValueError) as exc:
            result["error"] = f"No OMP gateway at {result['gateway']} ({exc})"
    else:
        result["error"] = f"No OMP credential store at {omp_home() / 'agent' / 'agent.db'}"

    _cache = (now, result)
    return result


def reset_cache() -> None:
    global _cache
    _cache = None


def fold_provider_rows(providers: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Collapse every OMP-sourced row into the single `OMP` row.

    Earlier builds wrote one row per OMP login ("OMP · Anthropic"), which the
    selector rendered as one tab each. Folding on read migrates those catalogs
    in place — the browser posts the folded snapshot back on its next save —
    and is a no-op once a catalog already has a single row.
    """
    sourced = [(label, provider) for label, provider in providers.items() if provider.get("source") == SOURCE]
    if not sourced or (len(sourced) == 1 and sourced[0][0] == PROVIDER_LABEL):
        return providers

    models: list[str] = []
    for _, provider in sourced:
        models.extend(model for model in provider["models"] if model not in models)
    merged = {"litellm_id": sourced[0][1]["litellm_id"], "models": models, "source": SOURCE}

    folded: dict[str, dict[str, Any]] = {}
    for label, provider in providers.items():
        if provider.get("source") != SOURCE:
            folded[label] = provider
        elif PROVIDER_LABEL not in folded:
            folded[PROVIDER_LABEL] = merged
    return folded
