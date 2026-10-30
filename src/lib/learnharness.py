"""Server-to-server client for LearnHarness, the learn mode's teaching surface, and the generator of its model configuration.

- Unmodified upstream LearnHarness (`run-learnharness.sh`) on loopback, the browser frames it, every other read goes through here
- `GIGACHAD_LEARNHARNESS_URL` is LearnHarness for the backend (default 127.0.0.1:30181),
  `GIGACHAD_LEARNHARNESS_FRAME_URL` the origin the browser frames it from (the Tailnet gate in production)
- Learn sessions (section "Sessions"): list, create, first turn, rename, delete, Markdown export. Responses are validated
  against the contract models, a changed LearnHarness shape fails loudly with the field name
- `python -m lib.learnharness write-config` derives LearnHarness's models from gigachad's catalog (section below)
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import fcntl
import getpass
import importlib.metadata
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time
import tomllib
from typing import TYPE_CHECKING, Any, Literal
import urllib.error

import httpx
from llm_baseclient.config import discover_ollama_models
from psycopg.errors import UndefinedTable
from psycopg_pool import PoolTimeout
from pydantic import AliasGenerator, BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
import yaml

from config import OLLAMA_BASE_URL
from lib import omp_source
from lib.changes import UserScope
from lib.model_provider_store import ModelProviderStore, providers_for_ui
from lib.postgres_data_store import PostgresDataStore
from lib.postgres_pool import close_postgres_pool, get_postgres_pool
from lib.upstream import (
    TIMEOUT_SECONDS,
    UpstreamClient,
    UpstreamContractError,
    UpstreamRequestFailed,
    UpstreamUnavailable,
    error_message,
    upstream_url,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping
    from uuid import UUID

    from psycopg_pool import ConnectionPool

DEFAULT_URL = "http://127.0.0.1:30181"

client = UpstreamClient(env="GIGACHAD_LEARNHARNESS_URL", default=DEFAULT_URL, label="LearnHarness")


def learnharness_url() -> str:
    """Base URL the backend reaches LearnHarness at."""
    return upstream_url("GIGACHAD_LEARNHARNESS_URL", DEFAULT_URL)


def learnharness_frame_url() -> str:
    """Base URL the browser frames LearnHarness from, `GIGACHAD_LEARNHARNESS_FRAME_URL` when it sits behind the Tailnet gate."""
    return upstream_url("GIGACHAD_LEARNHARNESS_FRAME_URL", learnharness_url())


# ── Sessions ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# LearnHarness's chat API (`app/api/chats`, `app/api/chat`, v0.4.0) is internal and churns, so every URL and shape lives here.

# A turn is acknowledged once the turn gates passed, which can wait for a running context compaction.
TURN_START_TIMEOUT = httpx.Timeout(10.0, read=60.0)
# Deleting a session waits up to 10 s for its extraction to wind down.
DELETE_TIMEOUT_SECONDS = 30.0
# LearnHarness's error text for an unknown session (`deleteChatSession`); any other 404 is a moved or removed route.
NO_SUCH_SESSION = "No such session"


class ContractModel(BaseModel):
    """LearnHarness JSON is camelCase, gigachad's API is snake_case: validate by alias, serialize by field name."""

    model_config = ConfigDict(alias_generator=AliasGenerator(validation_alias=to_camel), extra="ignore", frozen=True)


class LearnSession(ContractModel):
    """A learn session: LearnHarness's `ChatListItem`, reduced to what the shell shows (its sidebar row's state included)."""

    id: str
    title: str
    # LearnHarness's session kind, "teach" or "practice"; a freshly created session reports none.
    kind: str | None = None
    # A teach turn is running.
    busy: bool = False
    # A context compaction runs.
    compacting: bool = False
    # A question waits for the learner's answer.
    pending_question: bool = False
    # Practice: a posed problem waits for the learner.
    pending_problem: bool = False
    # When the session last changed (ISO time), None for a freshly created one.
    updated_at: str | None = None


# LearnHarness's two deletes: the session goes, its Knowledge stays in the graph, or its graph traces go too.
DeleteMode = Literal["preserve-graph", "delete-nodes"]


@dataclass(frozen=True)
class SessionExport:
    """A session as LearnHarness exports it: one Markdown document and the file name it suggests."""

    filename: str
    markdown: str


class _SessionList(ContractModel):
    chats: list[LearnSession]


class _Deleted(ContractModel):
    ok: bool


class _Renamed(ContractModel):
    title: str


class _Accepted(ContractModel):
    """LearnHarness's acknowledgement of a turn: `POST /api/chat` answers `{ok: true}` and closes, the turn runs on."""

    ok: bool


async def list_sessions() -> list[LearnSession]:
    """Every learn session, most recently updated first."""
    return (await client.request("GET", "/api/chats", _SessionList)).chats


async def create_session(title: str) -> LearnSession:
    """A new, empty learn session. LearnHarness stamps it with its current strong model tier."""
    return await client.request("POST", "/api/chats", LearnSession, json={"title": title})


async def submit_text_turn(session_id: str, text: str) -> None:
    """Start a turn on *text* in session *session_id*.

    Returns once LearnHarness acknowledged the turn with JSON `{ok: true}`. The turn runs on without a reader.
    A refusal (unknown session, no model configured, a turn already running) raises `UpstreamRequestFailed` with its message,
    any other 2xx body `UpstreamContractError`.
    """
    body = {"chatId": session_id, "input": {"type": "text", "text": text}}
    await client.request("POST", "/api/chat", _Accepted, json=body, timeout=TURN_START_TIMEOUT)


async def rename_session(session_id: str, title: str) -> str:
    """Rename session *session_id* to *title*, returning the title LearnHarness stored (trimmed).

    LearnHarness keeps the session's place in its list. It refuses a blank title (400) and an unknown session (404,
    `UpstreamRequestFailed`).
    """
    return (await client.request("PUT", f"/api/chats/{session_id}", _Renamed, json={"title": title})).title


async def delete_session(session_id: str, mode: DeleteMode = "preserve-graph") -> None:
    """Delete a learn session and its jobs. An unknown session counts as deleted.

    *mode* `preserve-graph` keeps its Knowledge in the graph, `delete-nodes` removes its graph traces too (topics only
    this session created go, shared ones stay). LearnHarness refuses (409) while a turn or background job of the session
    runs. A 404 without LearnHarness's "No such session" (a moved route, another app on the URL) raises
    `UpstreamContractError`: nothing counts as deleted that was not.
    """
    path = f"/api/chats/{session_id}"
    try:
        await client.request("DELETE", path, _Deleted, params={"mode": mode}, timeout=DELETE_TIMEOUT_SECONDS)
    except UpstreamRequestFailed as exc:
        if exc.status != 404:
            raise
        if str(exc) != NO_SUCH_SESSION:
            raise UpstreamContractError(f'LearnHarness DELETE {path}: 404 without "{NO_SUCH_SESSION}" ({exc})') from exc


# `filename="<slug>.md"` of LearnHarness's export `Content-Disposition`.
_EXPORT_FILENAME = re.compile(r'filename="([^"/\\]+)"')


async def export_markdown(session_id: str) -> SessionExport:
    """The whole session as one Markdown document (`GET /api/chats/<id>/markdown`, the sidebar's "Export as Markdown").

    An unknown session raises `UpstreamRequestFailed` (404), a body that is no Markdown attachment `UpstreamContractError`.
    """
    path = f"/api/chats/{session_id}/markdown"
    try:
        async with httpx.AsyncClient(base_url=client.url, timeout=TIMEOUT_SECONDS, transport=client.transport) as http:
            response = await http.get(path)
    except httpx.HTTPError as exc:
        raise UpstreamUnavailable(f"LearnHarness at {client.url} does not answer: {exc}") from exc
    if response.is_error:
        raise UpstreamRequestFailed(response.status_code, error_message(response))
    filename = _EXPORT_FILENAME.search(response.headers.get("content-disposition", ""))
    if not response.headers.get("content-type", "").startswith("text/markdown") or filename is None:
        raise UpstreamContractError(f"LearnHarness {path}: no Markdown attachment")
    return SessionExport(filename=filename.group(1), markdown=response.text)


# ── Model configuration ──────────────────────────────────────────────────────────────────────────────────────────────
# LearnHarness reaches gigachad's models through its one custom OpenAI endpoint, the LiteLLM proxy (run-litellm-proxy.sh).
# `write-config` derives both sides from one read of the host user's model catalog, so every model id is a gigachad
# selector string (`gemini/gemini-3.5-flash`, `litellm_proxy/anthropic/claude-sonnet-5-5`, `ollama/<tag>`).
# run-litellm-proxy.sh is its only caller, right before it starts the proxy: the proxy's routes and LearnHarness's
# snapshot never come from different catalog reads. `check-proxy` tells a reused proxy that serves another list apart.
# - proxy `config.yaml`: an explicit model list (wildcards would list literal `*` rows), `drop_params` because LearnHarness
#   always sends `store: false`. It holds no secret: the master key and the OMP gateway token are `os.environ/`
#   references the proxy launcher exports, provider keys come from the proxy's environment like gigachad's own calls
# - LearnHarness `settings.json`: the `customProvider` snapshot with capabilities, because the proxy's `/models` publishes
#   none, and the strong/cheap/vision tiers, seeded when empty or no longer served. Every other key is LearnHarness's
# - LearnHarness `auth.json`: the binding that enables the endpoint. The key itself is the master key, passed as
#   `CUSTOM_OPENAI_API_KEY` by run-learnharness.sh, never stored there
# A catalog that cannot be read whole (no such login, an empty catalog, the OMP gateway, OMP's registry or Ollama not
# answering) writes nothing and exits non-zero: unknown capabilities are never stored as "no", the last good files stay.

DEFAULT_PROXY_BIND = "127.0.0.1:30183"
DEFAULT_PROXY_HOME = "~/.local/share/gigachad-bot/litellm-proxy"
DEFAULT_LEARNHARNESS_HOME = "~/.local/share/gigachad-bot/learnharness"
CUSTOM_PROVIDER = "custom"
CUSTOM_AUTH_BINDING = "__learnharness_custom_provider_binding"
SETTINGS_VERSION = 1
# The proxy launcher exports these, config.yaml only names them.
MASTER_KEY_ENV = "LITELLM_MASTER_KEY"
GATEWAY_TOKEN_ENV = "GIGACHAD_OMP_GATEWAY_TOKEN"
# LearnHarness tier ← the gigachad default that seeds it.
TIER_DEFAULTS = {"strong": "default_model", "cheap": "small_model", "vision": "vision_model"}
# Limits of a model no source describes. LearnHarness's own fallback (8192 context, 4096 output) would compact every
# lesson early, current API models have at least this much. A `customProvider.overrides` pin in LearnHarness wins.
ASSUMED_CONTEXT = 128_000
ASSUMED_MAX_TOKENS = 16_384
# run.sh starts the backend, which starts PostgreSQL and creates the schema, alongside the launchers.
DATABASE_WAIT_SECONDS = 90.0
_SCHEMA_RETRY_SECONDS = 2.0
_OLLAMA_TIMEOUT_SECONDS = 5.0
_PROXY_TIMEOUT_SECONDS = 10.0
# LearnHarness rewrites settings.json itself: a write of ours that would land on a newer file is redone this often.
_SETTINGS_ATTEMPTS = 3
_LOCK_NAME = "write-config.lock"
_MASTER_KEY = re.compile(r"sk-[0-9a-f]{64}")


class ConfigUnavailable(Exception):
    """gigachad's catalog cannot be read whole: nothing is written, the last configuration stays."""


def proxy_home() -> Path:
    """`config.yaml` (generated) and `master.key` (created once)."""
    return Path(os.environ.get("GIGACHAD_LITELLM_PROXY_HOME") or DEFAULT_PROXY_HOME).expanduser()


def proxy_base_url() -> str:
    """The proxy as LearnHarness's custom endpoint. LearnHarness appends `/chat/completions` itself."""
    return f"http://{os.environ.get('GIGACHAD_LITELLM_PROXY_BIND') or DEFAULT_PROXY_BIND}"


def learnharness_home() -> Path:
    """Holds LearnHarness's `data/` (`LH_DATA_DIR`) and `auth.json` (`LH_AUTH_FILE`)."""
    return Path(os.environ.get("GIGACHAD_LEARNHARNESS_HOME") or DEFAULT_LEARNHARNESS_HOME).expanduser()


@dataclass(frozen=True)
class ServedModel:
    """One gigachad model as the proxy routes it and LearnHarness describes it. `None` limits are unknown."""

    id: str
    litellm_params: dict[str, str]
    image: bool = False
    reasoning: bool = False
    context: int | None = None
    max_tokens: int | None = None

    @property
    def assumed_limits(self) -> bool:
        return self.context is None or self.max_tokens is None

    def snapshot(self) -> dict[str, Any]:
        """The `customProvider.models[]` row. Unknown limits are assumed, the output at most half the window."""
        context = self.context or ASSUMED_CONTEXT
        max_tokens = self.max_tokens or min(ASSUMED_MAX_TOKENS, context // 2)
        return {"id": self.id, "image": self.image, "reasoning": self.reasoning, "context": context, "maxTokens": max_tokens}


def catalog_selectors(providers: Mapping[str, Mapping[str, Any]], ollama: Iterable[str]) -> list[str]:
    """Every model gigachad's selector offers, in its order: `<litellm_id>/<model>` per provider row, then Ollama."""
    selectors = [f"{provider['litellm_id']}/{model}" for provider in providers.values() for model in provider["models"]]
    return list(dict.fromkeys([*selectors, *ollama]))


def _positive(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _litellm_info(model: str) -> dict[str, Any]:
    """LiteLLM's registry entry, empty for a model it does not know."""
    import litellm  # ~2s: the master-key, check-proxy and proxy-tool-pin commands never need it

    try:
        return dict(litellm.get_model_info(model=model))
    except Exception:
        return {}


def _supports_vision(model: str) -> bool:
    import litellm  # ~2s, see _litellm_info

    try:
        return bool(litellm.supports_vision(model=model))
    except Exception:
        return False


def _ollama_context(details: Mapping[str, Any]) -> int | None:
    """`/api/show` reports the window as `<architecture>.context_length` in `model_info`."""
    info = details.get("model_info") or {}
    return _positive(info.get(f"{info.get('general.architecture')}.context_length"))


def served_model(
    selector: str,
    *,
    reasoning: bool,
    omp_entries: Mapping[str, Mapping[str, Any]],
    ollama_details: Mapping[str, Mapping[str, Any]],
) -> ServedModel:
    """Route and capabilities of *selector*: OMP's gateway rows, Ollama's `/api/show`, else LiteLLM's registry."""
    omp_prefix = f"{omp_source.LITELLM_ID}/"
    if selector.startswith(omp_prefix):
        entry = omp_entries.get(selector.removeprefix(omp_prefix)) or {}
        params = {"model": selector, "api_base": omp_source.gateway_url(), "api_key": f"os.environ/{GATEWAY_TOKEN_ENV}"}
        return ServedModel(
            selector,
            params,
            image="image" in (entry.get("input_modalities") or []),
            reasoning=reasoning,
            context=_positive(entry.get("context_length")),
            max_tokens=_positive(entry.get("max_output_tokens")),
        )
    if selector.startswith("ollama/"):
        details = ollama_details.get(selector.removeprefix("ollama/")) or {}
        return ServedModel(
            selector,
            {"model": selector, "api_base": OLLAMA_BASE_URL},
            image="vision" in (details.get("capabilities") or []),
            reasoning=reasoning,
            context=_ollama_context(details),
        )
    info = _litellm_info(selector)
    return ServedModel(
        selector,
        {"model": selector},
        image=_supports_vision(selector),
        reasoning=reasoning,
        context=_positive(info.get("max_input_tokens")),
        max_tokens=_positive(info.get("max_output_tokens") or info.get("max_tokens")),
    )


def proxy_config(models: Iterable[ServedModel]) -> dict[str, Any]:
    """LiteLLM proxy `config.yaml`: one route per model, named by its selector, secrets by environment name."""
    return {
        "model_list": [{"model_name": model.id, "litellm_params": dict(model.litellm_params)} for model in models],
        "litellm_settings": {"drop_params": True},
        "general_settings": {"master_key": f"os.environ/{MASTER_KEY_ENV}"},
    }


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _tier_model(slot: object) -> str | None:
    if not isinstance(slot, dict) or not isinstance(slot.get("model"), dict):
        return None
    model_id = slot["model"].get("id")
    return model_id if isinstance(model_id, str) and model_id else None


def _seed(tier: str, models: list[ServedModel], defaults: Mapping[str, str]) -> str | None:
    """gigachad's default for *tier* when it is served, else the first model. Vision takes only image models, else none."""
    preferred = defaults.get(TIER_DEFAULTS[tier])
    candidates = [model.id for model in models if model.image or tier != "vision"]
    if preferred in candidates:
        return preferred
    return candidates[0] if candidates else None


def seeded_tiers(current: object, models: list[ServedModel], defaults: Mapping[str, str]) -> dict[str, Any]:
    """The `models` section with every chosen, served tier kept and every other one seeded.

    A tier is empty when it has no model, or one the proxy does not serve: only the custom endpoint has a key. A reseeded
    tier drops its `thinkingLevel`, the new model may not have it. Vision without an image model stays empty, LearnHarness
    then refuses only visual and PDF jobs until the learner picks one. LearnHarness replaces the whole section on write.
    """
    served = {model.id for model in models}
    section = dict(current) if isinstance(current, dict) else {}
    for tier in TIER_DEFAULTS:
        slot = section[tier] if isinstance(section.get(tier), dict) else {}
        chosen = _tier_model(slot)
        if chosen in served and slot["model"].get("provider") == CUSTOM_PROVIDER:
            continue
        seed = _seed(tier, models, defaults)
        if chosen is not None:
            print(f"note: LearnHarness's {tier} tier {chosen} is not served, reseeded with {seed}", file=sys.stderr)
        if seed is None:
            _warn(f"no image model in the catalog, LearnHarness's {tier} tier stays empty until one is picked there")
            section.pop(tier, None)
            continue
        kept = {key: value for key, value in slot.items() if key not in ("model", "thinkingLevel")}
        section[tier] = {**kept, "model": {"provider": CUSTOM_PROVIDER, "id": seed}}
    return section


def binding_id(settings: Mapping[str, Any]) -> str:
    """The stored custom endpoint binding id, a new one when there is none."""
    provider = settings.get("customProvider")
    existing = provider.get("credentialBindingId") if isinstance(provider, dict) else None
    if isinstance(existing, str) and len(existing) == 32 and all(char in "0123456789abcdef" for char in existing):
        return existing
    return secrets.token_hex(16)


def merged_settings(
    current: Mapping[str, Any], *, models: list[ServedModel], defaults: Mapping[str, str], base_url: str, binding: str
) -> dict[str, Any]:
    """*current* with gigachad's `customProvider` and seeded tiers. Pins (`overrides`) and every other key stay."""
    settings = {"version": SETTINGS_VERSION, **current}
    provider = current.get("customProvider")
    settings["customProvider"] = {
        **(provider if isinstance(provider, dict) else {}),
        "baseUrl": base_url,
        "authMode": "apiKey",
        "credentialBindingId": binding,
        "models": [model.snapshot() for model in models],
    }
    settings["models"] = seeded_tiers(current.get("models"), models, defaults)
    return settings


def binding_record(binding: str, base_url: str) -> str:
    """The `auth.json` value enabling the endpoint. Without `fileKeyBaseUrl` LearnHarness takes the key from its env."""
    return json.dumps({"version": 1, "id": binding, "baseUrl": base_url, "authMode": "apiKey"}, separators=(",", ":"))


def _read_json(path: Path) -> dict[str, Any]:
    """The JSON object at *path*, empty when absent. Anything else raises rather than being overwritten."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def _stamp(path: Path) -> tuple[int, int] | None:
    """Modification time and size of *path*, None when absent: a different stamp means someone else wrote it."""
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return status.st_mtime_ns, status.st_size


def _write_private(path: Path, content: str, *, unless: Callable[[], bool] | None = None) -> bool:
    """Replace *path* atomically with an owner-only file, False instead when *unless* holds right before the rename.

    The staging file never outlives the call.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(content)
        if unless is not None and unless():
            return False
        os.replace(staging, path)
        return True
    finally:
        staging.unlink(missing_ok=True)


@contextmanager
def _locked(directory: Path) -> Iterator[None]:
    """Hold *directory*'s write-config lock: launchers and restarts run the generator concurrently."""
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory / _LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _master_key(home: Path) -> str:
    path = home / "master.key"
    try:
        key = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        key = ""
    if _MASTER_KEY.fullmatch(key):
        return key
    key = f"sk-{secrets.token_hex(32)}"
    _write_private(path, f"{key}\n")
    return key


def master_key(home: Path) -> str:
    """The proxy's master key in `<home>/master.key` (owner-only), created once. An empty or cut key is replaced.

    LiteLLM wants the `sk-` prefix. The file appears whole (staged, then renamed) under the home's lock.
    """
    with _locked(home):
        return _master_key(home)


def write_config(
    models: list[ServedModel],
    defaults: Mapping[str, str],
    *,
    proxy_dir: Path,
    learnharness_dir: Path,
    base_url: str,
) -> None:
    """Write LearnHarness's settings and auth binding, then the proxy config, under both homes' locks. Idempotent.

    An empty catalog raises before anything is written. A settings file LearnHarness rewrote meanwhile is merged again.
    """
    if not models:
        raise ConfigUnavailable("the catalog has no model")
    settings_path = learnharness_dir / "data" / "settings.json"
    auth_path = learnharness_dir / "auth.json"
    with _locked(proxy_dir), _locked(learnharness_dir):
        _master_key(proxy_dir)
        for _ in range(_SETTINGS_ATTEMPTS):
            stamp = _stamp(settings_path)
            current = _read_json(settings_path)
            binding = binding_id(current)
            settings = merged_settings(current, models=models, defaults=defaults, base_url=base_url, binding=binding)
            content = json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
            if _write_private(settings_path, content, unless=lambda: _stamp(settings_path) != stamp):  # noqa: B023 - called at once
                break
        else:
            raise ConfigUnavailable(f"{settings_path} kept changing while it was merged")

        auth = _read_json(auth_path)
        auth[CUSTOM_AUTH_BINDING] = binding_record(binding, base_url)
        _write_private(auth_path, json.dumps(auth, indent=2) + "\n")
        config = yaml.safe_dump(proxy_config(models), sort_keys=False, allow_unicode=True)
        _write_private(proxy_dir / "config.yaml", config)


def proxy_drift(base_url: str, key: str, config_path: Path) -> str | None:
    """Why the proxy at *base_url* is not the one *config_path* and *key* describe, None when it is."""
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return f"there is no {config_path}"
    expected = [route["model_name"] for route in config.get("model_list") or []]
    try:
        response = httpx.get(f"{base_url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=_PROXY_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        return f"it does not answer ({exc})"
    if response.status_code != httpx.codes.OK:
        return (
            f"it refuses the master key in {config_path.parent} (HTTP {response.status_code}: {error_message(response)}), "
            "it started from another GIGACHAD_LITELLM_PROXY_HOME or before master.key changed"
        )
    rows = response.json().get("data") or []
    served = {row.get("id") for row in rows if isinstance(row, dict)}
    missing = [model for model in expected if model not in served]
    extra = sorted(str(model) for model in served - set(expected))
    if missing or extra:
        return (
            f"it serves another model list than {config_path} (missing: {', '.join(missing) or 'none'}, "
            f"extra: {', '.join(extra) or 'none'}), it started before the last write-config"
        )
    return None


def proxy_tool_pin(lock_path: Path) -> str:
    """`<requirement> <python> <exclude-newer>` for `uv tool run`: the venv's LiteLLM, its Python, and the index as it was
    one day after that LiteLLM's upload, so the proxy's other dependencies resolve the same way on every machine."""
    version = importlib.metadata.version("litellm")
    with lock_path.open("rb") as file:
        packages = tomllib.load(file).get("package") or []
    package = next((entry for entry in packages if entry.get("name") == "litellm" and entry.get("version") == version), None)
    if package is None:
        raise ConfigUnavailable(f"{lock_path} does not lock litellm {version}, sync the venv")
    files = [package.get("sdist") or {}, *(package.get("wheels") or [])]
    uploaded = max(datetime.fromisoformat(entry["upload-time"]) for entry in files if "upload-time" in entry)
    exclude_newer = (uploaded + timedelta(days=1)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"litellm[proxy]=={version} {sys.version_info.major}.{sys.version_info.minor} {exclude_newer}"


# ── Catalog sources (live) ───────────────────────────────────────────────────────────────────────────────────────────


def _use_development_database() -> None:
    """Point at run-backend.sh's development database when the environment names none (production sets its own)."""
    if not os.environ.get("GIGACHAD_DATABASE_URL"):
        os.environ["GIGACHAD_DATABASE_URL"] = f"postgresql://{getpass.getuser()}@127.0.0.1:5432/gigachad"
        os.environ.setdefault("PGPASSWORD", os.environ.get("POSTGRES_PASSWORD_DEV", ""))


def _user_id(pool: ConnectionPool, login: str) -> UUID:
    """The user owning *login*. Read only: a mistyped login must not register a user."""
    with pool.connection() as connection:
        row = connection.execute("SELECT id FROM users WHERE tailscale_login = %s", (login,)).fetchone()
    if row is None:
        raise ConfigUnavailable(f"no gigachad user {login!r} (--login, GIGACHAD_LEARNHARNESS_LOGIN, else the OS user)")
    return row[0]


def _host_catalog(login: str) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """The provider rows gigachad's selector shows *login*, and its model defaults. Waits for a starting database."""
    _use_development_database()
    pool = get_postgres_pool()
    deadline = time.monotonic() + DATABASE_WAIT_SECONDS
    try:
        try:
            pool.wait(timeout=DATABASE_WAIT_SECONDS)
        except PoolTimeout as exc:
            raise ConfigUnavailable(f"PostgreSQL accepted no connection within {DATABASE_WAIT_SECONDS:.0f} s ({exc})") from exc
        while True:
            try:
                store = ModelProviderStore(PostgresDataStore(UserScope(pool, _user_id(pool, login))))
                return providers_for_ui(store.load()), store.load_defaults()
            except UndefinedTable as exc:
                if time.monotonic() >= deadline:
                    raise ConfigUnavailable(f"gigachad's schema is missing ({exc}), start the backend once") from exc
                time.sleep(_SCHEMA_RETRY_SECONDS)
    finally:
        close_postgres_pool()


def _omp_entries() -> dict[str, dict[str, Any]]:
    """The OMP gateway's `/models` rows by `<provider>/<model>`."""
    try:
        entries = omp_source.catalog_entries()  # raw rows: group_by_provider drops max_output_tokens
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ConfigUnavailable(f"no OMP gateway at {omp_source.gateway_url()} ({exc}), OMP models are undescribed") from exc
    return {str(entry.get("id") or "").lstrip("/"): entry for entry in entries}


def _omp_thinking() -> dict[str, list[str]]:
    """OMP's effort ladders by `<provider>/<model>`: one `omp models --json` for every model, and its failure raised."""
    try:
        levels = omp_source.thinking_registry()
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError) as exc:
        raise ConfigUnavailable(f"OMP's model registry (`omp models --json`) did not answer ({exc})") from exc
    if not levels:
        raise ConfigUnavailable("OMP's model registry (`omp models --json`) lists no model")
    return levels


async def _ollama_show(client: httpx.AsyncClient, tag: str) -> dict[str, Any]:
    try:
        response = await client.post("/api/show", json={"model": tag})
        response.raise_for_status()
        details = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ConfigUnavailable(f"Ollama at {OLLAMA_BASE_URL} does not describe {tag} ({exc})") from exc
    if not isinstance(details, dict):
        raise ConfigUnavailable(f"Ollama at {OLLAMA_BASE_URL} describes {tag} as {details!r}")
    return details


async def _reasons(selector: str, omp_levels: Mapping[str, list[str]]) -> bool:
    """gigachad's own answer (OMP registry, Ollama, LiteLLM). A failed lookup raises, it never counts as "no"."""
    from lib import reasoning_levels  # imports litellm, see _litellm_info

    omp_prefix = f"{omp_source.LITELLM_ID}/"
    if selector.startswith(omp_prefix):
        levels = omp_levels.get(selector.removeprefix(omp_prefix))
        # A model the registry lacks falls back to LiteLLM's flags, as reasoning_levels.resolve does.
        return bool(levels if levels is not None else reasoning_levels.litellm_levels(selector))
    levels = await reasoning_levels.resolve(selector)
    if levels is None:
        raise ConfigUnavailable(f"no reasoning answer for {selector}")
    return bool(levels)


async def served_catalog(login: str) -> tuple[list[ServedModel], dict[str, str]]:
    """Every model gigachad's selector offers *login*, routed and described, plus its defaults."""
    (providers, defaults), ollama = await asyncio.gather(
        asyncio.to_thread(_host_catalog, login), asyncio.to_thread(discover_ollama_models)
    )
    selectors = catalog_selectors(providers, ollama)
    if not selectors:
        raise ConfigUnavailable(f"{login}'s catalog has no model")
    omp_prefix = f"{omp_source.LITELLM_ID}/"
    omp_models = [selector.removeprefix(omp_prefix) for selector in selectors if selector.startswith(omp_prefix)]
    omp_entries: dict[str, dict[str, Any]] = {}
    omp_levels: dict[str, list[str]] = {}
    if omp_models:
        omp_entries, omp_levels = await asyncio.gather(asyncio.to_thread(_omp_entries), asyncio.to_thread(_omp_thinking))
        if unlisted := [model for model in omp_models if model not in omp_entries]:
            _warn(f"the OMP gateway does not list {', '.join(unlisted)}: served as text-only")
    tags = [selector.removeprefix("ollama/") for selector in selectors if selector.startswith("ollama/")]
    async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=_OLLAMA_TIMEOUT_SECONDS) as ollama_client:
        shown = await asyncio.gather(*(_ollama_show(ollama_client, tag) for tag in tags))
    reasoning = await asyncio.gather(*(_reasons(selector, omp_levels) for selector in selectors))
    models = [
        served_model(selector, reasoning=reasons, omp_entries=omp_entries, ollama_details=dict(zip(tags, shown, strict=True)))
        for selector, reasons in zip(selectors, reasoning, strict=True)
    ]
    if assumed := [model.id for model in models if model.assumed_limits]:
        _warn(
            f"no context or max output figure for {', '.join(assumed)}: assuming {ASSUMED_CONTEXT} and "
            f"{ASSUMED_MAX_TOKENS} tokens (at most half the window), pin others in LearnHarness's custom provider"
        )
    return models, defaults


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lib.learnharness", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser(
        "write-config", help="write the LiteLLM proxy config and LearnHarness's models from gigachad's catalog"
    )
    write.add_argument(
        "--login",
        default=os.environ.get("GIGACHAD_LEARNHARNESS_LOGIN") or getpass.getuser(),
        help="gigachad login whose model catalog LearnHarness serves (default: GIGACHAD_LEARNHARNESS_LOGIN, else the OS user)",
    )
    commands.add_parser("master-key", help="print the proxy's master key, created on first use")
    commands.add_parser("check-proxy", help="exit 1 when the running proxy is not the one this proxy home describes")
    pin = commands.add_parser("proxy-tool-pin", help="print the `uv tool run` requirement, Python and exclude-newer")
    pin.add_argument("lock", type=Path, help="the project's uv.lock")
    args = parser.parse_args(argv)

    try:
        if args.command == "master-key":
            print(master_key(proxy_home()))
            return 0
        if args.command == "proxy-tool-pin":
            print(proxy_tool_pin(args.lock))
            return 0
        if args.command == "check-proxy":
            drift = proxy_drift(proxy_base_url(), master_key(proxy_home()), proxy_home() / "config.yaml")
            if drift is not None:
                _warn(f"the LiteLLM proxy at {proxy_base_url()} is stale or mismatched: {drift}. Stop it, rerun run-litellm-proxy.sh")
                return 1
            return 0
        models, defaults = asyncio.run(served_catalog(args.login))
        write_config(models, defaults, proxy_dir=proxy_home(), learnharness_dir=learnharness_home(), base_url=proxy_base_url())
    except ConfigUnavailable as exc:
        print(f"error: {exc}. Nothing written, the last configuration stays", file=sys.stderr)
        return 1
    images = sum(model.image for model in models)
    reasoning = sum(model.reasoning for model in models)
    assumed = sum(model.assumed_limits for model in models)
    summary = f"{len(models)} ({images} image, {reasoning} reasoning, {assumed} with assumed limits)"
    print(f"LearnHarness models from {args.login}'s catalog: {summary} via {proxy_base_url()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
