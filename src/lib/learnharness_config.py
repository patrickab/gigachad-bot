"""Generator of LearnHarness's model configuration: its custom provider is pi-host's completion gateway.

- Consumes the host user's model catalog (Postgres), the installed Ollama models (`lib.ollama_models`) and pi-host's
  gateway model list (`lib.pi_models`)
- Provides `python -m lib.learnharness_config write-config` for run-learnharness.sh
- Standalone from the sessions adapter `lib.learnharness`: it may import from it, never the reverse
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import getpass
import json
import os
from pathlib import Path
import secrets
import sys
import time
from typing import TYPE_CHECKING, Any

from psycopg.errors import UndefinedTable
from psycopg_pool import PoolTimeout

from lib import ollama_models, pi_host, pi_models
from lib.changes import UserScope
from lib.model_provider_store import ModelProviderStore, providers_for_ui
from lib.postgres_data_store import PostgresDataStore
from lib.postgres_pool import close_postgres_pool, get_postgres_pool

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping
    from uuid import UUID

    from psycopg_pool import ConnectionPool


# ── Model configuration ──────────────────────────────────────────────────────────────────────────────────────────────
# LearnHarness reaches gigachad's models through its one custom OpenAI endpoint, pi-host's completion gateway.
# `write-config` (run by run-learnharness.sh right before LearnHarness starts) derives the snapshot from one read of the
# host user's catalog and one of the gateway's `/v1/models`. A model's LearnHarness id is its gateway id
# (`pi_models.gateway_model_id`: `google/gemini-3.5-flash`, `anthropic/claude-sonnet-5-5`, `ollama/<tag>`), the id
# LearnHarness sends back to the gateway. A catalog model the gateway does not serve is left out with a warning.
# - LearnHarness `settings.json`: the `customProvider` snapshot with the gateway's capabilities, because LearnHarness
#   reads none from `/models`, and the strong/cheap/vision tiers, seeded when empty or no longer served. A tier still
#   naming a stored gigachad selector (`litellm_proxy/x`, from the LiteLLM proxy) keeps its pick under the gateway id.
#   Every other key is LearnHarness's
# - LearnHarness `auth.json`: the binding that enables the endpoint. The key itself is pi-host's gateway-only token,
#   passed as `CUSTOM_OPENAI_API_KEY` by run-learnharness.sh, never stored there
# A catalog that cannot be read whole (no such login, an empty catalog, pi-host's gateway not answering) writes
# nothing and exits non-zero: unknown capabilities are never stored as "no", the last good files stay.

DEFAULT_LEARNHARNESS_HOME = "~/.local/share/gigachad-bot/learnharness"
CUSTOM_PROVIDER = "custom"
CUSTOM_AUTH_BINDING = "__learnharness_custom_provider_binding"
SETTINGS_VERSION = 1
# LearnHarness tier ← the gigachad default that seeds it.
TIER_DEFAULTS = {"strong": "default_model", "cheap": "small_model", "vision": "vision_model"}
# Limits of a model the gateway does not describe. LearnHarness's own fallback (8192 context, 4096 output) would compact
# every lesson early, current API models have at least this much. A `customProvider.overrides` pin in LearnHarness wins.
ASSUMED_CONTEXT = 128_000
ASSUMED_MAX_TOKENS = 16_384
# run.sh starts the backend, which starts PostgreSQL and creates the schema, alongside the launchers.
DATABASE_WAIT_SECONDS = 90.0
_SCHEMA_RETRY_SECONDS = 2.0
# LearnHarness rewrites settings.json itself: a write of ours that would land on a newer file is redone this often.
_SETTINGS_ATTEMPTS = 3
_LOCK_NAME = "write-config.lock"


class ConfigUnavailable(Exception):
    """gigachad's catalog cannot be read whole: nothing is written, the last configuration stays."""


def learnharness_home() -> Path:
    """Holds LearnHarness's `data/` (`LH_DATA_DIR`) and `auth.json` (`LH_AUTH_FILE`)."""
    return Path(os.environ.get("GIGACHAD_LEARNHARNESS_HOME") or DEFAULT_LEARNHARNESS_HOME).expanduser()


@dataclass(frozen=True)
class ServedModel:
    """One gateway model as LearnHarness describes it, by its gateway id. `None` limits are unknown."""

    id: str
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


def catalog_model_ids(providers: Mapping[str, Mapping[str, Any]], ollama_tags: Iterable[str]) -> list[str]:
    """Every model gigachad's selector offers, in its order, as gateway ids: each provider row's models, then Ollama's."""
    selectors = [f"{provider['provider_id']}/{model}" for provider in providers.values() for model in provider["models"]]
    selectors += [f"ollama/{tag}" for tag in ollama_tags]
    return list(dict.fromkeys(pi_models.gateway_model_id(selector) for selector in selectors))


def _positive(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def served_model(entry: dict[str, Any]) -> ServedModel:
    """A gateway `/v1/models` row as LearnHarness describes it: vision from `input`, reasoning from `thinking_levels`."""
    return ServedModel(
        str(entry["id"]),
        image="image" in (entry.get("input") or []),
        reasoning=any(level != "off" for level in pi_models.entry_thinking_levels(entry)),
        context=_positive(entry.get("context_window")),
        max_tokens=_positive(entry.get("max_tokens")),
    )


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
    if preferred and pi_models.gateway_model_id(preferred) in candidates:
        return pi_models.gateway_model_id(preferred)
    return candidates[0] if candidates else None


def seeded_tiers(current: object, models: list[ServedModel], defaults: Mapping[str, str]) -> dict[str, Any]:
    """The `models` section with every chosen, served tier kept and every other one seeded.

    A tier is empty when it has no model, or one the gateway does not serve: only the custom endpoint has a key. A tier
    naming a stored selector keeps its pick and level under the gateway id. A reseeded tier drops its `thinkingLevel`,
    the new model may not have it. Vision without an image model stays empty, LearnHarness then refuses only visual and
    PDF jobs until the learner picks one. LearnHarness replaces the whole section on write.
    """
    served = {model.id for model in models}
    section = dict(current) if isinstance(current, dict) else {}
    for tier in TIER_DEFAULTS:
        slot = section[tier] if isinstance(section.get(tier), dict) else {}
        chosen = _tier_model(slot)
        if chosen is not None and slot["model"].get("provider") == CUSTOM_PROVIDER:
            gateway_id = pi_models.gateway_model_id(chosen)
            if gateway_id in served:
                section[tier] = {**slot, "model": {**slot["model"], "id": gateway_id}}
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


def write_config(models: list[ServedModel], defaults: Mapping[str, str], *, learnharness_dir: Path, base_url: str) -> None:
    """Write LearnHarness's settings and auth binding under the home's lock. Idempotent.

    An empty catalog raises before anything is written. A settings file LearnHarness rewrote meanwhile is merged again.
    """
    if not models:
        raise ConfigUnavailable("the catalog has no model")
    settings_path = learnharness_dir / "data" / "settings.json"
    auth_path = learnharness_dir / "auth.json"
    with _locked(learnharness_dir):
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


def _gateway_entries() -> dict[str, dict[str, Any]]:
    """The Pi gateway's `/v1/models` rows by gateway id, capabilities and effort ladder included."""
    try:
        entries = pi_models.catalog_entries()  # raw rows: group_by_provider drops max_tokens
    except pi_models.GatewayUnavailable as exc:
        raise ConfigUnavailable(f"{exc}, LearnHarness's models are undescribed") from exc
    if not entries:
        raise ConfigUnavailable(f"the Pi gateway at {pi_host.gateway_url()} lists no model")
    return {str(entry.get("id") or "").lstrip("/"): entry for entry in entries}


def _ollama_and_gateway() -> tuple[list[str], dict[str, dict[str, Any]]]:
    """The installed Ollama tags and the gateway's rows."""
    return ollama_models.installed_chat_models(), _gateway_entries()


async def served_catalog(login: str) -> tuple[list[ServedModel], dict[str, str]]:
    """Every model gigachad's selector offers *login* that the gateway serves, described, plus its defaults."""
    (providers, defaults), (ollama_tags, entries) = await asyncio.gather(
        asyncio.to_thread(_host_catalog, login), asyncio.to_thread(_ollama_and_gateway)
    )
    model_ids = catalog_model_ids(providers, ollama_tags)
    if not model_ids:
        raise ConfigUnavailable(f"{login}'s catalog has no model")
    if unserved := [model_id for model_id in model_ids if model_id not in entries]:
        _warn(f"the Pi gateway does not serve {', '.join(unserved)}: left out of LearnHarness")
    models = [served_model(entries[model_id]) for model_id in model_ids if model_id in entries]
    if not models:
        raise ConfigUnavailable(f"the Pi gateway at {pi_host.gateway_url()} serves none of {login}'s models")
    if assumed := [model.id for model in models if model.assumed_limits]:
        _warn(
            f"no context or max output figure for {', '.join(assumed)}: assuming {ASSUMED_CONTEXT} and "
            f"{ASSUMED_MAX_TOKENS} tokens (at most half the window), pin others in LearnHarness's custom provider"
        )
    return models, defaults


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lib.learnharness_config", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write-config", help="write LearnHarness's models from gigachad's catalog and the Pi gateway")
    write.add_argument(
        "--login",
        default=os.environ.get("GIGACHAD_LEARNHARNESS_LOGIN") or getpass.getuser(),
        help="gigachad login whose model catalog LearnHarness serves (default: GIGACHAD_LEARNHARNESS_LOGIN, else the OS user)",
    )
    args = parser.parse_args(argv)

    base_url = pi_host.gateway_url()
    try:
        models, defaults = asyncio.run(served_catalog(args.login))
        write_config(models, defaults, learnharness_dir=learnharness_home(), base_url=base_url)
    except ConfigUnavailable as exc:
        print(f"error: {exc}. Nothing written, the last configuration stays", file=sys.stderr)
        return 1
    images = sum(model.image for model in models)
    reasoning = sum(model.reasoning for model in models)
    assumed = sum(model.assumed_limits for model in models)
    summary = f"{len(models)} ({images} image, {reasoning} reasoning, {assumed} with assumed limits)"
    print(f"LearnHarness models from {args.login}'s catalog: {summary} via {base_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
