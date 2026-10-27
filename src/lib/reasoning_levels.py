"""Which `reasoning_effort` levels a model accepts: live discovery, the stored answer, and the send-time check."""

import asyncio
from typing import Any

import httpx
import litellm

from config import OLLAMA_BASE_URL
from lib import omp_source
from lib.postgres_pool import get_postgres_pool

# The app's "off": a chat request at this level omits `reasoning_effort` entirely.
NONE = "none"
OLLAMA_LEVELS = [NONE, "low", "medium", "high"]


async def _ollama_supports_reasoning(model: str) -> bool | None:
    """Ask the Ollama server itself, None when it can't answer.

    - LiteLLM's static registry doesn't know pulled or Ollama Cloud tags (`gemma4:31b-cloud`)
    - `/api/show` reports real capabilities, `thinking` among them, for local and cloud models
    """
    try:
        async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=5.0) as client:
            resp = await client.post("/api/show", json={"model": model})
            resp.raise_for_status()
            return "thinking" in (resp.json().get("capabilities") or [])
    except (httpx.HTTPError, ValueError):
        return None


def litellm_levels(model: str) -> list[str]:
    """Effort levels from LiteLLM's `supports_<level>_reasoning_effort` flags.

    - Mirror LiteLLM's send-time checks: `xhigh`, `max` opt-in, `low` opt-out
    - `minimal` opt-in too, few models accept it natively
    - `medium` and `high` come with any reasoning model
    - `supports_none_reasoning_effort` is ignored, `none` is the app's own "off"
    """
    if not litellm.supports_reasoning(model=model):
        return []
    try:
        info = litellm.get_model_info(model=model)
    except Exception:
        info = {}
    flags = {level: info.get(f"supports_{level}_reasoning_effort") for level in ("minimal", "low", "xhigh", "max")}
    levels = [NONE]
    if flags["minimal"]:
        levels.append("minimal")
    if flags["low"] is not False:
        levels.append("low")
    levels += ["medium", "high"]
    levels += [level for level in ("xhigh", "max") if flags[level]]
    return levels


async def resolve(model: str) -> list[str] | None:
    """Live lookup of the levels `model` accepts, None when the lookup itself failed.

    - Ollama tags (`ollama/<tag>`, incl. cloud) aren't in LiteLLM's registry, ask the live server
    - OMP gateway models (`litellm_proxy/<provider>/<model>`) neither, use OMP's own registry
    """
    if model.startswith("ollama/"):
        supported = await _ollama_supports_reasoning(model.removeprefix("ollama/"))
        if supported is None:
            return None
        return OLLAMA_LEVELS if supported else []
    omp_prefix = f"{omp_source.LITELLM_ID}/"
    if model.startswith(omp_prefix):
        omp_levels = await asyncio.to_thread(omp_source.thinking_levels, model.removeprefix(omp_prefix))
        if omp_levels is not None:
            return [NONE, *omp_levels] if omp_levels else []
    return litellm_levels(model)


def load_stored(model: str) -> list[str] | None:
    """The last resolved levels of `model`, None when it was never resolved."""
    with get_postgres_pool().connection() as connection:
        row = connection.execute("SELECT levels FROM model_reasoning_levels WHERE model = %s", (model,)).fetchone()
    return row[0] if row else None


def store(model: str, levels: list[str]) -> None:
    """Persist the resolved levels of `model`, touching the row only when they changed."""
    with get_postgres_pool().connection() as connection:
        connection.execute(
            """
            INSERT INTO model_reasoning_levels (model, levels) VALUES (%s, %s)
            ON CONFLICT (model) DO UPDATE SET levels = EXCLUDED.levels, updated_at = now()
            WHERE model_reasoning_levels.levels IS DISTINCT FROM EXCLUDED.levels
            """,
            (model, levels),
        )


async def refresh(model: str) -> list[str] | None:
    """Resolve `model` live and persist the answer, a failed lookup keeps the stored row."""
    levels = await resolve(model)
    if levels is not None:
        await asyncio.to_thread(store, model, levels)
    return levels


def _supports_effort(model: str) -> bool:
    """Whether litellm lists `reasoning_effort` as supported for *model*.

    Some providers (e.g. nvidia_nim) reject the param outright, so consult litellm's
    table instead of an allowlist. Fail closed on lookup errors: dropping the param
    is harmless, forwarding it blind can crash the request.
    """
    try:
        return "reasoning_effort" in (litellm.get_supported_openai_params(model=model) or [])
    except Exception:
        return False


def effort_kwargs(effort: str | None, model: str) -> dict[str, Any]:
    """The completion kwargs carrying *effort*, empty when it is off or *model* rejects the param."""
    if effort and effort != NONE and _supports_effort(model):
        return {"reasoning_effort": effort}
    return {}
