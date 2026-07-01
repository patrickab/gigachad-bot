"""Web-search mode backed by Vane (Perplexica).

Proxies the local Vane sidecar's /api/chat endpoint. Vane identifies models by an
opaque providerId UUID assigned per provider, so we resolve gigachad's model
strings (e.g. "ollama/bge-m3:latest") against GET /api/providers at runtime and
cache the result. Nothing about Vane's UUIDs is hardcoded.
"""

import asyncio
from collections.abc import AsyncIterator
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from config import EMBEDDING_MODEL, VANE_URL

router = APIRouter(prefix="/api", tags=["search"])

_COMPOSE_FILE = str(Path(__file__).resolve().parent.parent.parent.parent / "docker-compose.vane.yml")
_vane_started = False
_vane_lock = threading.Lock()

# Cached {providerId, key} resolutions, keyed by gigachad model string.
_provider_cache: dict[str, dict[str, str]] = {}


def _clear_provider_cache() -> None:
    """Clear the provider resolution cache (e.g., after adding new providers)."""
    _provider_cache.clear()


# --- Docker lifecycle ------------------------------------------------------

def _docker_context() -> str | None:
    for ctx in ("default", "rootless"):
        try:
            r = subprocess.run(["docker", "--context", ctx, "info"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return ctx
        except Exception:
            continue
    return None


def _start_docker_daemon() -> str:
    ctx = _docker_context()
    if ctx:
        return ctx
    print("Starting Docker daemon...", file=sys.stderr)
    import os

    daemon = subprocess.Popen(
        ["dockerd-rootless.sh"],
        env={**os.environ, "HOME": str(Path.home())},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    for _ in range(30):
        time.sleep(1)
        if daemon.poll() is not None:
            stderr_output = daemon.stderr.read() if daemon.stderr else ""
            raise RuntimeError(f"Docker daemon exited early (rc={daemon.returncode}): {stderr_output[-500:]}")
        ctx = _docker_context()
        if ctx:
            print("Docker daemon is ready.", file=sys.stderr)
            return ctx
    daemon.kill()
    daemon.wait()
    raise RuntimeError("Docker daemon did not start within 30 seconds")


def _is_vane_ready(timeout: float = 30.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            r = subprocess.run(["curl", "-sf", VANE_URL], capture_output=True, timeout=3)
            if r.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        time.sleep(1)
    return False


def ensure_vane() -> None:
    global _vane_started
    if _vane_started:
        return
    with _vane_lock:
        if _vane_started:
            return
        ctx = _start_docker_daemon()
        print("Starting Vane container stack (first request)...", file=sys.stderr)
        subprocess.run(
            ["docker", "--context", ctx, "compose", "-f", _COMPOSE_FILE, "up", "-d", "--wait"],
            check=True,
        )
        if not _is_vane_ready():
            raise RuntimeError("Vane did not become ready within 30 seconds")
        print("Vane is ready.", file=sys.stderr)
        try:
            _ensure_providers()
        except Exception as e:
            print(f"Warning: Failed to initialize Vane providers: {e}", file=sys.stderr)
        _vane_started = True


def stop_vane() -> None:
    global _vane_started
    if not _vane_started:
        return
    print("Stopping Vane container stack...", file=sys.stderr)
    ctx = _docker_context()
    if ctx:
        subprocess.run(["docker", "--context", ctx, "compose", "-f", _COMPOSE_FILE, "down"], check=False)
    _vane_started = False
    print("Vane stopped.", file=sys.stderr)


# --- Provider resolution ---------------------------------------------------

# gigachad provider prefix -> Vane provider name (set by _ensure_providers).
# Without this scoping, the fuzzy pass cross-matches e.g. deepseek/deepseek-v4-pro
# onto Ollama's deepseek-v4-pro:cloud (subscription-gated) → wrong backend.
_PROVIDER_NAME_BY_PREFIX = {
    "ollama": "Ollama",
    "gemini": "Google",
    "deepseek": "DeepSeek",
    "openrouter": "OpenRouter",
}


def _model_key(model: str) -> str:
    """gigachad model string -> Vane model key. Strips the provider prefix."""
    return model.split("/", 1)[1] if "/" in model else model


def _resolve_from_providers(providers: list[dict], model: str, embedding: bool) -> dict[str, str] | None:
    """Find {providerId, key} for a model in a /api/providers payload.

    The gigachad model string is "<prefix>/<key>" (e.g. "deepseek/deepseek-v4-pro").
    The prefix names the intended backend, so we scope matching to the Vane provider
    it maps to — otherwise the fuzzy pass below would route deepseek/* onto Ollama's
    deepseek-*:cloud copies. Within that provider: exact match first, then substring.
    Returns None if that provider doesn't expose the model (caller surfaces a hint).
    """
    key = _model_key(model)
    field = "embeddingModels" if embedding else "chatModels"

    prefix = model.split("/", 1)[0] if "/" in model else ""
    expected = _PROVIDER_NAME_BY_PREFIX.get(prefix)
    # Known prefix → only that provider. Unknown prefix → fall back to all providers.
    scoped = [p for p in providers if p.get("name") == expected] if expected else providers

    # First pass: exact match on key or name
    for p in scoped:
        for m in p.get(field, []):
            if m.get("key") == key or m.get("name") == key:
                return {"providerId": p["id"], "key": m["key"]}

    # Second pass: substring/fuzzy match (case-insensitive)
    key_lower = key.lower()
    for p in scoped:
        for m in p.get(field, []):
            m_key_lower = (m.get("key") or "").lower()
            m_name_lower = (m.get("name") or "").lower()
            if key_lower in m_key_lower or key_lower in m_name_lower:
                return {"providerId": p["id"], "key": m["key"]}

    return None


def _fetch_providers() -> list[dict]:
    import httpx

    r = httpx.get(f"{VANE_URL.rstrip('/')}/api/providers", timeout=10.0)
    r.raise_for_status()
    return r.json().get("providers", [])


def _ensure_providers() -> None:
    """Initialize Vane with default Ollama + Gemini providers if missing."""
    import os

    import httpx

    providers = _fetch_providers()
    provider_names = {p.get("name") for p in providers}
    added = False

    # Ollama provider
    if "Ollama" not in provider_names:
        try:
            httpx.post(
                f"{VANE_URL.rstrip('/')}/api/providers",
                json={
                    "name": "Ollama",
                    "type": "ollama",
                    "apiBase": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                },
                timeout=10.0,
            ).raise_for_status()
            added = True
        except Exception as e:
            print(f"Warning: Failed to add Ollama provider to Vane: {e}", file=sys.stderr)

    # Gemini provider (if API key exists)
    if "Google" not in provider_names and os.environ.get("GEMINI_API_KEY"):
        try:
            httpx.post(
                f"{VANE_URL.rstrip('/')}/api/providers",
                json={
                    "name": "Google",
                    "type": "google",
                    "apiKey": os.environ.get("GEMINI_API_KEY", ""),
                },
                timeout=10.0,
            ).raise_for_status()
            added = True
        except Exception as e:
            print(f"Warning: Failed to add Google provider to Vane: {e}", file=sys.stderr)

    # OpenRouter provider (if API key exists)
    if "OpenRouter" not in provider_names and os.environ.get("OPENROUTER_API_KEY"):
        try:
            httpx.post(
                f"{VANE_URL.rstrip('/')}/api/providers",
                json={
                    "name": "OpenRouter",
                    "type": "openrouter",
                    "apiBase": "https://openrouter.ai/api/v1",
                    "apiKey": os.environ.get("OPENROUTER_API_KEY", ""),
                },
                timeout=10.0,
            ).raise_for_status()
            added = True
        except Exception as e:
            print(f"Warning: Failed to add OpenRouter provider to Vane: {e}", file=sys.stderr)

    if added:
        _clear_provider_cache()


def _resolve(model: str, embedding: bool) -> dict[str, str]:
    cache_key = f"{'embed:' if embedding else 'chat:'}{model}"
    if cache_key in _provider_cache:
        return _provider_cache[cache_key]
    resolved = _resolve_from_providers(_fetch_providers(), model, embedding)
    if resolved is None:
        kind = "embedding" if embedding else "chat"
        raise RuntimeError(
            f"Vane has no {kind} model matching '{_model_key(model)}'. "
            f"Add it under the provider's models in Vane Settings."
        )
    _provider_cache[cache_key] = resolved
    return resolved


# --- Routes ----------------------------------------------------------------

class WebSearchRequest(BaseModel):
    query: str
    focus_mode: str = "webSearch"
    optimization_mode: str = "balanced"
    system_instructions: str = ""
    model: str = ""


def _error_stream(msg: str) -> StreamingResponse:
    async def gen() -> AsyncIterator[str]:
        yield f"data: {json.dumps({'type': 'error', 'data': msg})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/web-search")
async def web_search(req: WebSearchRequest) -> StreamingResponse:
    import httpx

    if not req.model:
        return _error_stream("No chat model selected for web search.")
    try:
        await asyncio.to_thread(ensure_vane)
        chat_model = await asyncio.to_thread(_resolve, req.model, False)
        embed_model = await asyncio.to_thread(_resolve, EMBEDDING_MODEL, True)
    except Exception as e:  # noqa: BLE001 - surface any startup/resolution failure to the UI
        return _error_stream(str(e))

    payload = {
        "message": {"messageId": str(uuid.uuid4()), "chatId": str(uuid.uuid4()), "content": req.query},
        "focusMode": req.focus_mode,
        "optimizationMode": req.optimization_mode,
        "chatModel": chat_model,
        "embeddingModel": embed_model,
        "history": [],
        "files": [],
        "systemInstructions": req.system_instructions,
    }

    async def event_stream() -> AsyncIterator[str]:
        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client,
            client.stream(
                "POST",
                f"{VANE_URL.rstrip('/')}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response,
        ):
            if response.status_code != 200:
                body = await response.aread()
                yield f"data: {json.dumps({'type': 'error', 'data': body.decode(errors='replace')})}\n\n"
                return
            async for line in response.aiter_lines():
                s = line.strip()
                if not s:
                    continue
                # Normalize to SSE data lines regardless of Vane's framing.
                yield (s if s.startswith("data:") else f"data: {s}") + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class MediaSearchRequest(BaseModel):
    query: str
    model: str = ""


async def _media_search(path: str, req: MediaSearchRequest) -> JSONResponse:
    import httpx

    try:
        await asyncio.to_thread(ensure_vane)
        chat_model = await asyncio.to_thread(_resolve, req.model or EMBEDDING_MODEL, False)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=503)

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        r = await client.post(
            f"{VANE_URL.rstrip('/')}/api/{path}",
            json={"query": req.query, "chatHistory": [], "chatModel": chat_model},
        )
        return JSONResponse(r.json(), status_code=r.status_code)


@router.post("/web-search/images")
async def web_search_images(req: MediaSearchRequest) -> JSONResponse:
    return await _media_search("images", req)


@router.post("/web-search/videos")
async def web_search_videos(req: MediaSearchRequest) -> JSONResponse:
    return await _media_search("videos", req)


if __name__ == "__main__":
    # ponytail: self-check for the one non-trivial bit (provider resolution).
    _sample = [
        {
            "id": "uuid-ollama",
            "name": "Ollama",
            # Ollama's cloud catalog carries deepseek-*:cloud copies (subscription-gated).
            "chatModels": [
                {"name": "Gemma 4", "key": "gemma4:31b-cloud"},
                {"name": "DeepSeek v4 Pro", "key": "deepseek-v4-pro:cloud"},
            ],
            "embeddingModels": [{"name": "BGE M3", "key": "bge-m3:latest"}],
        },
        {
            "id": "uuid-google",
            "name": "Google",
            "chatModels": [{"name": "Flash", "key": "gemini-3.1-flash-lite"}],
            "embeddingModels": [],
        },
    ]

    def _check(model: str, embedding: bool) -> dict[str, str] | None:
        return _resolve_from_providers(_sample, model, embedding)

    assert _check("ollama/gemma4:31b-cloud", False) == {"providerId": "uuid-ollama", "key": "gemma4:31b-cloud"}
    assert _check("ollama/bge-m3:latest", True) == {"providerId": "uuid-ollama", "key": "bge-m3:latest"}
    assert _check("gemini/gemini-3.1-flash-lite", False) == {"providerId": "uuid-google", "key": "gemini-3.1-flash-lite"}
    assert _check("ollama/does-not-exist", False) is None
    # ollama/deepseek-* DOES belong to Ollama (its cloud copy).
    assert _check("ollama/deepseek-v4-pro", False) == {"providerId": "uuid-ollama", "key": "deepseek-v4-pro:cloud"}
    # Regression: deepseek/* must NOT cross-match onto Ollama's deepseek-*:cloud.
    # No DeepSeek provider here → None (caller surfaces "add it in Vane Settings").
    assert _check("deepseek/deepseek-v4-pro", False) is None
    print("ok")
