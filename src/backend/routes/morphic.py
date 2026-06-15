import asyncio
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import MORPHIC_URL

router = APIRouter(prefix="/api", tags=["morphic"])

_COMPOSE_FILE = str(Path(__file__).resolve().parent.parent.parent.parent / "docker-compose.morphic.yml")
_morphic_started = False
_morphic_lock = threading.Lock()


def _docker_context() -> str | None:
    for ctx in ("default", "rootless"):
        try:
            r = subprocess.run(
                ["docker", "--context", ctx, "info"],
                capture_output=True,
                timeout=5,
            )
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
    daemon = subprocess.Popen(
        ["dockerd-rootless.sh"],
        env={**__import__("os").environ, "HOME": str(Path.home())},
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


def _is_morphic_ready(timeout: float = 30.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            r = subprocess.run(
                ["curl", "-sf", MORPHIC_URL],
                capture_output=True,
                timeout=3,
            )
            if r.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        time.sleep(1)
    return False


def ensure_morphic() -> None:
    global _morphic_started
    if _morphic_started:
        return
    with _morphic_lock:
        if _morphic_started:
            return
        ctx = _start_docker_daemon()
        print("Starting morphic container stack (first request)...", file=sys.stderr)
        subprocess.run(
            ["docker", "--context", ctx, "compose", "-f", _COMPOSE_FILE, "up", "-d", "--wait"],
            check=True,
        )
        if not _is_morphic_ready():
            raise RuntimeError("Morphic did not become ready within 30 seconds")
        print("Morphic is ready.", file=sys.stderr)
        _morphic_started = True


def stop_morphic() -> None:
    global _morphic_started
    if not _morphic_started:
        return
    print("Stopping morphic container stack...", file=sys.stderr)
    ctx = _docker_context()
    if ctx:
        subprocess.run(
            ["docker", "--context", ctx, "compose", "-f", _COMPOSE_FILE, "down"],
            check=False,
        )
    _morphic_started = False
    print("Morphic stopped.", file=sys.stderr)


class MorphicSearchRequest(BaseModel):
    query: str
    search_depth: str = "adaptive"
    model: str = ""


@router.post("/morphic-search")
async def morphic_search(req: MorphicSearchRequest) -> StreamingResponse:
    import httpx

    try:
        await asyncio.to_thread(ensure_morphic)
    except Exception as e:
        import json

        error_msg = str(e)

        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'errorText': f'morphic startup failed: {error_msg}'})}\n\n"

        return StreamingResponse(
            error_stream(),
            media_type="text/plain; charset=utf-8",
        )

    chat_id = str(uuid.uuid4())

    morphic_payload = {
        "message": {"role": "user", "parts": [{"type": "text", "text": req.query}]},
        "chatId": chat_id,
        "trigger": "submit-message",
        "isNewChat": True,
    }

    cookies: dict[str, str] = {"searchMode": req.search_depth}
    if req.model:
        provider_id = "ollama"
        model_id = req.model
        if model_id.startswith("ollama/"):
            model_id = model_id[len("ollama/") :]
        elif "/" in model_id:
            provider_id, model_id = model_id.split("/", 1)
        cookies["selectedModel"] = f"{provider_id}:{model_id}"

    async def event_stream():
        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client,
            client.stream(
                "POST",
                f"{MORPHIC_URL.rstrip('/')}/api/chat",
                json=morphic_payload,
                headers={
                    "Content-Type": "application/json",
                    "Cookie": "; ".join(f"{k}={v}" for k, v in cookies.items()),
                },
            ) as response,
        ):
            if response.status_code != 200:
                error_body = await response.aread()
                yield f"event: error\ndata: {error_body.decode(errors='replace')}\n\n"
                return
            async for line in response.aiter_lines():
                if line.strip():
                    yield line + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
