import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import langchain_community.chat_models as _lccm

# gpt_researcher imports `langchain_community.chat_models.litellm`, which the installed
# langchain_community no longer ships, so write a shim re-exporting it from langchain_litellm.
_lccm_dir = Path(_lccm.__file__).parent
_litellm_shim = _lccm_dir / "litellm.py"
if not _litellm_shim.exists():
    _litellm_shim.write_text('from langchain_litellm import ChatLiteLLM\n\n__all__ = ["ChatLiteLLM"]\n')

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routes.architecture import router as architecture_router
from backend.routes.canvases import router as canvases_router
from backend.routes.assets import router as assets_router
from lib.architecture_workspace import ArchitectureError, ArchitectureNotFound
from lib.c4_service import C4ServiceUnavailable
from lib.data_store import StorageConflictError
from backend.routes.chat import router as chat_router
from backend.routes.config import router as config_router
from backend.routes.deps import get_client, shutdown_client
from backend.routes.deploy import router as deploy_router
from backend.routes.neovim import router as neovim_router
from backend.routes.documents import router as documents_router
from backend.routes.files import router as files_router
from backend.routes.fileviewer import router as fileviewer_router
from backend.routes.histories import router as histories_router
from backend.routes.memory import router as memory_router
from lib.mineru import kill_all_mineru_servers, reset_cancel
from lib import omp_source
from backend.routes.models import router as models_router
from backend.routes.file_roots import router as file_roots_router
from backend.routes.file_attach import router as file_attach_router
from backend.routes.file_git import router as file_git_router
from backend.routes.factory import router as factory_router
from backend.routes.ocr import router as ocr_router
from backend.routes.omp import router as omp_router
from backend.routes.projects import router as projects_router
from backend.routes.sync import router as sync_router
from backend.sync import get_change_broker
from lib.db_schema import database_url, upgrade
from lib.postgres_pool import close_postgres_pool
from config import CORS_ORIGINS, is_cors_unsafe_for_production

if is_cors_unsafe_for_production():
    raise RuntimeError("GIGACHAD_CORS_ORIGINS must list exact browser origins in production, never `*`")


# Kill MinerU child servers before exiting so they do not outlive the API.
def _signal_handler(signum: int, frame: object) -> None:
    kill_all_mineru_servers()
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

# `asyncio.to_thread` runs on the loop's default executor, only `min(32, cpus + 4)` workers (12 on 8 cores).
# Streaming chats hold one worker per pending token, sandbox runs and Neovim sessions one for their whole
# lifetime, so a few of those starved short calls (chat save, canvas load, change replay) for seconds.
# Workers start lazily, and database work stays bounded by the connection pool.
DEFAULT_EXECUTOR_WORKERS = 64


def widen_default_executor() -> None:
    """Give the running loop a default executor that long-lived blocking work cannot exhaust."""
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=DEFAULT_EXECUTOR_WORKERS, thread_name_prefix="to-thread")
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    widen_default_executor()
    # Must precede the first LiteLLM call: it teaches the `litellm_proxy/`
    # prefix where the local OMP gateway lives, which is how OMP-sourced
    # models (including OAuth subscription quota) reach a provider.
    omp_source.configure_litellm_proxy()
    get_client()
    upgrade(database_url())
    reset_cancel()
    from lib import extract_queue

    await extract_queue.start()
    await get_change_broker().start(database_url())
    yield
    shutdown_client()
    kill_all_mineru_servers()
    await extract_queue.stop()
    await get_change_broker().stop()
    close_postgres_pool()


app = FastAPI(title="gigachad-bot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    # pdf.js only requests the byte ranges it needs when it can read these on a split-origin deployment.
    expose_headers=["Accept-Ranges", "Content-Range"],
    # Chrome's Private Network Access check blocks a public origin (e.g. the
    # Vercel frontend) from fetching a server whose resolved address is in a
    # private range (Tailscale's CGNAT 100.64.0.0/10) unless the preflight
    # response explicitly grants it. Only grant it when the allow-list is an
    # exact set of origins. Granting it alongside the wildcard default would
    # let any public web page reach this unauthenticated backend.
    allow_private_network="*" not in CORS_ORIGINS,
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


def _json_error(status: int) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
    async def handle(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    return handle


# Architecture routes raise their store's vocabulary; map it to HTTP once here rather than per handler.
# Starlette resolves a handler along the exception's MRO, so a subclass entry wins over its base (not-found is 404, not 400).
# A stale revision is a precondition failure, not a server error: the client must reload and retry.
for _exception, _status in (
    (ArchitectureNotFound, 404),
    (ArchitectureError, 400),
    (C4ServiceUnavailable, 503),
    # A stale revision or taken name. Project rename, chat save and document rename map it to 409 themselves.
    (StorageConflictError, 412),
):
    app.add_exception_handler(_exception, _json_error(_status))


app.include_router(chat_router)
app.include_router(architecture_router)
app.include_router(config_router)
app.include_router(documents_router)
app.include_router(files_router)
app.include_router(assets_router)
app.include_router(fileviewer_router)
app.include_router(histories_router)
app.include_router(memory_router)
app.include_router(models_router)
app.include_router(neovim_router)
app.include_router(deploy_router)
app.include_router(file_roots_router)
app.include_router(file_attach_router)
app.include_router(file_git_router)
app.include_router(ocr_router)
app.include_router(projects_router)
app.include_router(sync_router)
app.include_router(canvases_router)
app.include_router(omp_router)
app.include_router(factory_router)

