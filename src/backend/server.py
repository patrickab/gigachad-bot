import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.http_plumbing import TransferMiddleware
from backend.routes import (
    architecture,
    assets,
    canvases,
    chat,
    config,
    deploy,
    documents,
    factory,
    factory_preview,
    file_attach,
    file_git,
    file_roots,
    file_watch,
    files,
    fileviewer,
    histories,
    jupyter,
    latex,
    learn,
    memory,
    models,
    neovim,
    ocr,
    pi,
    projects,
    share_links,
    sync,
    terminal,
    theme,
    workbench,
)
from backend.routes.upstream import UPSTREAM_ERRORS
from backend.sync import get_change_broker
from config import CORS_ORIGINS, is_cors_unsafe_for_production
from lib import factory_migration, jupyter_service, ollama_models, repository_index, upstream
from lib.architecture_workspace import ArchitectureError, ArchitectureNotFound
from lib.c4_service import C4ServiceUnavailable
from lib.data_store import StorageConflictError
from lib.db_schema import database_url, upgrade
from lib.mineru import kill_all_mineru_servers, reset_cancel
from lib.postgres_pool import close_postgres_pool, get_postgres_pool
from lib.process_owner import reap_orphans, stop_owned

log = logging.getLogger(__name__)

if is_cors_unsafe_for_production():
    raise RuntimeError("GIGACHAD_CORS_ORIGINS must list exact browser origins in production, never `*`")


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


# pi-host may start after the backend: a move that failed is tried again a few times.
FACTORY_MOVE_ATTEMPTS = 5
FACTORY_MOVE_RETRY_SECONDS = 30.0


async def _move_factory_rows() -> None:
    for attempt in range(FACTORY_MOVE_ATTEMPTS):
        if attempt:
            await asyncio.sleep(FACTORY_MOVE_RETRY_SECONDS)
        try:
            if await factory_migration.migrate_factory_rows(get_postgres_pool()):
                return
        except Exception:
            log.exception("The factory rows did not move into the factory core")
    log.warning("The factory rows not moved yet stay until the next start")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    if reaped := reap_orphans():  # children an earlier backend left behind when it died without stopping them
        log.warning("Stopped %d processes an earlier backend left running: %s", len(reaped), reaped)
    widen_default_executor()
    # Pi's provider `ollama` follows the installed models, so pi-host's gateway serves them (Ollama down: no-op).
    asyncio.get_running_loop().run_in_executor(None, ollama_models.sync_pi_provider)
    upgrade(database_url())
    # The factory core owns factory repos and tasks now: the old rows move into it once each (needs pi-host, retried each start).
    factory_move = asyncio.create_task(_move_factory_rows())
    # Pi's start folders: requests serve the stored list at once, this rescan of `~` brings it up to date.
    asyncio.get_running_loop().run_in_executor(None, repository_index.sweep_quietly)
    reset_cancel()
    from lib import extract_queue

    await extract_queue.start()
    await get_change_broker().start(database_url())
    yield
    factory_move.cancel()
    kill_all_mineru_servers()
    await jupyter_service.aclose()
    await upstream.close_clients()
    if stopped := stop_owned():  # what Neovim sessions left running: language servers, programs they ran
        log.info("Stopped %d processes the backend's children left running: %s", len(stopped), stopped)
    await extract_queue.stop()
    await get_change_broker().stop()
    close_postgres_pool()


app = FastAPI(title="gigachad-bot", lifespan=lifespan, exception_handlers=UPSTREAM_ERRORS)

# Inside CORS, so 304s and gzipped bodies still carry the CORS headers.
app.add_middleware(TransferMiddleware)
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
    # The frontend switches to credentialed requests to escape a dead pooled connection. Echoing any origin with
    # credentials would be unsafe, so only exact allow-lists grant them.
    allow_credentials="*" not in CORS_ORIGINS,
    # A split-origin frontend preflights every distinct URL (custom `X-Device-Id` header):
    # cache each for Chrome's 2 h cap, not Starlette's 10 min.
    max_age=7200,
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
    # A stale revision or taken name. Project rename and document rename map it to 409 themselves.
    (StorageConflictError, 412),
):
    app.add_exception_handler(_exception, _json_error(_status))


for _router in (
    chat.router,
    architecture.router,
    config.router,
    theme.router,
    documents.router,
    files.router,
    file_watch.router,
    assets.router,
    fileviewer.router,
    histories.router,
    memory.router,
    models.router,
    neovim.router,
    terminal.router,
    jupyter.router,
    latex.router,
    workbench.router,
    deploy.router,
    file_roots.router,
    file_attach.router,
    file_git.router,
    ocr.router,
    projects.router,
    sync.router,
    canvases.router,
    share_links.owner_router,
    share_links.guest_router,
    pi.router,
    learn.router,
    factory.router,
    factory_preview.router,
):
    app.include_router(_router)
