from contextlib import asynccontextmanager
from pathlib import Path
import os
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import langchain_community.chat_models as _lccm

_lccm_dir = Path(_lccm.__file__).parent
_litellm_shim = _lccm_dir / "litellm.py"
if not _litellm_shim.exists():
    _litellm_shim.write_text('from langchain_litellm import ChatLiteLLM\n\n__all__ = ["ChatLiteLLM"]\n')

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.routes.architecture_graphs import router as architecture_graphs_router
from lib.architecture_graph import ArchitectureGraphError, ArchitectureGraphNotFound
from backend.routes.chat import router as chat_router
from backend.routes.config import router as config_router
from backend.routes.deps import get_chat_store, get_client, get_project_store, get_prompt_store, shutdown_client
from backend.routes.documents import router as documents_router
from backend.routes.files import router as files_router
from backend.routes.fileviewer import router as fileviewer_router
from backend.routes.histories import router as histories_router
from backend.routes.memory import router as memory_router
from backend.routes.mineru import kill_all_mineru_servers, reset_cancel
from backend.routes.mineru import router as mineru_router
from backend.routes.models import router as models_router
from backend.routes.file_vaults import router as file_vaults_router
from backend.routes.ocr import router as ocr_router
from backend.routes.projects import router as projects_router
from backend.routes.research import router as research_router
from backend.routes.search import router as search_router
from backend.routes.search import stop_vane
from backend.routes.study import router as study_router
from config import DIRECTORY_CHAT_HISTORIES, DIRECTORY_CHAT_UPLOADS, DIRECTORY_OUTPUT_MINERU, ensure_directories

ensure_directories()


def _signal_handler(signum: int, frame: object) -> None:
    kill_all_mineru_servers()
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_client()
    get_chat_store()
    get_project_store()
    get_prompt_store()
    reset_cancel()
    from lib.document_library import backfill_pdf_library
    from lib import extract_queue

    backfill_pdf_library()
    await extract_queue.start()
    yield
    shutdown_client()
    kill_all_mineru_servers()
    stop_vane()
    await extract_queue.stop()


app = FastAPI(title="gigachad-bot", lifespan=lifespan)

# Defaults to the pre-desktop wildcard; the Tauri shell narrows it to its own
# webview origins via this env var, and server deployments can do the same.
_cors_origins = [o.strip() for o in os.environ.get("GIGACHAD_CORS_ORIGINS", "*").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    # Chrome's Private Network Access check blocks a public origin (e.g. the
    # Vercel frontend) from fetching a server whose resolved address is in a
    # private range (Tailscale's CGNAT 100.64.0.0/10) unless the preflight
    # response explicitly grants it. Only grant it when the allow-list is an
    # exact set of origins — granting it alongside the wildcard default would
    # let any public web page reach this unauthenticated backend.
    allow_private_network="*" not in _cors_origins,
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}

# Graph routes raise their store's vocabulary; map it to HTTP once here rather than per handler.
# The subclass is registered first so Starlette's MRO lookup gives not-found a 404, not a 400.
@app.exception_handler(ArchitectureGraphNotFound)
async def _architecture_graph_not_found(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ArchitectureGraphError)
async def _architecture_graph_invalid(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


app.include_router(chat_router)
app.include_router(architecture_graphs_router)
app.include_router(config_router)
app.include_router(documents_router)
app.include_router(files_router)
app.include_router(fileviewer_router)
app.include_router(histories_router)
app.include_router(memory_router)
app.include_router(models_router)
app.include_router(search_router)
app.include_router(file_vaults_router)
app.include_router(mineru_router)
app.include_router(ocr_router)
app.include_router(projects_router)
app.include_router(research_router)
app.include_router(study_router)

if (DIRECTORY_OUTPUT_MINERU / "images").exists():
    app.mount("/mineru/images", StaticFiles(directory=str(DIRECTORY_OUTPUT_MINERU / "images")), name="mineru_images")
# Non-project uploads are still served from /chat-uploads for backward compatibility.
# Project-scoped uploads are served from /chat-histories/<slug>/_uploads/ (mounted below).
if DIRECTORY_CHAT_UPLOADS.exists():
    app.mount("/chat-uploads", StaticFiles(directory=str(DIRECTORY_CHAT_UPLOADS)), name="chat_uploads")
if DIRECTORY_CHAT_HISTORIES.exists():
    app.mount("/chat-histories", StaticFiles(directory=str(DIRECTORY_CHAT_HISTORIES), html=False), name="chat_histories")
