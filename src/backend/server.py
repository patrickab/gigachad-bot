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
    _litellm_shim.write_text("from langchain_litellm import ChatLiteLLM\n\n__all__ = [\"ChatLiteLLM\"]\n")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes.chat import router as chat_router
from backend.routes.deps import get_client, shutdown_client
from backend.routes.files import router as files_router
from backend.routes.histories import router as histories_router
from backend.routes.mineru import kill_all_mineru_servers, reset_cancel
from backend.routes.mineru import router as mineru_router
from backend.routes.models import router as models_router
from backend.routes.morphic import router as morphic_router, stop_morphic
from backend.routes.ocr import router as ocr_router
from backend.routes.research import router as research_router
from config import DIRECTORY_CHAT_UPLOADS, DIRECTORY_OUTPUT_MINERU, ensure_directories

ensure_directories()


def _signal_handler(signum: int, frame: object) -> None:
    kill_all_mineru_servers()
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_client()
    reset_cancel()
    yield
    shutdown_client()
    kill_all_mineru_servers()
    stop_morphic()


app = FastAPI(title="gigachad-bot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(files_router)
app.include_router(histories_router)
app.include_router(models_router)
app.include_router(morphic_router)
app.include_router(mineru_router)
app.include_router(ocr_router)
app.include_router(research_router)

app.mount("/mineru/images", StaticFiles(directory=str(DIRECTORY_OUTPUT_MINERU / "images")), name="mineru_images")

if DIRECTORY_CHAT_UPLOADS.exists():
    app.mount("/chat-uploads", StaticFiles(directory=str(DIRECTORY_CHAT_UPLOADS)), name="chat_uploads")