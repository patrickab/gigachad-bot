from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.chat import router as chat_router
from backend.routes.deps import get_client, shutdown_client
from backend.routes.histories import router as histories_router
from backend.routes.models import router as models_router
from backend.routes.morphic import router as morphic_router
from backend.routes.ocr import router as ocr_router
from backend.routes.research import router as research_router

app = FastAPI(title="gigachad-bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(histories_router)
app.include_router(models_router)
app.include_router(morphic_router)
app.include_router(ocr_router)
app.include_router(research_router)


@app.on_event("startup")
async def startup() -> None:
    get_client()


@app.on_event("shutdown")
async def shutdown() -> None:
    shutdown_client()