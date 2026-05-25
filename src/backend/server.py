import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from fastapi.responses import StreamingResponse

from config import DIRECTORY_CHAT_HISTORIES
from lib.prompts import (
    SYS_ADVISOR,
    SYS_ARTICLE,
    SYS_CONCEPT_IN_DEPTH,
    SYS_CONCEPTUAL_OVERVIEW,
    SYS_EMPTY_PROMPT,
    SYS_QUICK_OVERVIEW,
    SYS_TUTOR,
)
from llm_client import LLMClient
from llm_config import MODEL_CONFIGS

try:
    from llm_baseclient.config import AVAILABLE_MODELS, MODELS_GEMINI, MODELS_OLLAMA, MODELS_OPENAI
except ImportError:
    AVAILABLE_MODELS = MODELS_GEMINI = MODELS_OLLAMA = MODELS_OPENAI = []

app = FastAPI(title="gigachad-bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client: LLMClient | None = None

PROMPT_MAP: dict[str, str] = {
    "Quick Overview": SYS_QUICK_OVERVIEW,
    "Advisor": SYS_ADVISOR,
    "Tutor": SYS_TUTOR,
    "Concept - High-Level": SYS_CONCEPTUAL_OVERVIEW,
    "Concept - In-Depth": SYS_CONCEPT_IN_DEPTH,
    "Concept - Article": SYS_ARTICLE,
    "<empty prompt>": SYS_EMPTY_PROMPT,
}


class ChatRequest(BaseModel):
    model: str
    user_msg: str
    system_prompt: str = ""
    temperature: float = 0.2
    top_p: float = 0.95
    reasoning_effort: str | None = None
    img_base64: str | None = None


class SaveRequest(BaseModel):
    filename: str


class MoveRequest(BaseModel):
    target_dir: str | None = None


def get_client() -> LLMClient:
    global client
    if client is None:
        client = LLMClient(model_configs=MODEL_CONFIGS)
    return client


@app.on_event("startup")
async def startup() -> None:
    get_client()


@app.on_event("shutdown")
async def shutdown() -> None:
    global client
    if client:
        client.kill_inference_engines()


@app.get("/api/models")
async def get_models() -> dict[str, Any]:
    return {
        "all": AVAILABLE_MODELS,
        "ollama": MODELS_OLLAMA,
        "gemini": MODELS_GEMINI,
        "openai": MODELS_OPENAI,
    }


@app.get("/api/prompts")
async def get_prompts() -> dict[str, list[str]]:
    return {"prompts": list(PROMPT_MAP.keys())}


@app.get("/api/history")
async def get_history() -> dict[str, Any]:
    return {"messages": get_client().messages}


@app.delete("/api/history")
async def reset_history() -> dict[str, str]:
    get_client().reset_history()
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    c = get_client()
    kwargs: dict[str, Any] = {"temperature": req.temperature, "top_p": req.top_p}
    if req.reasoning_effort and req.reasoning_effort != "none":
        kwargs["reasoning_effort"] = req.reasoning_effort

    img = _decode_image(req.img_base64)

    async def event_stream() -> Any:
        try:
            for chunk in c.chat(
                model=req.model,
                user_msg=req.user_msg,
                system_prompt=req.system_prompt,
                img=img,
                stream=True,
                **kwargs,
            ):
                yield {"event": "token", "data": chunk}
            yield {"event": "done", "data": ""}
        except Exception as e:
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_stream())


@app.post("/api/chat/nonstream")
async def chat_nonstream(req: ChatRequest) -> dict[str, Any]:
    c = get_client()
    kwargs: dict[str, Any] = {"temperature": req.temperature, "top_p": req.top_p}
    if req.reasoning_effort and req.reasoning_effort != "none":
        kwargs["reasoning_effort"] = req.reasoning_effort

    img = _decode_image(req.img_base64)

    response = c.chat(
        model=req.model,
        user_msg=req.user_msg,
        system_prompt=req.system_prompt,
        img=img,
        stream=False,
        **kwargs,
    )
    content = response.choices[0].message.content or ""
    return {"content": content, "messages": c.messages}


@app.get("/api/chat-histories")
async def list_chat_histories() -> dict[str, Any]:
    if not DIRECTORY_CHAT_HISTORIES.exists():
        return {"histories": {}}

    histories: dict[str, list[str]] = {}
    for f in sorted(DIRECTORY_CHAT_HISTORIES.rglob("*.json")):
        if f.is_file() and "archived" not in f.parts:
            dir_key = str(f.parent.relative_to(DIRECTORY_CHAT_HISTORIES)) if f.parent != DIRECTORY_CHAT_HISTORIES else "root"
            histories.setdefault(dir_key, []).append(f.name)
    return {"histories": histories}


@app.get("/api/chat-histories/{filename:path}")
async def load_chat_history(filename: str) -> dict[str, Any]:
    filepath = _resolve_history_path(filename)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat history not found")
    get_client().load_history(str(filepath))
    return {"messages": get_client().messages, "filename": filename}


@app.put("/api/chat-histories/{filename:path}")
async def save_chat_history(filename: str, data: SaveRequest | None = None) -> dict[str, str]:
    DIRECTORY_CHAT_HISTORIES.mkdir(parents=True, exist_ok=True)
    name = filename if not data or not data.filename else data.filename
    filepath = DIRECTORY_CHAT_HISTORIES / name
    get_client().store_history(str(filepath))
    return {"status": "ok", "filename": str(name)}


@app.delete("/api/chat-histories/{filename:path}")
async def delete_chat_history(filename: str) -> dict[str, str]:
    filepath = _resolve_history_path(filename)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat history not found")
    filepath.unlink()
    return {"status": "ok"}


@app.put("/api/chat-histories/{filename:path}/archive")
async def archive_chat_history(filename: str) -> dict[str, str]:
    filepath = _resolve_history_path(filename)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat history not found")
    archive_dir = DIRECTORY_CHAT_HISTORIES / "archived"
    archive_dir.mkdir(parents=True, exist_ok=True)
    filepath.rename(archive_dir / filepath.name)
    return {"status": "ok"}


@app.post("/api/chat-histories/{filename:path}/unarchive")
async def unarchive_chat_history(filename: str) -> dict[str, str]:
    archive_dir = DIRECTORY_CHAT_HISTORIES / "archived"
    filepath = archive_dir / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Unreachable history")
    filepath.rename(DIRECTORY_CHAT_HISTORIES / filepath.name)
    return {"status": "ok"}


def _resolve_history_path(filename: str) -> Path:
    return DIRECTORY_CHAT_HISTORIES / filename


def _decode_image(base64_data: str | None) -> bytes | None:
    import base64
    import re

    if not base64_data:
        return None
    match = re.match(r"data:image/\w+;base64,(.+)", base64_data)
    if match:
        return base64.b64decode(match.group(1))
    return base64.b64decode(base64_data)
