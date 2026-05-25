import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

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
from lib.non_user_prompts import SYS_TAVILY_QUERY_EXPANSION, SYS_OCR_TEXT_EXTRACTION
from llm_client import LLMClient
from llm_config import MODEL_CONFIGS, DEFAULT_VISION_MODEL

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
    downscale_images: bool = True
    messages: list[dict[str, Any]] = []


class SaveRequest(BaseModel):
    filename: str | None = None
    messages: list[dict[str, Any]] = []


class MoveRequest(BaseModel):
    target_dir: str | None = None


class ResearchRequest(BaseModel):
    query: str
    fast_model: str
    smart_model: str
    strategic_model: str
    depth: int = 2
    breadth: int = 4
    reasoning_effort: str | None = None
    report_type: str = "deep"


class TavilySearchRequest(BaseModel):
    query: str
    num_queries: int = 3
    results_per_query: int = 5
    expander_model: str = "ollama/gemma4:31b-cloud"


class OCRRequest(BaseModel):
    img_base64: str
    model: str = ""


class DownscaleRequest(BaseModel):
    img_base64: str
    max_tokens: int = 2048


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
    return {"messages": []}


@app.delete("/api/history")
async def reset_history() -> dict[str, str]:
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
            for chunk in c.api_query(
                model=req.model,
                user_msg=req.user_msg,
                user_msg_history=req.messages,
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

    response = c.api_query(
        model=req.model,
        user_msg=req.user_msg,
        user_msg_history=req.messages,
        system_prompt=req.system_prompt,
        img=img,
        stream=False,
        **kwargs,
    )
    content = response.choices[0].message.content or ""
    return {"content": content}


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
    messages = json.loads(filepath.read_text(encoding="utf-8"))
    return {"messages": messages, "filename": filename}


@app.put("/api/chat-histories/{filename:path}")
async def save_chat_history(filename: str, data: SaveRequest | None = None) -> dict[str, str]:
    DIRECTORY_CHAT_HISTORIES.mkdir(parents=True, exist_ok=True)
    name = filename if not data or not data.filename else data.filename
    filepath = DIRECTORY_CHAT_HISTORIES / name
    filepath.write_text(json.dumps(data.messages if data else [], indent=2), encoding="utf-8")
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


@app.post("/api/research")
async def research(req: ResearchRequest) -> dict[str, Any]:
    from gpt_researcher import GPTResearcher

    config_path = Path(__file__).resolve().parent.parent.parent / ".gpt-researcher-config.json"

    config: dict[str, Any] = {
        "RETRIEVER": "tavily",
        "EMBEDDING": "ollama:nomic-embed-text",
        "FAST_LLM": f"litellm:{req.fast_model}",
        "SMART_LLM": f"litellm:{req.smart_model}",
        "STRATEGIC_LLM": f"litellm:{req.strategic_model}",
        "DEEP_RESEARCH_DEPTH": req.depth,
        "DEEP_RESEARCH_BREADTH": req.breadth,
    }
    if req.reasoning_effort and req.reasoning_effort != "none":
        config["REASONING_EFFORT"] = req.reasoning_effort

    with config_path.open("w") as f:
        json.dump(config, f)

    os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    if req.reasoning_effort and req.reasoning_effort != "none":
        os.environ["REASONING_EFFORT"] = req.reasoning_effort

    researcher = GPTResearcher(
        query=req.query,
        report_type=req.report_type,
        config_path=str(config_path),
    )
    await researcher.conduct_research()
    report = await researcher.write_report()
    sources = researcher.get_source_urls()
    costs = researcher.get_costs()

    return {"report": report, "sources": sources, "costs": costs}


@app.post("/api/tavily-search")
def tavily_search(req: TavilySearchRequest) -> dict[str, Any]:
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="TAVILY_API_KEY not set")

    c = get_client()
    prompt = SYS_TAVILY_QUERY_EXPANSION.replace("{k}", str(req.num_queries))

    queries: list[str] = []
    try:
        response = c.api_query(
            model=req.expander_model,
            user_msg=req.query,
            system_prompt=prompt,
            temperature=0.4,
            top_p=0.95,
            stream=False,
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("\n", 1)[0]
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
            queries = parsed[:req.num_queries]
    except Exception:
        queries = [req.query]

    if not queries:
        queries = [req.query]

    all_results: list[dict[str, Any]] = []
    for q in queries:
        try:
            r = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": q,
                    "max_results": req.results_per_query,
                    "search_depth": "basic",
                },
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            for item in data.get("results", [])[:req.results_per_query]:
                all_results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                })
        except Exception:
            pass

    return {"results": all_results, "queries": queries}


@app.post("/api/ocr")
async def ocr(req: OCRRequest) -> StreamingResponse:
    c = get_client()
    model = req.model or DEFAULT_VISION_MODEL
    img = _decode_image(req.img_base64)

    async def event_stream() -> Any:
        try:
            for chunk in c.api_query(
                model=model,
                user_msg="Extract all text and LaTeX from this image.",
                system_prompt=SYS_OCR_TEXT_EXTRACTION,
                img=img,
                temperature=0.1,
                top_p=0.95,
                stream=True,
            ):
                yield {"event": "token", "data": chunk}
            yield {"event": "done", "data": ""}
        except Exception as e:
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_stream())


@app.post("/api/downscale-image")
def downscale_image(req: DownscaleRequest) -> dict[str, str]:
    c = get_client()
    try:
        result = c.downscale_img(img=req.img_base64, max_tokens=req.max_tokens)
        return {"img_base64": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
