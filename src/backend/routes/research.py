import contextlib
import json
import os
import tempfile
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from config import OLLAMA_BASE_URL

router = APIRouter(prefix="/api", tags=["research"])


class ResearchRequest(BaseModel):
    query: str
    fast_model: str
    smart_model: str
    strategic_model: str
    depth: int = 2
    breadth: int = 4
    reasoning_effort: str | None = None
    report_type: str = "deep"


@router.post("/research")
async def research(req: ResearchRequest) -> dict[str, Any]:
    from gpt_researcher import GPTResearcher

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

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="gpt-researcher-", delete=False
    ) as tmp:
        json.dump(config, tmp)
        config_path = tmp.name

    saved_env: dict[str, str | None] = {}
    env_keys = ["OLLAMA_API_BASE", "OLLAMA_BASE_URL", "REASONING_EFFORT"]
    for key in env_keys:
        saved_env[key] = os.environ.get(key)

    try:
        os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL
        os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL
        if req.reasoning_effort and req.reasoning_effort != "none":
            os.environ["REASONING_EFFORT"] = req.reasoning_effort

        researcher = GPTResearcher(
            query=req.query,
            report_type=req.report_type,
            config_path=config_path,
        )
        await researcher.conduct_research()
        report = await researcher.write_report()
        sources = researcher.get_source_urls()
        costs = researcher.get_costs()
    finally:
        for key, val in saved_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        with contextlib.suppress(OSError):
            os.unlink(config_path)

    return {"report": report, "sources": sources, "costs": costs}