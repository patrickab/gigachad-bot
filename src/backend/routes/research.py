from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

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
    import json
    import os
    from pathlib import Path

    from gpt_researcher import GPTResearcher

    config_path = Path(__file__).resolve().parent.parent.parent.parent / ".gpt-researcher-config.json"

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