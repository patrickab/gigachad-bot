import asyncio
import os
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from config import OLLAMA_BASE_URL
from lib.research_config import build_research_config, write_research_config

router = APIRouter(prefix="/api", tags=["research"])

_env_lock = asyncio.Lock()


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

    config = build_research_config(
        fast_model=req.fast_model,
        smart_model=req.smart_model,
        strategic_model=req.strategic_model,
        depth=req.depth,
        breadth=req.breadth,
        reasoning_effort=req.reasoning_effort,
    )
    config_path = write_research_config(config)

    async with _env_lock:
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
            import contextlib
            with contextlib.suppress(OSError):
                os.unlink(config_path)

    return {"report": report, "sources": sources, "costs": costs}