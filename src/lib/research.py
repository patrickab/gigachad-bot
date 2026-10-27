"""Configure and execute serialized GPT-Researcher runs."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
import contextlib
import json
import os
import tempfile
from typing import Any

from config import OLLAMA_BASE_URL

# GPT-Researcher reads process-wide env vars, so runs must not overlap.
RESEARCH_ENV_LOCK = asyncio.Lock()


@contextlib.contextmanager
def temp_environ(**vals: str | None) -> Iterator[None]:
    """Set env vars for the block, restore originals on exit."""
    old = {k: os.environ.get(k) for k in vals}
    for k, v in vals.items():
        if v is not None:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


async def run_deep_research(
    query: str,
    *,
    fast_model: str,
    smart_model: str,
    strategic_model: str,
    depth: int,
    breadth: int,
    reasoning_effort: str | None,
    report_type: str,
    fallback_model: str,
) -> tuple[str, list[str], float]:
    """Run GPT-Researcher to completion and return (report, source urls, costs)."""
    # Imported lazily because gpt_researcher is heavy.
    from gpt_researcher import GPTResearcher

    # `none` means unset, and `temp_environ` skips None values.
    reasoning = reasoning_effort if reasoning_effort and reasoning_effort != "none" else None
    config: dict[str, Any] = {
        "RETRIEVER": "duckduckgo",
        "EMBEDDING": "ollama:nomic-embed-text",
        "FAST_LLM": f"litellm:{fast_model or fallback_model}",
        "SMART_LLM": f"litellm:{smart_model or fallback_model}",
        "STRATEGIC_LLM": f"litellm:{strategic_model or fallback_model}",
        "DEEP_RESEARCH_DEPTH": depth,
        "DEEP_RESEARCH_BREADTH": breadth,
    }
    if reasoning:
        config["REASONING_EFFORT"] = reasoning
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="gpt-researcher-", delete=False) as tmp:
        json.dump(config, tmp)
    config_path = tmp.name
    async with RESEARCH_ENV_LOCK:
        try:
            with temp_environ(
                OLLAMA_API_BASE=OLLAMA_BASE_URL,
                OLLAMA_BASE_URL=OLLAMA_BASE_URL,
                REASONING_EFFORT=reasoning,
            ):
                researcher = GPTResearcher(
                    query=query,
                    report_type=report_type,
                    config_path=config_path,
                )
                await researcher.conduct_research()
                report = await researcher.write_report()
                return report or "", researcher.get_source_urls(), researcher.get_costs()
        finally:
            with contextlib.suppress(OSError):
                os.unlink(config_path)
