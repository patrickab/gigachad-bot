"""Configure and execute serialized GPT-Researcher runs, its chat models served by pi-host's gateway."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
import contextlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from config import OLLAMA_BASE_URL
from lib import pi_host, pi_models

# GPT-Researcher reads process-wide env vars, so runs must not overlap.
RESEARCH_ENV_LOCK = asyncio.Lock()
# GPT-Researcher prices every report with tiktoken, whose encodings download once into a cache (by default under /tmp,
# gone after a reboot). A failed download costs the whole report: `write_report` swallows the error and returns "".
TIKTOKEN_CACHE = os.environ.get("TIKTOKEN_CACHE_DIR") or str(
    Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "gigachad-bot" / "tiktoken"
)


@contextlib.contextmanager
def temp_environ(**vals: str) -> Iterator[None]:
    """Set env vars for the block, restore originals on exit."""
    old = {k: os.environ.get(k) for k in vals}
    os.environ.update(vals)
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

    # GPT-Researcher's `vllm_openai` provider is langchain's ChatOpenAI on `VLLM_OPENAI_API_BASE`/`_KEY`, any
    # OpenAI-compatible server, here pi-host's gateway. Its `openai` provider reads `OPENAI_API_KEY`, which whatever the
    # backend spawns during the run (a terminal) would inherit in place of the user's own key.
    config: dict[str, Any] = {
        "RETRIEVER": "duckduckgo",
        # The gateway serves no embeddings, they stay on Ollama.
        "EMBEDDING": "ollama:nomic-embed-text",
        "FAST_LLM": f"vllm_openai:{pi_models.gateway_model_id(fast_model or fallback_model)}",
        "SMART_LLM": f"vllm_openai:{pi_models.gateway_model_id(smart_model or fallback_model)}",
        "STRATEGIC_LLM": f"vllm_openai:{pi_models.gateway_model_id(strategic_model or fallback_model)}",
        "DEEP_RESEARCH_DEPTH": depth,
        "DEEP_RESEARCH_BREADTH": breadth,
    }
    # Its own `REASONING_EFFORT` reaches OpenAI o-series names only. `LLM_KWARGS` reach every call that passes them
    # (deep research's plan and learnings calls do not), the gateway clamps the effort per model. `none` means unset.
    if reasoning_effort and reasoning_effort != "none":
        config["LLM_KWARGS"] = {"reasoning_effort": reasoning_effort}
    # GPT-Researcher reads its config from a file path.
    with tempfile.TemporaryDirectory(prefix="gpt-researcher-") as config_dir:
        config_path = Path(config_dir) / "config.json"
        config_path.write_text(json.dumps(config))
        async with RESEARCH_ENV_LOCK:
            with temp_environ(
                VLLM_OPENAI_API_BASE=pi_host.gateway_url(),
                # The gateway-only token, since processes the backend spawns during the run inherit it.
                VLLM_OPENAI_API_KEY=pi_host.gateway_token(),
                OLLAMA_API_BASE=OLLAMA_BASE_URL,
                OLLAMA_BASE_URL=OLLAMA_BASE_URL,
                TIKTOKEN_CACHE_DIR=TIKTOKEN_CACHE,
            ):
                researcher = GPTResearcher(
                    query=query,
                    report_type=report_type,
                    config_path=str(config_path),
                )
                await researcher.conduct_research()
                report = await researcher.write_report()
                # Scraped pages are its source urls. A search hit with a snippet over 100 characters is used as it came
                # (DuckDuckGo's usually are) and lands only in its research sources.
                sources = [*researcher.get_source_urls(), *(source.get("url") for source in researcher.get_research_sources())]
                return report or "", list(dict.fromkeys(url for url in sources if url)), researcher.get_costs()
