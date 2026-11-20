"""GPT-Researcher runs on pi-host's gateway: the config and environment one run sees."""

import json
import os
from pathlib import Path
import sys
import types
from typing import ClassVar

import pytest

from lib import research

RUN_ENV = ("VLLM_OPENAI_API_BASE", "VLLM_OPENAI_API_KEY", "OPENAI_API_KEY", "TIKTOKEN_CACHE_DIR")


class FakeResearcher:
    """Records what GPT-Researcher would read: its config file and the process environment."""

    seen: ClassVar[dict] = {}

    def __init__(self, *, query: str, report_type: str, config_path: str) -> None:
        FakeResearcher.seen = {
            "config": json.loads(Path(config_path).read_text()),
            "env": {name: os.environ.get(name) for name in RUN_ENV},
        }

    async def conduct_research(self) -> None:
        pass

    async def write_report(self) -> str:
        return "report"

    def get_source_urls(self) -> list[str]:
        return ["https://example.org/scraped"]

    def get_research_sources(self) -> list[dict]:
        """Search hits used without scraping, one of them also scraped."""
        return [{"url": "https://example.org/hit"}, {"url": "https://example.org/scraped"}, {"title": "no url"}]

    def get_costs(self) -> float:
        return 0.5


@pytest.fixture(autouse=True)
def _fake_gpt_researcher(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "gpt_researcher", types.SimpleNamespace(GPTResearcher=FakeResearcher))
    monkeypatch.setenv("GIGACHAD_PI_HOST_URL", "http://127.0.0.1:8399")
    (tmp_path / "pi-gateway.token").write_text("gateway-token\n", encoding="utf-8")
    monkeypatch.setenv("PI_GATEWAY_TOKEN_FILE", str(tmp_path / "pi-gateway.token"))
    for name in RUN_ENV:
        monkeypatch.delenv(name, raising=False)


async def _run(reasoning_effort: str | None) -> tuple[str, list[str], float]:
    return await research.run_deep_research(
        "query",
        fast_model="litellm_proxy/anthropic/claude-haiku-5-5",
        smart_model="",
        strategic_model="ollama/qwen3:8b",
        depth=1,
        breadth=1,
        reasoning_effort=reasoning_effort,
        report_type="deep",
        fallback_model="gemini/gemini-3.5-flash",
    )


async def test_models_and_effort_go_to_the_gateway_for_the_run_only():
    report, _, costs = await _run("high")
    assert (report, costs) == ("report", 0.5)

    config, env = FakeResearcher.seen["config"], FakeResearcher.seen["env"]
    assert (config["FAST_LLM"], config["SMART_LLM"], config["STRATEGIC_LLM"]) == (
        "vllm_openai:anthropic/claude-haiku-5-5",
        "vllm_openai:google/gemini-3.5-flash",
        "vllm_openai:ollama/qwen3:8b",
    )
    assert config["LLM_KWARGS"] == {"reasoning_effort": "high"}
    assert "REASONING_EFFORT" not in config
    # The gateway-only token, and the user's own `OPENAI_API_KEY` stays untouched.
    assert env == {
        "VLLM_OPENAI_API_BASE": "http://127.0.0.1:8399/v1",
        "VLLM_OPENAI_API_KEY": "gateway-token",
        "OPENAI_API_KEY": None,
        "TIKTOKEN_CACHE_DIR": research.TIKTOKEN_CACHE,
    }
    assert all(name not in os.environ for name in RUN_ENV)


async def test_sources_are_scraped_pages_and_search_hits_once_each():
    _, urls, _ = await _run(None)

    assert urls == ["https://example.org/scraped", "https://example.org/hit"]


async def test_effort_none_sends_no_effort():
    await _run("none")

    assert "LLM_KWARGS" not in FakeResearcher.seen["config"]
