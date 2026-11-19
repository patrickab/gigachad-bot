import asyncio

import pytest

from backend.routes.models import get_reasoning_levels
from lib import reasoning_levels
from lib.reasoning_levels import effort_kwargs


def _stub_gateway(monkeypatch: pytest.MonkeyPatch, ladders: dict[str, list[str]]) -> list[str]:
    """Serve `ladders` as the gateway's `thinking_levels`, returning the gateway ids asked for."""
    requested: list[str] = []

    def thinking_levels(model_id: str) -> list[str] | None:
        requested.append(model_id)
        return ladders.get(model_id)

    monkeypatch.setattr(reasoning_levels.pi_models, "thinking_levels", thinking_levels)
    return requested


@pytest.mark.parametrize(
    ("selector", "gateway_id", "ladder", "expected"),
    [
        ("litellm_proxy/anthropic/claude-opus-4-5", "anthropic/claude-opus-4-5", ["off", "minimal", "low", "medium", "high"], ["none", "minimal", "low", "medium", "high"]),
        ("openai/gpt-5.6", "openai/gpt-5.6", ["minimal", "low", "medium", "high", "xhigh"], ["minimal", "low", "medium", "high", "xhigh"]),
        ("gemini/gemini-3.1-pro", "google/gemini-3.1-pro", ["off", "low", "high"], ["none", "low", "high"]),
        ("ollama/qwen3:8b", "ollama/qwen3:8b", [], []),
    ],
)
def test_levels_are_the_gateways_ladder_under_its_model_id(
    monkeypatch: pytest.MonkeyPatch, selector: str, gateway_id: str, ladder: list[str], expected: list[str]
) -> None:
    """Pi's `off` (present only when thinking can be off) is the app's `none`."""
    requested = _stub_gateway(monkeypatch, {gateway_id: ladder})
    assert asyncio.run(get_reasoning_levels(selector)) == {"levels": expected}
    assert requested == [gateway_id]


def test_unlisted_model_or_offline_gateway_has_no_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_gateway(monkeypatch, {})
    assert reasoning_levels.levels("deepseek/deepseek-chat") == []


@pytest.mark.parametrize(
    ("effort", "expected"),
    [(None, {}), ("", {}), ("none", {}), ("off", {}), ("high", {"reasoning_effort": "high"}), ("max", {"reasoning_effort": "max"})],
)
def test_effort_kwargs_omit_only_an_off_level(effort: str | None, expected: dict) -> None:
    assert effort_kwargs(effort) == expected
