import asyncio

import pytest

from backend.routes import models
from backend.routes.models import _litellm_reasoning_levels, get_reasoning_levels


def _stub_litellm(monkeypatch: pytest.MonkeyPatch, supports_reasoning: bool, model_info: dict | Exception) -> None:
    """Replace LiteLLM's registry lookups with a fixed answer."""
    monkeypatch.setattr(models.litellm, "supports_reasoning", lambda model: supports_reasoning)

    def get_model_info(model: str) -> dict:
        if isinstance(model_info, Exception):
            raise model_info
        return model_info

    monkeypatch.setattr(models.litellm, "get_model_info", get_model_info)


@pytest.mark.parametrize(
    ("model_info", "expected"),
    [
        ({}, ["none", "low", "medium", "high"]),
        (Exception("not mapped"), ["none", "low", "medium", "high"]),
        (
            {"supports_low_reasoning_effort": False, "supports_minimal_reasoning_effort": False, "supports_xhigh_reasoning_effort": True},
            ["none", "medium", "high", "xhigh"],
        ),
        (
            {"supports_minimal_reasoning_effort": True, "supports_low_reasoning_effort": None},
            ["none", "minimal", "low", "medium", "high"],
        ),
        (
            {"supports_xhigh_reasoning_effort": True, "supports_max_reasoning_effort": True, "supports_none_reasoning_effort": False},
            ["none", "low", "medium", "high", "xhigh", "max"],
        ),
    ],
)
def test_levels_follow_litellm_effort_flags(monkeypatch: pytest.MonkeyPatch, model_info: dict | Exception, expected: list[str]) -> None:
    _stub_litellm(monkeypatch, supports_reasoning=True, model_info=model_info)
    assert _litellm_reasoning_levels("some/model") == expected


def test_non_reasoning_model_has_no_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_litellm(monkeypatch, supports_reasoning=False, model_info={"supports_xhigh_reasoning_effort": True})
    assert _litellm_reasoning_levels("some/model") == []


@pytest.mark.parametrize(
    ("omp_levels", "expected"),
    [
        (["low", "medium", "high", "xhigh", "max"], ["none", "low", "medium", "high", "xhigh", "max"]),
        ([], []),
        (None, ["none", "low", "medium", "high"]),
    ],
)
def test_omp_models_use_omp_registry_before_litellm(monkeypatch: pytest.MonkeyPatch, omp_levels: list[str] | None, expected: list[str]) -> None:
    """OMP's ladder wins, an empty one means no reasoning, an unknown model falls back to LiteLLM."""
    _stub_litellm(monkeypatch, supports_reasoning=True, model_info={})
    requested: list[str] = []

    def thinking_levels(model_id: str) -> list[str] | None:
        requested.append(model_id)
        return omp_levels

    monkeypatch.setattr(models.omp_source, "thinking_levels", thinking_levels)
    assert asyncio.run(get_reasoning_levels("litellm_proxy/openai-codex/gpt-6-astra")) == {"levels": expected}
    assert requested == ["openai-codex/gpt-6-astra"]
