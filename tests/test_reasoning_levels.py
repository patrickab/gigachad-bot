import asyncio

from fastapi import BackgroundTasks
import pytest

from backend.routes.models import get_reasoning_levels
from lib import reasoning_levels
from lib.reasoning_levels import litellm_levels, resolve


def _stub_litellm(monkeypatch: pytest.MonkeyPatch, supports_reasoning: bool, model_info: dict | Exception) -> None:
    """Replace LiteLLM's registry lookups with a fixed answer."""
    monkeypatch.setattr(reasoning_levels.litellm, "supports_reasoning", lambda model: supports_reasoning)

    def get_model_info(model: str) -> dict:
        if isinstance(model_info, Exception):
            raise model_info
        return model_info

    monkeypatch.setattr(reasoning_levels.litellm, "get_model_info", get_model_info)


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
    assert litellm_levels("some/model") == expected


def test_non_reasoning_model_has_no_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_litellm(monkeypatch, supports_reasoning=False, model_info={"supports_xhigh_reasoning_effort": True})
    assert litellm_levels("some/model") == []


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

    monkeypatch.setattr(reasoning_levels.omp_source, "thinking_levels", thinking_levels)
    assert asyncio.run(resolve("litellm_proxy/openai-codex/gpt-6-astra")) == expected
    assert requested == ["openai-codex/gpt-6-astra"]


@pytest.fixture
def stored_levels(monkeypatch: pytest.MonkeyPatch, postgres_pool, clean_database):
    """Route the levels table to the test database, emptied per test."""
    monkeypatch.setattr(reasoning_levels, "get_postgres_pool", lambda: postgres_pool)


def _stub_live_levels(monkeypatch: pytest.MonkeyPatch, live: dict[str, list[str] | None]) -> None:
    async def resolve_live(model: str) -> list[str] | None:
        return live[model]

    monkeypatch.setattr(reasoning_levels, "resolve", resolve_live)


def _request_levels(model: str) -> tuple[list[str], BackgroundTasks]:
    background_tasks = BackgroundTasks()
    response = asyncio.run(get_reasoning_levels(model, background_tasks))
    return response["levels"], background_tasks


def test_stored_levels_answer_first_and_revalidate_in_background(monkeypatch: pytest.MonkeyPatch, stored_levels) -> None:
    live = {"omp/model": ["none", "low", "medium"]}
    _stub_live_levels(monkeypatch, live)
    assert _request_levels("omp/model")[0] == ["none", "low", "medium"]

    live["omp/model"] = ["none", "low", "medium", "high"]
    levels, background_tasks = _request_levels("omp/model")
    assert levels == ["none", "low", "medium"]

    asyncio.run(background_tasks())
    assert _request_levels("omp/model")[0] == ["none", "low", "medium", "high"]


def test_failed_lookup_keeps_stored_levels(monkeypatch: pytest.MonkeyPatch, stored_levels) -> None:
    live: dict[str, list[str] | None] = {"ollama/thinker": None}
    _stub_live_levels(monkeypatch, live)
    assert _request_levels("ollama/thinker")[0] == []

    live["ollama/thinker"] = ["none", "low", "medium", "high"]
    assert _request_levels("ollama/thinker")[0] == ["none", "low", "medium", "high"]

    live["ollama/thinker"] = None
    levels, background_tasks = _request_levels("ollama/thinker")
    asyncio.run(background_tasks())
    assert levels == _request_levels("ollama/thinker")[0] == ["none", "low", "medium", "high"]
