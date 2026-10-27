"""sandbox_plot writes a script with the LLM, repairs it once, then hands the plot to the workspace agent."""

from types import SimpleNamespace
from typing import Any

import pytest

from config import TEST_MODEL
from lib import llm_resilience, tools
from lib.agent_sandbox_adapter import SandboxScriptError
from lib.sandbox_service import SandboxToolResult
from lib.tool_context import ToolContext, ToolOptions
from lib.toolcalling import ToolOutcome
from tests.fakes import ClientStub, SandboxServiceStub

PLOT_REQUEST = {"chat_id": "chat-1", "tool_call_id": "call-1", "user_msg": "Plot the trend"}


def _script_reply(script: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=script))])


async def _plot(sandbox: SandboxServiceStub, **context: Any) -> ToolOutcome:
    """Run sandbox_plot through the dispatcher against *sandbox*."""
    ctx = ToolContext(client=ClientStub(), model=TEST_MODEL, opts=ToolOptions(), sandbox_service=sandbox, **context)
    return await tools.BUILTIN_TOOLS.execute("sandbox_plot", {}, ctx)


@pytest.mark.asyncio
async def test_sandbox_plot_treats_an_unfenced_reply_as_the_whole_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that forgets the fence still plots, and its code still reaches the answer round."""
    sandbox = SandboxServiceStub("unused", script_stdout='{"data": [{"type": "bar"}], "layout": {}}')
    monkeypatch.setattr(llm_resilience, "api_query_resilient", lambda *_a, **_k: _script_reply("print(fig.to_json())"))

    outcome = await _plot(sandbox, **PLOT_REQUEST)

    assert sandbox.scripts == ["print(fig.to_json())"]
    assert "print(fig.to_json())" in outcome.content


@pytest.mark.asyncio
async def test_sandbox_plot_repairs_fast_script_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = SandboxServiceStub("unused")
    generation_calls: list[dict[str, Any]] = []
    replies = iter([_script_reply("```python\nbroken script\n```"), _script_reply("repaired script")])

    def reply(*_args: Any, **kwargs: Any) -> SimpleNamespace:
        generation_calls.append(kwargs)
        return next(replies)

    async def run_script(script: str, **_kwargs: Any) -> str:
        sandbox.scripts.append(script)
        if script == "broken script":
            raise SandboxScriptError("repairable script failure")
        return '{"data": [{"type": "bar"}], "layout": {}}'

    monkeypatch.setattr(llm_resilience, "api_query_resilient", reply)
    monkeypatch.setattr(sandbox, "run_script", run_script)

    outcome = await _plot(sandbox, **PLOT_REQUEST, small_model="provider/small-model")

    assert (outcome.summary, outcome.error) == ("1 trace", None)
    assert outcome.detail["script"] == "repaired script"
    assert "repaired script" in outcome.content  # The answer round sees the code that actually ran.
    assert sandbox.scripts == ["broken script", "repaired script"]
    assert sandbox.calls == []
    assert len(generation_calls) == 2
    assert generation_calls[0]["user_msg"] == "Plot the trend"
    assert "repairable script failure" in generation_calls[1]["user_msg"]
    assert [call["model"] for call in generation_calls] == [TEST_MODEL, TEST_MODEL]


@pytest.mark.asyncio
async def test_sandbox_plot_generation_exception_hands_off_without_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = SandboxServiceStub('{"data": [{"type": "scatter"}], "layout": {}}')
    generation_calls: list[dict[str, Any]] = []

    def unavailable(*_args: Any, **kwargs: Any) -> RuntimeError:
        generation_calls.append(kwargs)
        return RuntimeError("provider unavailable")

    monkeypatch.setattr(llm_resilience, "api_query_resilient", unavailable)

    outcome = await _plot(sandbox, **PLOT_REQUEST, small_model="provider/small-model")

    assert (outcome.summary, outcome.error) == ("1 trace", None)
    assert outcome.detail["script"] == ""
    assert sandbox.scripts == []
    assert len(generation_calls) == 1
    assert generation_calls[0]["model"] == TEST_MODEL
    assert [call["scope"] for call in sandbox.calls] == ["sandbox_plot"]


@pytest.mark.asyncio
async def test_sandbox_plot_falls_back_to_the_agent_when_the_script_keeps_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = SandboxServiceStub('{"data": [{"type": "scatter"}], "layout": {}}', script_stdout=None)
    monkeypatch.setattr(llm_resilience, "api_query_resilient", lambda *_args, **_kwargs: _script_reply("print(fig.to_json())"))

    outcome = await _plot(sandbox, **PLOT_REQUEST)

    assert outcome.error is None
    assert len(sandbox.scripts) == 2  # One repair attempt, then hand over.
    assert [call["scope"] for call in sandbox.calls] == ["sandbox_plot"]
    assert sandbox.calls[0]["prompt"] == "Plot the trend"
    append_system = sandbox.calls[0]["append_system"]
    assert "layered, skimmable Markdown explanation" in append_system
    assert "`$...$` inline and `$$...$$`" in append_system


@pytest.mark.asyncio
async def test_sandbox_plot_invalid_output_fails_without_exposing_transcript() -> None:
    sandbox = SandboxServiceStub("not json")

    outcome = await _plot(sandbox)

    assert (outcome.summary, outcome.error) == ("Failed", "invalid_figure")
    # The stub's transcript is what must not reach the model, so assert on that exact text.
    assert "hidden transcript" not in outcome.content


@pytest.mark.asyncio
async def test_sandbox_plot_runner_failure_reports_its_own_error_not_invalid_figure() -> None:
    """A sandbox that never ran (e.g. runtime failure) must not be reported as a bad figure."""
    sandbox = SandboxServiceStub(
        "not json",
        invoke_result=SandboxToolResult(
            status="failed",
            summary="The workspace agent could not complete this run.",
            manifest_id=None,
            workspace_changed=False,
            error="runner_error",
        ),
    )

    outcome = await _plot(sandbox)

    assert outcome.error == "runner_error"
    assert outcome.summary == "The workspace agent could not complete this run."
