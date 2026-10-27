"""Built-in tools: web search labels, the workspace agent and the LLM-rendered mind map, diagram and OCR cards."""

from types import SimpleNamespace
from typing import Any

import pytest

from config import TEST_MODEL
from lib import llm_resilience, toolcalling, tools, web_search
from lib.tool_context import ToolContext, ToolOptions
from tests.fakes import ClientStub, SandboxServiceStub, collect_tool_turn, text_chunk, tool_chunk


def test_source_label_extracts_a_stable_hostname_label() -> None:
    assert web_search.source_label("https://www.example.com/article", set()) == "example"


@pytest.mark.asyncio
async def test_workspace_agent_result_carries_sandbox_state_to_the_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only sandbox-backed tools attach a `sandbox` payload; ordinary tools leave it null."""
    rounds = [
        iter([tool_chunk(0, "call_1", "workspace_agent", '{"prompt": "write a script"}')]),
        iter([text_chunk("OK")]),
    ]
    monkeypatch.setattr(toolcalling.litellm, "completion", lambda **_: rounds.pop(0))

    events = await collect_tool_turn(enabled=["workspace_agent"], chat_id="chat-1", sandbox_service=SandboxServiceStub("unused"))
    (result,) = [data for name, data in events if name == "tool_result"]

    assert result["name"] == "workspace_agent"
    assert result["sandbox"]["status"] == "completed"
    assert result["sandbox"]["workspace_changed"] is True

    rounds = [
        iter([tool_chunk(0, "call_2", "web_search", '{"query": "x"}')]),
        iter([text_chunk("OK")]),
    ]

    async def fake_execute(*_: Any, **__: Any) -> toolcalling.ToolOutcome:
        return toolcalling.ToolOutcome(content="evidence", summary="1 sources")

    monkeypatch.setattr(tools.BUILTIN_TOOLS, "execute", fake_execute)

    events = await collect_tool_turn(enabled=["web_search"], chat_id="chat-1", sandbox_service=SandboxServiceStub("unused"))
    (result,) = [data for name, data in events if name == "tool_result"]

    assert result["sandbox"] is None


@pytest.mark.asyncio
async def test_workspace_agent_without_a_sandbox_service_degrades_instead_of_raising() -> None:
    """A missing sandbox must become a tool error the model can talk about, never a dead stream."""
    ctx = ToolContext(client=ClientStub(), model=TEST_MODEL, opts=ToolOptions(), chat_id="chat-1", sandbox_service=None)

    outcome = await tools.BUILTIN_TOOLS.execute("workspace_agent", {"prompt": "do work"}, ctx)

    assert outcome.error is not None
    assert outcome.content


@pytest.mark.asyncio
async def test_mindmap_tool_returns_a_renderable_markmap(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_query(_client: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="```markmap\n# Topic\n```"))])

    monkeypatch.setattr(llm_resilience, "api_query_resilient", fake_query)
    context = ToolContext(
        client=ClientStub(),
        model=TEST_MODEL,
        opts=ToolOptions(),
        history=({"role": "assistant", "content": "Prior explanation"},),
        user_msg="Make a mind map",
    )

    outcome = await tools.BUILTIN_TOOLS.execute("mindmap", {}, context)

    assert outcome.detail["mindmap"] == "```markmap\n# Topic\n```"
    assert "<transcript>" in calls[0]["user_msg"]
    assert calls[0]["stream"] is False


@pytest.mark.asyncio
async def test_diagram_tool_returns_mermaid_source_for_the_tool_card(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_query(_client: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="```diagram\nflowchart LR\nA --> B\n```"))])

    monkeypatch.setattr(llm_resilience, "api_query_resilient", fake_query)
    context = ToolContext(
        client=ClientStub(),
        model=TEST_MODEL,
        opts=ToolOptions(),
        history=({"role": "assistant", "content": "Prior explanation"},),
        user_msg="Make a diagram",
    )

    outcome = await tools.BUILTIN_TOOLS.execute("diagram", {}, context)

    assert outcome.detail["mermaid"] == "flowchart LR\nA --> B"
    assert "Do not create, return, or repeat Mermaid or any other diagram." in outcome.content
    assert "extremely concise companion explanation" not in outcome.content
    assert "skimmable Markdown layout" not in outcome.content
    assert "```mermaid\nflowchart LR\nA --> B\n```" in outcome.content
    assert "<transcript>" in calls[0]["user_msg"]
    assert calls[0]["system_prompt"] == tools.SYS_DIAGRAM_MERMAID
    assert "significantly reduce the text needed" in tools.BUILTIN_TOOLS.specs(["diagram"])[0]["function"]["description"]


@pytest.mark.asyncio
async def test_diagram_answer_round_keeps_main_system_prompt_and_rendered_source(monkeypatch: pytest.MonkeyPatch) -> None:
    rounds = [
        iter([tool_chunk(0, "diagram_a", "diagram", "{}")]),
        iter([text_chunk("The explanation appears below the diagram.")]),
    ]
    sent: list[list[dict[str, Any]]] = []

    def fake_completion(**kwargs: Any) -> Any:
        sent.append(kwargs["messages"])
        return rounds.pop(0)

    def fake_query(_client: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="```mermaid\nflowchart LR\nA --> B\n```"))])

    monkeypatch.setattr(toolcalling.litellm, "completion", fake_completion)
    monkeypatch.setattr(llm_resilience, "api_query_resilient", fake_query)

    events = await collect_tool_turn(
        user_msg="Show the request lifecycle",
        system_prompt="Follow the main system prompt.",
        enabled=["diagram"],
    )

    assert [name for name, _ in events][-2:] == ["tool_result", "token"]
    assert events[-1] == ("token", "The explanation appears below the diagram.")
    assert "Follow the main system prompt." in sent[1][0]["content"]
    assert "```mermaid\nflowchart LR\nA --> B\n```" in sent[1][3]["content"]
    assert "Do not create, return, or repeat Mermaid" in sent[1][3]["content"]


@pytest.mark.asyncio
async def test_latex_ocr_tool_uses_the_vision_model_for_attached_images(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_query(_client: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="$a^2 + b^2 = c^2$"))])

    monkeypatch.setattr(llm_resilience, "api_query_resilient", fake_query)
    context = ToolContext(
        client=ClientStub(),
        model=TEST_MODEL,
        vision_model="provider/vision",
        opts=ToolOptions(),
        img=b"image-bytes",
    )

    outcome = await tools.BUILTIN_TOOLS.execute("latex_ocr", {}, context)

    assert outcome.detail["text"] == "$a^2 + b^2 = c^2$"
    assert calls[0]["model"] == "provider/vision"
    assert calls[0]["img"] == b"image-bytes"
