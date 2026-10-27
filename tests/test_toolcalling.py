"""The tool loop is the only new chat primitive, so its event order is what gets tested."""

from typing import Any

import pytest

from config import TEST_MODEL
from lib import toolcalling, tools
from lib.tool_context import ToolContext, ToolOptions
from tests.fakes import ClientStub, collect_tool_turn, text_chunk, tool_chunk, usage_chunk


@pytest.mark.asyncio
async def test_tool_free_turn_streams_tokens_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """With tools offered but unused, the turn is an ordinary streamed answer."""
    monkeypatch.setattr(
        toolcalling.litellm,
        "completion",
        lambda **_: iter([text_chunk("no "), text_chunk("tools"), usage_chunk(7)]),
    )

    usage = {"prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7}
    assert await collect_tool_turn() == [("token", "no "), ("token", "tools"), ("usage", usage)]


@pytest.mark.asyncio
async def test_tool_call_keeps_ui_detail_out_of_the_answer_round(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep model content and browser-only metadata on separate projections."""
    rounds = [
        # Arguments arrive fragmented, which is how providers actually stream them.
        iter([tool_chunk(0, "call_a", "web_search", '{"query": "elect'), tool_chunk(0, "", None, 'ion"}')]),
        iter([text_chunk("Answer [ap]")]),
    ]
    sent: list[list[dict[str, Any]]] = []
    sentinel_script = "UI_SCRIPT_SENTINEL"
    source_evidence = "SOURCE_EVIDENCE_SENTINEL"

    def fake_completion(**kwargs: Any) -> Any:
        sent.append(kwargs["messages"])
        return rounds.pop(0)

    async def fake_execute(name: str, args: dict[str, Any], _ctx: Any, **_kwargs: Any) -> toolcalling.ToolOutcome:
        assert (name, args) == ("web_search", {"query": "election"})
        sources = [{"label": "ap", "url": "u", "content": source_evidence}]
        return toolcalling.ToolOutcome(
            content="[ap] evidence",
            summary="1 sources",
            sources=sources,
            detail={"script": sentinel_script},
        )

    monkeypatch.setattr(toolcalling.litellm, "completion", fake_completion)
    monkeypatch.setattr(tools.BUILTIN_TOOLS, "execute", fake_execute)

    events = await collect_tool_turn()

    assert [name for name, _ in events] == ["tool_call", "tool_progress", "tool_progress", "tool_result", "token"]
    assert events[0][1] == {"id": "call_a", "name": "web_search", "arguments": {"query": "election"}}
    assert events[1][1]["stages"][0]["status"] == "running"
    assert events[2][1]["stages"][0]["status"] == "done"
    assert events[3][1]["summary"] == "1 sources"
    assert events[3][1]["detail"] == {"script": sentinel_script}
    assert events[3][1]["sources"] == [{"label": "ap", "url": "u"}]
    assert events[4] == ("token", "Answer [ap]")

    # Round two must carry the assistant tool_calls turn and its tool result.
    assert [m["role"] for m in sent[1]] == ["system", "user", "assistant", "tool"]
    assert sent[1][3] == {"role": "tool", "tool_call_id": "call_a", "content": "[ap] evidence"}
    assert sentinel_script not in str(sent[1])
    assert source_evidence not in str(sent[1])


@pytest.mark.asyncio
async def test_mindmap_tool_result_does_not_trigger_an_assistant_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rendered tool card is the complete mind map response."""
    rounds = [iter([tool_chunk(0, "call_mindmap", "mindmap", "{}")])]
    monkeypatch.setattr(toolcalling.litellm, "completion", lambda **_: rounds.pop(0))

    async def fake_execute(*_: Any, **__: Any) -> toolcalling.ToolOutcome:
        return toolcalling.ToolOutcome(content="```markmap\n# Topic\n```", summary="Mind map ready")

    monkeypatch.setattr(tools.BUILTIN_TOOLS, "execute", fake_execute)

    events = await collect_tool_turn(enabled=["mindmap"])

    assert [name for name, _ in events] == ["tool_call", "tool_progress", "tool_progress", "tool_result"]
    assert rounds == []


@pytest.mark.asyncio
async def test_exactly_one_tool_call_per_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that keeps calling tools gets exactly one, then a round with no tools offered."""
    offered: list[Any] = []

    def fake_completion(**kwargs: Any) -> Any:
        offered.append(kwargs.get("tools"))
        return iter([tool_chunk(0, f"call_{len(offered)}", "web_search", '{"query": "x"}')])

    async def fake_execute(*_: Any, **__: Any) -> toolcalling.ToolOutcome:
        return toolcalling.ToolOutcome(content="evidence")

    monkeypatch.setattr(toolcalling.litellm, "completion", fake_completion)
    monkeypatch.setattr(tools.BUILTIN_TOOLS, "execute", fake_execute)

    events = await collect_tool_turn()

    assert len(offered) == 2
    assert offered[1] is None  # the answer round is never handed tools again
    assert [name for name, _ in events].count("tool_call") == 1


@pytest.mark.asyncio
async def test_tool_call_written_as_plain_text_is_recovered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Models without native tool calling (Ollama gemma) emit the call as content.

    That JSON must become a real tool call and must never reach the user as tokens.
    """
    rounds = [
        iter([text_chunk('{"name": "web_search", '), text_chunk('"arguments": {"query": "cheesecake"}}')]),
        iter([text_chunk("Here you go")]),
    ]
    monkeypatch.setattr(toolcalling.litellm, "completion", lambda **_: rounds.pop(0))

    async def fake_execute(name: str, args: dict[str, Any], _ctx: Any, **_kwargs: Any) -> toolcalling.ToolOutcome:
        assert (name, args) == ("web_search", {"query": "cheesecake"})
        return toolcalling.ToolOutcome(content="evidence", summary="3 sources")

    monkeypatch.setattr(tools.BUILTIN_TOOLS, "execute", fake_execute)

    events = await collect_tool_turn()

    assert [name for name, _ in events] == ["tool_call", "tool_progress", "tool_progress", "tool_result", "token"]
    assert events[0][1]["arguments"] == {"query": "cheesecake"}


@pytest.mark.asyncio
async def test_held_json_that_is_not_a_tool_call_is_still_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Buffering JSON-looking text must not swallow an answer that genuinely is JSON."""
    monkeypatch.setattr(toolcalling.litellm, "completion", lambda **_: iter([text_chunk('{"shape": "answer"}')]))

    assert await collect_tool_turn() == [("token", '{"shape": "answer"}')]


@pytest.mark.asyncio
async def test_unknown_tool_names_are_never_offered(monkeypatch: pytest.MonkeyPatch) -> None:
    """The browser cannot widen the tool surface by naming something that is not registered."""
    offered: list[Any] = []

    def fake_completion(**kwargs: Any) -> Any:
        offered.append(kwargs.get("tools"))
        return iter([text_chunk("hi")])

    monkeypatch.setattr(toolcalling.litellm, "completion", fake_completion)

    await collect_tool_turn(enabled=["rm_rf", "deep_research"])

    assert [spec["function"]["name"] for spec in offered[0]] == ["deep_research"]


@pytest.mark.asyncio
async def test_failed_tool_degrades_the_turn_instead_of_the_stream() -> None:
    """Argument validation and unknown names are contained by the dispatcher, not each tool."""
    ctx = ToolContext(client=ClientStub(), model=TEST_MODEL, opts=ToolOptions())

    outcome = await tools.BUILTIN_TOOLS.execute("web_search", {"query": " "}, ctx)
    assert outcome.error == "Missing argument: `query`."

    outcome = await tools.BUILTIN_TOOLS.execute("nope", {"query": "q"}, ctx)
    assert outcome.error == "Unknown tool `nope`."

    async def boom(_args: dict[str, Any], _ctx: ToolContext, _stage: toolcalling.Stage) -> toolcalling.ToolOutcome:
        raise RuntimeError("brave is down")

    catalog = toolcalling.ToolCatalog([toolcalling.ToolDefinition("explode", "d", {"type": "object", "properties": {}}, boom)])
    outcome = await catalog.execute("explode", {}, ctx)
    assert (outcome.error, outcome.summary) == ("brave is down", "Failed")


@pytest.mark.asyncio
async def test_a_tool_the_turn_did_not_enable_is_refused_without_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registration is not authorisation: a model naming an off tool must not execute it."""
    ran: list[str] = []

    async def record(_args: dict[str, Any], _ctx: Any, _stage: toolcalling.Stage) -> toolcalling.ToolOutcome:
        ran.append("ran")
        return toolcalling.ToolOutcome(content="evidence")

    schema = {"type": "object", "additionalProperties": False, "properties": {}}
    catalog = toolcalling.ToolCatalog(
        [
            toolcalling.ToolDefinition("web_search", "d", schema, record),
            toolcalling.ToolDefinition("workspace_agent", "d", schema, record),
        ]
    )
    rounds = [iter([tool_chunk(0, "call_a", "workspace_agent", "{}")]), iter([text_chunk("cannot do that")])]
    monkeypatch.setattr(toolcalling.litellm, "completion", lambda **_: rounds.pop(0))

    events = [
        event
        async for event in toolcalling.stream_tool_turn(
            client=ClientStub(),
            model=TEST_MODEL,
            user_msg="run code",
            history=[],
            system_prompt="be terse",
            enabled=["web_search"],
            catalog=catalog,
            context_for_call=lambda _id: None,
            guidance="guidance",
        )
    ]

    (result,) = [data for name, data in events if name == "tool_result"]
    assert ran == []
    assert result["error"] == "Tool `workspace_agent` was not enabled for this turn."
    assert events[-1] == ("token", "cannot do that")


@pytest.mark.asyncio
async def test_arguments_outside_the_declared_schema_are_refused() -> None:
    """Declared shapes are enforced by the dispatcher, so malformed calls degrade the turn."""
    ctx = ToolContext(client=ClientStub(), model=TEST_MODEL, opts=ToolOptions())

    outcome = await tools.BUILTIN_TOOLS.execute("web_search", {"query": "q", "site": "example.com"}, ctx)
    assert (outcome.error, outcome.summary) == ("Unknown argument: `site`.", "Rejected")

    outcome = await tools.BUILTIN_TOOLS.execute("sandbox_plot", {"code": "x"}, ctx)
    assert (outcome.error, outcome.summary) == ("Unknown argument: `code`.", "Rejected")

    # No declared properties means no arguments to reject: the call reaches the handler.
    outcome = await tools.BUILTIN_TOOLS.execute("sandbox_plot", {}, ctx)
    assert (outcome.summary, outcome.error) == ("Unavailable", "sandbox_service_unavailable")

    outcome = await tools.BUILTIN_TOOLS.execute("web_search", {"query": {"nested": "object"}}, ctx)
    assert outcome.error == "Invalid argument: `query` must be a string."
    assert outcome.content.startswith("Tool call rejected:")
