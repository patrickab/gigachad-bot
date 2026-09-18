"""The tool loop is the only new chat primitive, so its event order is what gets tested."""

from types import SimpleNamespace
from typing import Any

import pytest

from lib import tools


class ClientStub:
    """Only the two client seams the loop borrows: routing and payload construction."""

    def _resolve_routing(self, model: str) -> tuple[str, None, None]:
        return model, None, None

    def _construct_message_payload(
        self,
        user_msg: str | None = None,
        user_msg_history: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        img: Any = None,  # noqa: ARG002 - mirrors the client signature
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(user_msg_history or [])
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        return messages


def _text_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))], usage=None)


def _tool_chunk(index: int, call_id: str, name: str | None, arguments: str) -> SimpleNamespace:
    call = SimpleNamespace(index=index, id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[call]))], usage=None)


def _usage_chunk(total: int) -> SimpleNamespace:
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=total, completion_tokens=0, total_tokens=total), choices=[])


async def _collect(**overrides: Any) -> list[tuple[str, Any]]:
    kwargs: dict[str, Any] = {
        "client": ClientStub(),
        "model": "openai/gpt-4o",
        "user_msg": "who won?",
        "history": [],
        "system_prompt": "be terse",
        "enabled": ["web_search"],
        "opts": tools.ToolOptions(),
    }
    kwargs.update(overrides)
    return [event async for event in tools.stream_chat_with_tools(**kwargs)]


@pytest.mark.asyncio
async def test_tool_free_turn_streams_tokens_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """With tools offered but unused, the turn is an ordinary streamed answer."""
    monkeypatch.setattr(
        tools.litellm, "completion", lambda **_: iter([_text_chunk("no "), _text_chunk("tools"), _usage_chunk(7)])
    )

    usage = {"prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7}
    assert await _collect() == [("token", "no "), ("token", "tools"), ("usage", usage)]


@pytest.mark.asyncio
async def test_tool_call_is_executed_and_answer_follows(monkeypatch: pytest.MonkeyPatch) -> None:
    """A streamed tool call surfaces as its own events, then the model answers with the result."""
    rounds = [
        # Arguments arrive fragmented, which is how providers actually stream them.
        iter([_tool_chunk(0, "call_a", "web_search", '{"query": "elect'), _tool_chunk(0, "", None, 'ion"}')]),
        iter([_text_chunk("Answer [ap]")]),
    ]
    sent: list[list[dict[str, Any]]] = []

    def fake_completion(**kwargs: Any) -> Any:
        sent.append(kwargs["messages"])
        return rounds.pop(0)

    async def fake_execute(name: str, args: dict[str, Any], _ctx: Any) -> tools.ToolOutcome:
        assert (name, args) == ("web_search", {"query": "election"})
        sources = [{"label": "ap", "url": "u", "content": "drop"}]
        return tools.ToolOutcome(content="[ap] evidence", summary="1 sources", sources=sources)

    monkeypatch.setattr(tools.litellm, "completion", fake_completion)
    monkeypatch.setattr(tools, "execute_tool", fake_execute)

    events = await _collect()

    assert [name for name, _ in events] == ["tool_call", "tool_result", "token"]
    assert events[0][1] == {"id": "call_a", "name": "web_search", "arguments": {"query": "election"}}
    assert events[1][1]["summary"] == "1 sources"
    # Source bodies are evidence for the model, not payload for the browser.
    assert events[1][1]["sources"] == [{"label": "ap", "url": "u"}]
    assert events[2] == ("token", "Answer [ap]")

    # Round two must carry the assistant tool_calls turn and its tool result.
    assert [m["role"] for m in sent[1]] == ["system", "user", "assistant", "tool"]
    assert sent[1][3] == {"role": "tool", "tool_call_id": "call_a", "content": "[ap] evidence"}


@pytest.mark.asyncio
async def test_exactly_one_tool_call_per_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that keeps calling tools gets exactly one, then a round with no tools offered."""
    offered: list[Any] = []

    def fake_completion(**kwargs: Any) -> Any:
        offered.append(kwargs.get("tools"))
        return iter([_tool_chunk(0, f"call_{len(offered)}", "web_search", '{"query": "x"}')])

    async def fake_execute(*_: Any, **__: Any) -> tools.ToolOutcome:
        return tools.ToolOutcome(content="evidence")

    monkeypatch.setattr(tools.litellm, "completion", fake_completion)
    monkeypatch.setattr(tools, "execute_tool", fake_execute)

    events = await _collect()

    assert len(offered) == 2
    assert offered[1] is None  # the answer round is never handed tools again
    assert [name for name, _ in events].count("tool_call") == 1


@pytest.mark.asyncio
async def test_tool_call_written_as_plain_text_is_recovered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Models without native tool calling (Ollama gemma) emit the call as content.

    That JSON must become a real tool call and must never reach the user as tokens.
    """
    rounds = [
        iter([_text_chunk('{"name": "web_search", '), _text_chunk('"arguments": {"query": "cheesecake"}}')]),
        iter([_text_chunk("Here you go")]),
    ]
    monkeypatch.setattr(tools.litellm, "completion", lambda **_: rounds.pop(0))

    async def fake_execute(name: str, args: dict[str, Any], _ctx: Any) -> tools.ToolOutcome:
        assert (name, args) == ("web_search", {"query": "cheesecake"})
        return tools.ToolOutcome(content="evidence", summary="3 sources")

    monkeypatch.setattr(tools, "execute_tool", fake_execute)

    events = await _collect()

    assert [name for name, _ in events] == ["tool_call", "tool_result", "token"]
    assert events[0][1]["arguments"] == {"query": "cheesecake"}
    assert events[2] == ("token", "Here you go")


@pytest.mark.asyncio
async def test_held_json_that_is_not_a_tool_call_is_still_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Buffering JSON-looking text must not swallow an answer that genuinely is JSON."""
    monkeypatch.setattr(tools.litellm, "completion", lambda **_: iter([_text_chunk('{"shape": "answer"}')]))

    assert await _collect() == [("token", '{"shape": "answer"}')]


@pytest.mark.asyncio
async def test_unknown_tool_names_are_never_offered(monkeypatch: pytest.MonkeyPatch) -> None:
    """The browser cannot widen the tool surface by naming something that is not registered."""
    offered: list[Any] = []

    def fake_completion(**kwargs: Any) -> Any:
        offered.append(kwargs.get("tools"))
        return iter([_text_chunk("hi")])

    monkeypatch.setattr(tools.litellm, "completion", fake_completion)

    await _collect(enabled=["rm_rf", "deep_research"])

    assert [spec["function"]["name"] for spec in offered[0]] == ["deep_research"]


@pytest.mark.asyncio
async def test_failed_tool_degrades_the_turn_instead_of_the_stream() -> None:
    """Argument validation and unknown names are contained by the dispatcher, not each tool."""
    ctx = tools.ToolContext(client=ClientStub(), model="m", opts=tools.ToolOptions())

    outcome = await tools.execute_tool("web_search", {"query": " "}, ctx)
    assert outcome.error == "Missing argument: `query`."

    outcome = await tools.execute_tool("nope", {"query": "q"}, ctx)
    assert outcome.error == "Unknown tool `nope`."

    async def boom(_args: dict[str, Any], _ctx: tools.ToolContext) -> tools.ToolOutcome:
        raise RuntimeError("brave is down")

    tools.REGISTRY["explode"] = tools.Tool("explode", "d", {"type": "object", "properties": {}}, boom)
    try:
        outcome = await tools.execute_tool("explode", {}, ctx)
    finally:
        del tools.REGISTRY["explode"]
    assert (outcome.error, outcome.summary) == ("brave is down", "Failed")


def _chat_app(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The real /api/chat router, with only the provider call and identity faked out."""
    from fastapi import FastAPI

    from backend.routes import chat as chat_route

    monkeypatch.setattr(chat_route, "request_client", lambda: _client_ctx())
    app = FastAPI()
    app.include_router(chat_route.router)
    app.dependency_overrides[chat_route.get_memory_store] = lambda: SimpleNamespace(
        augment_system_prompt=lambda prompt, slug: prompt  # noqa: ARG005 - signature parity
    )
    app.dependency_overrides[chat_route.get_asset_store] = lambda: None
    return app


class _client_ctx:  # noqa: N801 - stands in for the request_client contextmanager
    def __enter__(self) -> ClientStub:
        return ClientStub()

    def __exit__(self, *_: Any) -> None:
        return None


def _sse_events(body: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for block in body.replace("\r\n", "\n").strip().split("\n\n"):
        name, data = "message", []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[6:] if line.startswith("data: ") else line[5:])
        events.append((name, "\n".join(data)))
    return events


def test_chat_route_streams_tool_events_over_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end over the wire: a tool call reaches the browser as its own SSE events."""
    from fastapi.testclient import TestClient

    rounds = [
        iter([_tool_chunk(0, "call_a", "web_search", '{"query": "cheesecake"}')]),
        iter([_text_chunk("Try [ap]")]),
    ]
    monkeypatch.setattr(tools.litellm, "completion", lambda **_: rounds.pop(0))

    async def fake_execute(*_: Any, **__: Any) -> tools.ToolOutcome:
        return tools.ToolOutcome(content="[ap] evidence", summary="1 sources", sources=[{"label": "ap", "url": "u"}])

    monkeypatch.setattr(tools, "execute_tool", fake_execute)

    from backend.routes import chat as chat_route

    monkeypatch.setattr(chat_route, "_resolve_images", lambda *_, **__: None)

    client = TestClient(_chat_app(monkeypatch))
    response = client.post(
        "/api/chat",
        json={
            "model": "openai/gpt-4o",
            "chat_id": "c1",
            "user_msg": "search the web for cheesecake recipes",
            "tools": ["web_search"],
        },
    )

    assert response.status_code == 200, response.text
    events = _sse_events(response.text)
    assert [name for name, _ in events] == ["tool_call", "tool_result", "token", "done"]
    assert '"name": "web_search"' in events[0][1]
    assert events[2][1] == "Try [ap]"
