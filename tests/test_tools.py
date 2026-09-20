"""The tool loop is the only new chat primitive, so its event order is what gets tested."""

from types import SimpleNamespace
from typing import Any

from agent_sandbox import PromptImage
import pytest

from config import MODEL_DEFAULT_KEYS, TEST_MODEL
from lib import sandbox_plot, toolcalling, tools, web_search
from lib.agent_sandbox_adapter import SandboxScriptError
from lib.sandbox_service import SandboxToolResult


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


def test_source_label_extracts_a_stable_hostname_label() -> None:
    assert web_search.source_label("https://www.example.com/article", set()) == "example"


class SandboxServiceStub:
    def __init__(self, stdout: str, script_stdout: str | None = None) -> None:
        self.model = TEST_MODEL
        self.stdout = stdout
        self.script_stdout = script_stdout
        self.calls: list[dict[str, Any]] = []
        self.scripts: list[str] = []

    async def invoke(self, **kwargs: Any) -> SandboxToolResult:
        self.calls.append(kwargs)
        return SandboxToolResult("completed", "hidden transcript", "manifest", True)

    async def run_script(self, script: str, **_kwargs: Any) -> str:
        self.scripts.append(script)
        if self.script_stdout is None:
            raise SandboxScriptError("ModuleNotFoundError: no such module")
        return self.script_stdout

    def output_texts(self, _result: SandboxToolResult, *, media_type: str = "text/plain") -> tuple[str, ...]:
        assert media_type == sandbox_plot.PLOTLY_MEDIA_TYPE
        return (self.stdout,)


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
        "model": TEST_MODEL,
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
        toolcalling.litellm,
        "completion",
        lambda **_: iter([_text_chunk("no "), _text_chunk("tools"), _usage_chunk(7)]),
    )

    usage = {"prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7}
    assert await _collect() == [("token", "no "), ("token", "tools"), ("usage", usage)]


@pytest.mark.asyncio
async def test_tool_call_keeps_ui_detail_out_of_the_answer_round(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep model content and browser-only metadata on separate projections."""
    rounds = [
        # Arguments arrive fragmented, which is how providers actually stream them.
        iter([_tool_chunk(0, "call_a", "web_search", '{"query": "elect'), _tool_chunk(0, "", None, 'ion"}')]),
        iter([_text_chunk("Answer [ap]")]),
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

    events = await _collect()

    assert [name for name, _ in events] == ["tool_call", "tool_result", "token"]
    assert events[0][1] == {"id": "call_a", "name": "web_search", "arguments": {"query": "election"}}
    assert events[1][1]["summary"] == "1 sources"
    assert events[1][1]["detail"] == {"script": sentinel_script}
    assert events[1][1]["sources"] == [{"label": "ap", "url": "u"}]
    assert events[2] == ("token", "Answer [ap]")

    # Round two must carry the assistant tool_calls turn and its tool result.
    assert [m["role"] for m in sent[1]] == ["system", "user", "assistant", "tool"]
    assert sent[1][3] == {"role": "tool", "tool_call_id": "call_a", "content": "[ap] evidence"}
    assert sentinel_script not in str(sent[1])
    assert source_evidence not in str(sent[1])


@pytest.mark.asyncio
async def test_exactly_one_tool_call_per_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that keeps calling tools gets exactly one, then a round with no tools offered."""
    offered: list[Any] = []

    def fake_completion(**kwargs: Any) -> Any:
        offered.append(kwargs.get("tools"))
        return iter([_tool_chunk(0, f"call_{len(offered)}", "web_search", '{"query": "x"}')])

    async def fake_execute(*_: Any, **__: Any) -> toolcalling.ToolOutcome:
        return toolcalling.ToolOutcome(content="evidence")

    monkeypatch.setattr(toolcalling.litellm, "completion", fake_completion)
    monkeypatch.setattr(tools.BUILTIN_TOOLS, "execute", fake_execute)

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
    monkeypatch.setattr(toolcalling.litellm, "completion", lambda **_: rounds.pop(0))

    async def fake_execute(name: str, args: dict[str, Any], _ctx: Any, **_kwargs: Any) -> toolcalling.ToolOutcome:
        assert (name, args) == ("web_search", {"query": "cheesecake"})
        return toolcalling.ToolOutcome(content="evidence", summary="3 sources")

    monkeypatch.setattr(tools.BUILTIN_TOOLS, "execute", fake_execute)

    events = await _collect()

    assert [name for name, _ in events] == ["tool_call", "tool_result", "token"]
    assert events[0][1]["arguments"] == {"query": "cheesecake"}
    assert events[2] == ("token", "Here you go")


@pytest.mark.asyncio
async def test_held_json_that_is_not_a_tool_call_is_still_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Buffering JSON-looking text must not swallow an answer that genuinely is JSON."""
    monkeypatch.setattr(toolcalling.litellm, "completion", lambda **_: iter([_text_chunk('{"shape": "answer"}')]))

    assert await _collect() == [("token", '{"shape": "answer"}')]


@pytest.mark.asyncio
async def test_unknown_tool_names_are_never_offered(monkeypatch: pytest.MonkeyPatch) -> None:
    """The browser cannot widen the tool surface by naming something that is not registered."""
    offered: list[Any] = []

    def fake_completion(**kwargs: Any) -> Any:
        offered.append(kwargs.get("tools"))
        return iter([_text_chunk("hi")])

    monkeypatch.setattr(toolcalling.litellm, "completion", fake_completion)

    await _collect(enabled=["rm_rf", "deep_research"])

    assert [spec["function"]["name"] for spec in offered[0]] == ["deep_research"]


@pytest.mark.asyncio
async def test_failed_tool_degrades_the_turn_instead_of_the_stream() -> None:
    """Argument validation and unknown names are contained by the dispatcher, not each tool."""
    ctx = tools.ToolContext(client=ClientStub(), model=TEST_MODEL, opts=tools.ToolOptions())

    outcome = await tools.BUILTIN_TOOLS.execute("web_search", {"query": " "}, ctx)
    assert outcome.error == "Missing argument: `query`."

    outcome = await tools.BUILTIN_TOOLS.execute("nope", {"query": "q"}, ctx)
    assert outcome.error == "Unknown tool `nope`."

    async def boom(_args: dict[str, Any], _ctx: tools.ToolContext) -> toolcalling.ToolOutcome:
        raise RuntimeError("brave is down")

    catalog = toolcalling.ToolCatalog(
        [toolcalling.ToolDefinition("explode", "d", {"type": "object", "properties": {}}, boom)]
    )
    outcome = await catalog.execute("explode", {}, ctx)
    assert (outcome.error, outcome.summary) == ("brave is down", "Failed")


@pytest.mark.asyncio
async def test_a_tool_the_turn_did_not_enable_is_refused_without_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registration is not authorisation: a model naming an off tool must not execute it."""
    ran: list[str] = []

    async def record(_args: dict[str, Any], _ctx: Any) -> toolcalling.ToolOutcome:
        ran.append("ran")
        return toolcalling.ToolOutcome(content="evidence")

    schema = {"type": "object", "additionalProperties": False, "properties": {}}
    catalog = toolcalling.ToolCatalog(
        [
            toolcalling.ToolDefinition("web_search", "d", schema, record),
            toolcalling.ToolDefinition("workspace_agent", "d", schema, record),
        ]
    )
    rounds = [iter([_tool_chunk(0, "call_a", "workspace_agent", "{}")]), iter([_text_chunk("cannot do that")])]
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
    ctx = tools.ToolContext(client=ClientStub(), model=TEST_MODEL, opts=tools.ToolOptions())

    outcome = await tools.BUILTIN_TOOLS.execute("web_search", {"query": "q", "site": "example.com"}, ctx)
    assert (outcome.error, outcome.summary) == ("Unknown argument: `site`.", "Rejected")

    outcome = await tools.BUILTIN_TOOLS.execute("sandbox_plot", {"brief": "x" * 12001}, ctx)
    assert (outcome.error, outcome.summary) == ("Argument too long: `brief`.", "Rejected")

    outcome = await tools.BUILTIN_TOOLS.execute("web_search", {"query": {"nested": "object"}}, ctx)
    assert outcome.error == "Invalid argument: `query` must be a string."
    assert outcome.content.startswith("Tool call rejected:")


def _script_reply(script: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=script))])


@pytest.mark.asyncio
async def test_sandbox_plot_fast_path_runs_one_generated_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """A self-contained chart needs one model call and one disposable execution, not a coding agent."""
    sandbox = SandboxServiceStub("unused", script_stdout='{"data": [{"type": "bar"}], "layout": {}}')
    fenced = "```python\nprint(fig.to_json())\n```"
    seen: list[str] = []

    def reply(*_args: Any, **kwargs: Any) -> SimpleNamespace:
        seen.append(kwargs["model"])
        return _script_reply(fenced)

    monkeypatch.setattr(sandbox_plot, "api_query_resilient", reply)
    ctx = tools.ToolContext(
        client=ClientStub(),
        model=TEST_MODEL,
        opts=tools.ToolOptions(),
        chat_id="chat-1",
        tool_call_id="call-1",
        sandbox_service=sandbox,
        small_model="provider/small-model",
    )

    outcome = await tools.BUILTIN_TOOLS.execute("sandbox_plot", {"brief": "Plot the trend"}, ctx)

    assert outcome.summary == "1 trace"
    assert outcome.detail == {
        "figure": {"data": [{"type": "bar"}], "layout": {}},
        "brief": "Plot the trend",
        "script": "print(fig.to_json())",
    }
    assert sandbox.scripts == ["print(fig.to_json())"]
    assert sandbox.calls == []  # The agent path stayed unused.
    assert seen == ["provider/small-model"]  # The user's small / fast default writes the script.


@pytest.mark.asyncio
async def test_sandbox_plot_repairs_fast_script_once_before_using_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = SandboxServiceStub("unused")
    generation_calls: list[dict[str, Any]] = []
    replies = iter([_script_reply("broken script"), _script_reply("repaired script")])

    def reply(*_args: Any, **kwargs: Any) -> SimpleNamespace:
        generation_calls.append(kwargs)
        return next(replies)

    async def run_script(script: str, **_kwargs: Any) -> str:
        sandbox.scripts.append(script)
        if script == "broken script":
            raise SandboxScriptError("repairable script failure")
        return '{"data": [{"type": "bar"}], "layout": {}}'

    monkeypatch.setattr(sandbox_plot, "api_query_resilient", reply)
    monkeypatch.setattr(sandbox, "run_script", run_script)
    ctx = tools.ToolContext(
        client=ClientStub(),
        model=TEST_MODEL,
        opts=tools.ToolOptions(),
        chat_id="chat-1",
        tool_call_id="call-1",
        sandbox_service=sandbox,
        small_model="provider/small-model",
    )

    outcome = await tools.BUILTIN_TOOLS.execute("sandbox_plot", {"brief": "Plot the trend"}, ctx)

    assert (outcome.summary, outcome.error) == ("1 trace", None)
    assert outcome.detail["script"] == "repaired script"
    assert sandbox.scripts == ["broken script", "repaired script"]
    assert sandbox.calls == []
    assert len(generation_calls) == 2
    assert generation_calls[0]["user_msg"] == "Plot the trend"
    assert "repairable script failure" in generation_calls[1]["user_msg"]
    assert [call["model"] for call in generation_calls] == ["provider/small-model", "provider/small-model"]


@pytest.mark.asyncio
async def test_sandbox_plot_generation_exception_hands_off_without_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = SandboxServiceStub('{"data": [{"type": "scatter"}], "layout": {}}')
    generation_calls: list[dict[str, Any]] = []

    def unavailable(*_args: Any, **kwargs: Any) -> RuntimeError:
        generation_calls.append(kwargs)
        return RuntimeError("provider unavailable")

    monkeypatch.setattr(sandbox_plot, "api_query_resilient", unavailable)
    ctx = tools.ToolContext(
        client=ClientStub(),
        model=TEST_MODEL,
        opts=tools.ToolOptions(),
        chat_id="chat-1",
        tool_call_id="call-1",
        sandbox_service=sandbox,
        small_model="provider/small-model",
    )

    outcome = await tools.BUILTIN_TOOLS.execute("sandbox_plot", {"brief": "Plot the trend"}, ctx)

    assert (outcome.summary, outcome.error) == ("1 trace", None)
    assert outcome.detail["script"] == ""
    assert sandbox.scripts == []
    assert len(generation_calls) == 1
    assert generation_calls[0]["model"] == "provider/small-model"
    assert [call["scope"] for call in sandbox.calls] == ["sandbox_plot"]


@pytest.mark.asyncio
async def test_sandbox_plot_falls_back_to_the_agent_when_the_script_keeps_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = SandboxServiceStub('{"data": [{"type": "scatter"}], "layout": {}}', script_stdout=None)
    monkeypatch.setattr(sandbox_plot, "api_query_resilient", lambda *_args, **_kwargs: _script_reply("print(fig.to_json())"))
    ctx = tools.ToolContext(
        client=ClientStub(),
        model=TEST_MODEL,
        opts=tools.ToolOptions(),
        chat_id="chat-1",
        tool_call_id="call-1",
        sandbox_service=sandbox,
    )

    outcome = await tools.BUILTIN_TOOLS.execute("sandbox_plot", {"brief": "Plot the trend"}, ctx)

    assert outcome.error is None
    assert len(sandbox.scripts) == 2  # One repair attempt, then hand over.
    assert [call["scope"] for call in sandbox.calls] == ["sandbox_plot"]


@pytest.mark.asyncio
async def test_sandbox_plot_with_images_uses_the_agent_path() -> None:
    """Only the agent path can see prompt images, so image briefs skip the fast path."""
    sandbox = SandboxServiceStub('{"data": [{"type": "scatter"}], "layout": {}, "frames": null}')
    ctx = tools.ToolContext(
        client=ClientStub(),
        model=TEST_MODEL,
        opts=tools.ToolOptions(),
        chat_id="chat-1",
        tool_call_id="call-1",
        prompt_images=(PromptImage("photo.png", b"image"),),
        sandbox_service=sandbox,
    )

    outcome = await tools.BUILTIN_TOOLS.execute("sandbox_plot", {"brief": "Plot the trend"}, ctx)

    assert outcome.error is None
    assert outcome.summary == "1 trace"
    assert outcome.detail["brief"] == "Plot the trend"
    assert outcome.detail["script"] == ""
    assert "transcript" not in outcome.content
    assert sandbox.scripts == []
    # The images must reach the agent under the plot scope. Tuning knobs like
    # thinking/lean are deliberately not pinned.
    assert len(sandbox.calls) == 1
    assert sandbox.calls[0]["scope"] == "sandbox_plot"
    assert sandbox.calls[0]["prompt"] == "Plot the trend"
    assert sandbox.calls[0]["prompt_images"] == (PromptImage("photo.png", b"image"),)


@pytest.mark.asyncio
async def test_sandbox_plot_invalid_output_fails_without_exposing_transcript() -> None:
    sandbox = SandboxServiceStub("not json")
    ctx = tools.ToolContext(client=ClientStub(), model=TEST_MODEL, opts=tools.ToolOptions(), sandbox_service=sandbox)

    outcome = await tools.BUILTIN_TOOLS.execute("sandbox_plot", {"brief": "Plot the trend"}, ctx)

    assert (outcome.summary, outcome.error) == ("Failed", "invalid_figure")
    # The stub's transcript is what must not reach the model, so assert on that exact text.
    assert "hidden transcript" not in outcome.content


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
    app.dependency_overrides[chat_route.get_sandbox_service] = lambda: None
    app.dependency_overrides[chat_route.get_model_provider_store] = lambda: SimpleNamespace(
        load_defaults=lambda: dict.fromkeys(MODEL_DEFAULT_KEYS, TEST_MODEL)
    )
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
    monkeypatch.setattr(toolcalling.litellm, "completion", lambda **_: rounds.pop(0))

    async def fake_execute(*_: Any, **__: Any) -> toolcalling.ToolOutcome:
        return toolcalling.ToolOutcome(content="[ap] evidence", summary="1 sources", sources=[{"label": "ap", "url": "u"}])

    monkeypatch.setattr(tools.BUILTIN_TOOLS, "execute", fake_execute)

    from backend.routes import chat as chat_route

    monkeypatch.setattr(chat_route, "_resolve_images", lambda *_, **__: None)

    client = TestClient(_chat_app(monkeypatch))
    response = client.post(
        "/api/chat",
        json={
            "model": TEST_MODEL,
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


@pytest.mark.asyncio
async def test_workspace_agent_result_carries_sandbox_state_to_the_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only sandbox-backed tools attach a `sandbox` payload; ordinary tools leave it null."""
    rounds = [
        iter([_tool_chunk(0, "call_1", "workspace_agent", '{"prompt": "write a script"}')]),
        iter([_text_chunk("OK")]),
    ]
    monkeypatch.setattr(toolcalling.litellm, "completion", lambda **_: rounds.pop(0))

    events = await _collect(
        enabled=["workspace_agent"], chat_id="chat-1", sandbox_service=SandboxServiceStub("unused")
    )
    (result,) = [data for name, data in events if name == "tool_result"]

    assert result["name"] == "workspace_agent"
    assert result["sandbox"]["status"] == "completed"
    assert result["sandbox"]["workspace_changed"] is True

    rounds = [
        iter([_tool_chunk(0, "call_2", "web_search", '{"query": "x"}')]),
        iter([_text_chunk("OK")]),
    ]

    async def fake_execute(*_: Any, **__: Any) -> toolcalling.ToolOutcome:
        return toolcalling.ToolOutcome(content="evidence", summary="1 sources")

    monkeypatch.setattr(tools.BUILTIN_TOOLS, "execute", fake_execute)

    events = await _collect(enabled=["web_search"], chat_id="chat-1", sandbox_service=SandboxServiceStub("unused"))
    (result,) = [data for name, data in events if name == "tool_result"]

    assert result["sandbox"] is None


@pytest.mark.asyncio
async def test_workspace_agent_without_a_sandbox_service_degrades_instead_of_raising() -> None:
    """A missing sandbox must become a tool error the model can talk about, never a dead stream."""
    ctx = tools.ToolContext(
        client=ClientStub(), model=TEST_MODEL, opts=tools.ToolOptions(), chat_id="chat-1", sandbox_service=None
    )

    outcome = await tools.BUILTIN_TOOLS.execute("workspace_agent", {"prompt": "do work"}, ctx)

    assert outcome.error is not None
    assert outcome.content
