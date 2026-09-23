"""Chat-route injection of the chat notebook into the turn's prompt and tool list."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from backend.routes import chat as chat_route


class FakeSandboxService:
    """Shape-compatible stand-in: read_notebook returns the staged record or None."""

    def __init__(self, notebook: dict | None):
        self._notebook = notebook
        self.read_calls: list[str] = []

    async def read_notebook(self, chat_id: str) -> dict | None:
        self.read_calls.append(chat_id)
        return self._notebook


class FakeMemoryStore:
    def augment_system_prompt(self, prompt: str, slug: str | None) -> str:
        return prompt


class FakeModels:
    def load_defaults(self) -> dict:
        return {"small_model": "m", "vision_model": "v"}


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch heavy collaborators so the test observes exactly what the route passes on."""
    seen: dict = {}

    async def fake_stream_chat_with_tools(**kwargs):
        seen.update(kwargs)
        yield ("event", {})

    @contextmanager
    def fake_request_client():
        yield SimpleNamespace(reset_history=lambda: None)

    monkeypatch.setattr(chat_route, "stream_chat_with_tools", fake_stream_chat_with_tools)
    monkeypatch.setattr(chat_route, "sse_tool_event_stream", lambda events: events)
    monkeypatch.setattr(chat_route, "request_client", fake_request_client)
    monkeypatch.setattr(chat_route, "_resolve_images", lambda c, req, assets: None)
    return seen


async def _run_chat(monkeypatch, tools: list[str], sandbox_service) -> dict:
    req = chat_route.ChatRequest(model="m", chat_id="chat-1", user_msg="hi", tools=tools)
    result = await chat_route.chat(
        req,
        memory_store=FakeMemoryStore(),
        assets=object(),
        sandbox_service=sandbox_service,
        models=FakeModels(),
    )
    # The route hands back the (patched-to-identity) event stream; drain it so
    # the patched stream_chat_with_tools actually runs and records its kwargs.
    async for _ in result:
        pass
    assert result is not None
    return req


async def test_staged_notebook_is_injected_and_tool_enabled(captured, monkeypatch):
    source = "# %%\nprint('hello')\n"
    notebook = {"revision_id": "rev-1", "source": source, "outputs": {}}
    sandbox = FakeSandboxService(notebook)
    await _run_chat(monkeypatch, tools=[], sandbox_service=sandbox)

    assert sandbox.read_calls == ["chat-1"]
    assert captured["enabled"] == ["notebook_edit"]
    assert "# Current notebook\n" in captured["system_prompt"]
    assert source in captured["system_prompt"]


async def test_notebook_edit_joins_existing_tools(captured, monkeypatch):
    sandbox = FakeSandboxService({"revision_id": "rev-1", "source": "# %%\nx=1\n", "outputs": {}})
    await _run_chat(monkeypatch, tools=["workspace_agent"], sandbox_service=sandbox)
    assert captured["enabled"] == ["workspace_agent", "notebook_edit"]


async def test_long_notebook_source_is_capped(captured, monkeypatch):
    source = "# %%\nx = " + "1" * 12_000 + "\n"
    sandbox = FakeSandboxService({"revision_id": "rev-1", "source": source, "outputs": {}})
    await _run_chat(monkeypatch, tools=[], sandbox_service=sandbox)
    block = captured["system_prompt"].split("# Current notebook\n", 1)[1]
    assert len(block) <= chat_route.NOTEBOOK_PROMPT_CAP + len("\n… (truncated)")


async def test_without_notebook_nothing_is_injected(captured, monkeypatch):
    sandbox = FakeSandboxService(None)
    await _run_chat(monkeypatch, tools=["workspace_agent"], sandbox_service=sandbox)
    assert "# Current notebook" not in captured["system_prompt"]
    assert "notebook_edit" not in captured["enabled"]