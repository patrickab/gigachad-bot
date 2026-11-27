"""Built-in tools: the workspace agent, the OCR card, learn sessions."""

import base64
from typing import Any

import pytest

from config import TEST_MODEL
from lib import completions, learnharness, toolcalling, tools
from lib.chat_store import ChatStore
from lib.project_store import ProjectStore
from lib.tool_context import ToolContext, ToolOptions
from lib.upstream import UpstreamContractError, UpstreamRequestFailed
from tests.fakes import MemoryDataStore, SandboxServiceStub, collect_tool_turn, text_chunk, tool_chunk


@pytest.mark.asyncio
async def test_workspace_agent_result_carries_sandbox_state_to_the_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only sandbox-backed tools attach a `sandbox` payload; ordinary tools leave it null."""
    rounds = [
        iter([tool_chunk(0, "call_1", "workspace_agent", '{"prompt": "write a script"}')]),
        iter([text_chunk("OK")]),
    ]
    monkeypatch.setattr(completions, "stream", lambda *_, **__: rounds.pop(0))

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
    ctx = ToolContext(model=TEST_MODEL, opts=ToolOptions(), chat_id="chat-1", sandbox_service=None)

    outcome = await tools.BUILTIN_TOOLS.execute("workspace_agent", {"prompt": "do work"}, ctx)

    assert outcome.error is not None
    assert outcome.content


@pytest.mark.asyncio
async def test_latex_ocr_tool_uses_the_vision_model_for_attached_images(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_text(model: str, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        calls.append({"model": model, "messages": messages, **kwargs})
        return "$a^2 + b^2 = c^2$"

    monkeypatch.setattr(completions, "text", fake_text)
    context = ToolContext(
        model=TEST_MODEL,
        vision_model="provider/vision",
        opts=ToolOptions(),
        img=b"image-bytes",
    )

    outcome = await tools.BUILTIN_TOOLS.execute("latex_ocr", {}, context)

    assert outcome.detail["text"] == "$a^2 + b^2 = c^2$"
    assert calls[0]["model"] == "provider/vision"
    image_url = calls[0]["messages"][-1]["content"][-1]["image_url"]["url"]
    assert image_url == f"data:image/jpeg;base64,{base64.b64encode(b'image-bytes').decode()}"


class _LearnHarnessStub:
    """Records the sessions the tool creates and the turns it starts, `refusal` makes the turn fail."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, refusal: Exception | None = None) -> None:
        self.titles: list[str] = []
        self.turns: list[tuple[str, str]] = []
        self._refusal = refusal
        monkeypatch.setattr(learnharness, "create_session", self.create_session)
        monkeypatch.setattr(learnharness, "submit_text_turn", self.submit_text_turn)

    async def create_session(self, title: str) -> learnharness.LearnSession:
        self.titles.append(title)
        return learnharness.LearnSession(id="chat_new_1", title=title)

    async def submit_text_turn(self, session_id: str, text: str) -> None:
        if self._refusal is not None:
            raise self._refusal
        self.turns.append((session_id, text))


def _learn_context(project_slug: str | None) -> tuple[ToolContext, ProjectStore]:
    store = ProjectStore(ChatStore(data_store=MemoryDataStore()), rename=lambda *_: None)
    store.create_project("Physics")
    context = ToolContext(model=TEST_MODEL, opts=ToolOptions(), project_slug=project_slug, project_store=store)
    return context, store


GOAL = "I want to understand Fourier series properly. I know calculus but not complex numbers."


@pytest.mark.asyncio
async def test_start_learn_session_starts_the_tutor_on_the_goal_in_the_chats_project(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _LearnHarnessStub(monkeypatch)
    context, store = _learn_context("physics")

    outcome = await tools.BUILTIN_TOOLS.execute("start_learn_session", {"goal": GOAL}, context)

    assert outcome.error is None
    assert harness.titles[0].startswith("I want to understand Fourier series")
    assert "complex numbers" not in harness.titles[0]
    assert harness.turns == [("chat_new_1", GOAL)]
    assert outcome.detail == {"session_id": "chat_new_1", "title": harness.titles[0], "project_slug": "physics"}
    assert store.learn_session_projects() == {"chat_new_1": "physics"}


@pytest.mark.asyncio
async def test_start_learn_session_outside_a_project_joins_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _LearnHarnessStub(monkeypatch)
    context, store = _learn_context(None)

    outcome = await tools.BUILTIN_TOOLS.execute("start_learn_session", {"goal": GOAL}, context)

    assert outcome.detail["project_slug"] is None
    assert store.learn_session_projects() == {}


@pytest.mark.asyncio
async def test_a_refused_first_turn_still_links_the_created_session(monkeypatch: pytest.MonkeyPatch) -> None:
    _LearnHarnessStub(monkeypatch, refusal=UpstreamRequestFailed(503, "No model configured — set up your models first"))
    context, store = _learn_context("physics")

    outcome = await tools.BUILTIN_TOOLS.execute("start_learn_session", {"goal": GOAL}, context)

    assert outcome.error == "No model configured — set up your models first"
    assert outcome.detail["session_id"] == "chat_new_1"
    assert store.learn_session_projects() == {"chat_new_1": "physics"}


@pytest.mark.asyncio
async def test_an_unexpected_turn_answer_still_links_the_created_session(monkeypatch: pytest.MonkeyPatch) -> None:
    _LearnHarnessStub(monkeypatch, refusal=UpstreamContractError("LearnHarness /api/chat: field `ok` field required"))
    context, _ = _learn_context("physics")

    outcome = await tools.BUILTIN_TOOLS.execute("start_learn_session", {"goal": GOAL}, context)

    assert "`ok`" in (outcome.error or "")
    assert outcome.detail["session_id"] == "chat_new_1"


@pytest.mark.asyncio
async def test_a_failed_assignment_still_starts_and_links_the_session_outside_the_project(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _LearnHarnessStub(monkeypatch)
    context, store = _learn_context("deleted-meanwhile")

    outcome = await tools.BUILTIN_TOOLS.execute("start_learn_session", {"goal": GOAL}, context)

    assert outcome.error is None
    assert outcome.detail["project_slug"] is None
    assert harness.turns == [("chat_new_1", GOAL)]
    assert "listed under Learn" in outcome.content
    assert store.learn_session_projects() == {}
