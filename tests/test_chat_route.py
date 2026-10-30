"""The real /api/chat router streams over unbuffered SSE, with only the provider call faked out."""

from typing import Any

from fastapi.testclient import TestClient
import pytest

from backend import http_plumbing
from backend.routes import chat as chat_route
from config import MODEL_DEFAULT_KEYS, TEST_MODEL
from lib import toolcalling, tools
from tests.conftest import request_identity, route_client
from tests.fakes import ClientStub, text_chunk, tool_chunk


class _client_ctx:  # noqa: N801 - stands in for the request_client contextmanager
    def __enter__(self) -> ClientStub:
        return ClientStub()

    def __exit__(self, *_: Any) -> None:
        return None


def _chat_client(monkeypatch: pytest.MonkeyPatch, *, has_host_access: bool = True) -> TestClient:
    monkeypatch.setattr(chat_route, "request_client", lambda: _client_ctx())
    monkeypatch.setattr(chat_route, "augment_system_prompt", lambda store, prompt, slug: prompt)  # noqa: ARG005 - signature parity
    monkeypatch.setattr(chat_route, "_resolve_images", lambda *_, **__: None)
    return route_client(
        chat_route.router,
        overrides={
            chat_route.get_memory_store: None,
            chat_route.get_asset_store: None,
            chat_route.get_sandbox_service: None,
            chat_route.get_project_store: None,
            chat_route.get_request_identity: request_identity(has_host_access),
            chat_route.get_model_defaults: dict.fromkeys(MODEL_DEFAULT_KEYS, TEST_MODEL),
        },
    )


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
    rounds = [
        iter([tool_chunk(0, "call_a", "web_search", '{"query": "cheesecake"}')]),
        iter([text_chunk("Try [ap]")]),
    ]
    monkeypatch.setattr(toolcalling.litellm, "completion", lambda **_: rounds.pop(0))

    async def fake_execute(*_: Any, **__: Any) -> toolcalling.ToolOutcome:
        return toolcalling.ToolOutcome(content="[ap] evidence", summary="1 sources", sources=[{"label": "ap", "url": "u"}])

    monkeypatch.setattr(tools.BUILTIN_TOOLS, "execute", fake_execute)

    response = _chat_client(monkeypatch).post(
        "/api/chat",
        json={
            "model": TEST_MODEL,
            "chat_id": "c1",
            "user_msg": "search the web for cheesecake recipes",
            "tools": ["web_search"],
        },
    )

    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"
    events = _sse_events(response.text)
    assert [name for name, _ in events] == ["tool_call", "tool_progress", "tool_progress", "tool_result", "token", "done"]
    assert '"name": "web_search"' in events[0][1]
    assert events[4][1] == "Try [ap]"


def test_chat_route_streams_plain_turn_without_proxy_buffering(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ordinary chat path uses the same unbuffered SSE contract as tool turns."""
    monkeypatch.setattr(http_plumbing, "api_query_resilient", lambda *_, **__: iter(["First", " answer"]))

    response = _chat_client(monkeypatch).post(
        "/api/chat",
        json={"model": TEST_MODEL, "chat_id": "c1", "user_msg": "hello"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"
    assert _sse_events(response.text) == [("token", "First"), ("token", " answer"), ("done", "")]


@pytest.mark.parametrize(("has_host_access", "offered"), [(True, ["web_search", "start_learn_session"]), (False, ["web_search"])])
def test_learn_sessions_are_offered_to_host_logins_only(monkeypatch: pytest.MonkeyPatch, has_host_access, offered) -> None:
    """A browser asking for a host tool without the host capability never gets it in front of the model."""
    offers: list[list[str]] = []

    def completion(**kwargs: Any) -> Any:
        offers.append([spec["function"]["name"] for spec in kwargs.get("tools") or []])
        return iter([text_chunk("Hi")])

    monkeypatch.setattr(toolcalling.litellm, "completion", completion)

    response = _chat_client(monkeypatch, has_host_access=has_host_access).post(
        "/api/chat",
        json={"model": TEST_MODEL, "chat_id": "c1", "user_msg": "teach me", "tools": ["web_search", "start_learn_session"]},
    )

    assert response.status_code == 200, response.text
    assert offers == [offered]
