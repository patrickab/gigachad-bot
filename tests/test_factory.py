"""Factory engine answers become gigachad's snake_case contract, and a changed shape or refusal is reported, not hidden."""

import json
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest

from backend.identity import RequestIdentity, get_request_identity
from backend.routes.factory import router
from lib import factory

TASK_ID = "k3x9a7bq"

TASK = {
    "id": TASK_ID,
    "repo": "gigachad-bot",
    "workflow": "code",
    "title": "Rename the sidebar",
    "request": "Call it Library",
    "state": "human_review",
    "phase": "review",
    "round": 1,
    "config": {"merge": "review"},
    "worktree": f"/home/me/git/worktrees/gigachad-bot/{TASK_ID}",
    "branch": f"factory/{TASK_ID}",
    "base_sha": "a" * 40,
    "error": None,
    "created_at": "2026-10-21T08:00:00Z",
    "updated_at": "2026-10-21T08:05:00Z",
    "added_by_a_newer_engine": True,
}
DETAIL = {
    **TASK,
    "phases": [
        {
            "id": 1,
            "phase": "gates",
            "round": 1,
            "status": "ok",
            "started_at": "2026-10-21T08:01:00Z",
            "ended_at": "2026-10-21T08:02:00Z",
            "summary": "gates passed",
            "envelope": None,
            "gate_results": [{"argv": ["pytest", "-q"], "exit_code": 0, "output_tail": "141 passed"}],
        }
    ],
    "messages": [{"id": 1, "role": "agent", "text": "Done", "phase_run_id": 1, "created_at": "2026-10-21T08:02:00Z"}],
    "preview": {"status": "ready", "url": "http://127.0.0.1:8100", "open_url": "http://127.0.0.1:8100/?open=abc", "error": None},
}


class Engine:
    """Answers `METHOD /path` from `routes`, anything else is a connection failure, and keeps every request it saw."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, routes: dict[str, tuple[int, object]]) -> None:
        self.requests: list[httpx.Request] = []
        monkeypatch.setattr(factory, "_transport", httpx.MockTransport(self._handle))
        self._routes = routes

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f"{request.method} {request.url.path}"
        if key not in self._routes:
            raise httpx.ConnectError("refused", request=request)
        status, body = self._routes[key]
        return httpx.Response(status, content=json.dumps(body).encode())


def _client(has_host_access: bool) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_request_identity] = lambda: RequestIdentity(uuid4(), "me@example.test", None, has_host_access)
    return TestClient(app)


@pytest.fixture
def client() -> TestClient:
    return _client(has_host_access=True)


async def test_task_detail_validates_and_drops_unknown_engine_fields(monkeypatch):
    Engine(monkeypatch, {f"GET /v1/tasks/{TASK_ID}": (200, DETAIL)})

    detail = (await factory.task(TASK_ID)).model_dump(mode="json")

    assert "added_by_a_newer_engine" not in detail


def test_logins_without_host_access_never_reach_the_engine(monkeypatch):
    engine = Engine(monkeypatch, {"GET /v1/tasks": (200, [TASK])})
    blocked = _client(has_host_access=False)

    assert blocked.get("/api/factory/tasks", params={"repo": "gigachad-bot"}).status_code == 403
    assert blocked.post(f"/api/factory/tasks/{TASK_ID}/approve").status_code == 403
    assert engine.requests == []


def test_refusals_keep_the_engines_status_and_message(monkeypatch, client):
    refusal = {"error": "conflict", "detail": "Task is not in human_review"}
    Engine(monkeypatch, {f"POST /v1/tasks/{TASK_ID}/approve": (409, refusal)})

    response = client.post(f"/api/factory/tasks/{TASK_ID}/approve")

    assert response.status_code == 409
    assert response.json()["detail"] == refusal["detail"]


def test_invalid_requests_never_reach_the_engine(monkeypatch, client):
    engine = Engine(monkeypatch, {})

    assert client.get("/api/factory/tasks/a.b").status_code == 422
    assert client.get("/api/factory/workflows", params={"repo": "../etc"}).status_code == 422
    assert engine.requests == []
