"""omp-web answers become gigachad's snake_case contract, and a changed shape or refusal is reported, not hidden."""

import json
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest

from backend.identity import RequestIdentity, get_request_identity
from backend.routes.omp import router
from lib import omp_web

# Trimmed from a live omp-web 0.5.2 (patrickab fork) response.
SESSIONS = {
    "sessions": [
        {
            "path": "/home/me/.omp/agent/sessions/-git-app/2026-10-18T16-35-53-333Z_01a14fde.jsonl",
            "id": "01a14fde-a375-71a4-b661-1ab558f72944",
            "cwd": "/home/me/git/app-worktrees/nav",
            "name": "NAV: Improve diff viewer",
            "created": "2026-10-18T16:35:53.333Z",
            "modified": "2026-10-18T17:00:53.200Z",
            "messageCount": 2,
            "firstMessage": "UI and UX advice",
            "parentSessionId": "01a14bb7-8824-7655-ac0d-18a0d727da62",
            "projectRoot": "/home/me/git/app",
            "projectKey": "/home/me/git/app",
            "worktreeBranch": "nav",
        }
    ],
    "runningSessionIds": ["01a14fde-a375-71a4-b661-1ab558f72944"],
    "runningSessions": [],
}
PROJECTS = {"projects": [{"path": "/home/me/git/app", "addedAt": "2026-10-01T10:00:00.000Z"}, {"path": "/home/me/.os"}]}


def _serve(monkeypatch, routes: dict[str, tuple[int, object]]) -> None:
    """Answer `METHOD /path` from `routes`, anything else is a connection failure."""

    def handle(request: httpx.Request) -> httpx.Response:
        key = f"{request.method} {request.url.path}"
        if key not in routes:
            raise httpx.ConnectError("refused", request=request)
        status, body = routes[key]
        return httpx.Response(status, content=json.dumps(body).encode())

    monkeypatch.setattr(omp_web, "_transport", httpx.MockTransport(handle))


async def test_overview_joins_sessions_and_workspaces_in_snake_case(monkeypatch):
    _serve(monkeypatch, {"GET /api/sessions": (200, SESSIONS), "GET /api/projects": (200, PROJECTS)})

    overview = (await omp_web.overview()).model_dump()

    [session] = overview["sessions"]
    assert session["parent_session_id"] == "01a14bb7-8824-7655-ac0d-18a0d727da62"
    assert session["project_root"] == "/home/me/git/app"
    assert session["worktree_branch"] == "nav"
    assert session["message_count"] == 2
    assert overview["running_session_ids"] == ["01a14fde-a375-71a4-b661-1ab558f72944"]
    assert [workspace["path"] for workspace in overview["workspaces"]] == ["/home/me/git/app", "/home/me/.os"]


async def test_a_dropped_field_names_the_field(monkeypatch):
    broken = {**SESSIONS, "sessions": [{key: value for key, value in SESSIONS["sessions"][0].items() if key != "messageCount"}]}
    _serve(monkeypatch, {"GET /api/sessions": (200, broken), "GET /api/projects": (200, PROJECTS)})

    with pytest.raises(omp_web.OmpWebContractError, match=r"/api/sessions: field `sessions\.0\.messageCount` field required"):
        await omp_web.overview()


async def test_system_prompt_joins_parts_and_is_none_while_stopped(monkeypatch):
    session_id = "01a14fde-a375-71a4-b661-1ab558f72944"
    _serve(
        monkeypatch, {f"GET /api/agent/{session_id}": (200, {"running": True, "state": {"systemPrompt": ["You are omp.", "Rules."]}})}
    )
    assert (await omp_web.system_prompt(session_id)).model_dump() == {"running": True, "system_prompt": "You are omp.\n\nRules."}

    _serve(monkeypatch, {f"GET /api/agent/{session_id}": (200, {"running": False})})
    assert (await omp_web.system_prompt(session_id)).model_dump() == {"running": False, "system_prompt": None}


def _client(has_host_access: bool) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_request_identity] = lambda: RequestIdentity(uuid4(), "me@example.test", None, has_host_access)
    return TestClient(app)


@pytest.fixture
def client() -> TestClient:
    return _client(has_host_access=True)


def test_logins_without_host_access_never_reach_omp_web(monkeypatch):
    _serve(monkeypatch, {"GET /api/sessions": (200, SESSIONS), "GET /api/projects": (200, PROJECTS)})
    assert _client(has_host_access=False).get("/api/omp/overview").status_code == 403


def test_refusals_keep_omp_webs_status_and_message(monkeypatch, client):
    refusal = {"error": "Archive child sessions before archiving this session", "code": "session_has_children"}
    _serve(monkeypatch, {"POST /api/sessions/abc-1/archive": (409, refusal)})

    response = client.post("/api/omp/sessions/abc-1/archive")

    assert response.status_code == 409
    assert response.json()["detail"] == refusal["error"]


def test_unreachable_omp_web_is_503_and_unknown_shape_is_502(monkeypatch, client):
    _serve(monkeypatch, {})
    assert client.get("/api/omp/archive").status_code == 503

    _serve(monkeypatch, {"GET /api/sessions/archive": (200, {"archive": []})})
    assert client.get("/api/omp/archive").status_code == 502


def test_session_ids_outside_the_uuid_alphabet_never_reach_omp_web(monkeypatch, client):
    _serve(monkeypatch, {})
    assert client.get("/api/omp/sessions/..%2Fprojects/system-prompt").status_code == 404
    assert client.get("/api/omp/sessions/a.b/system-prompt").status_code == 422
