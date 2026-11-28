"""pi-host answers reach the browser unchanged, a missing host, a changed shape or a refusal is reported, and the WS proxy relays."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
import json
import threading

from fastapi.testclient import TestClient
import httpx
import pytest
from starlette.websockets import WebSocketDisconnect
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from backend import websocket_origin
from backend.identity import get_request_identity
from backend.routes.deps import get_file_roots
from backend.routes.pi import router
import config
from lib import pi_host, pi_media
from lib.file_roots import FileRoots
from tests.conftest import mounted_roots, request_identity, route_client

TOKEN = "secret-token"
ORIGIN = "http://localhost:2999"
SESSION_ID = "0b3c1d7e-5f7a-4d2b-9a51-6c1e2f3a4b5c"
SESSION = {
    "sessionId": SESSION_ID,
    "sessionPath": "/home/me/.pi/agent/sessions/--home-me-git-app--/2026-11-19T10-00-00-000Z_0b3c1d7e.jsonl",
    "cwd": "/home/me/git/app",
    "machine": "local",
    "createdAt": "2026-11-19T10:00:00.000Z",
    "modifiedAt": "2026-11-19T10:05:00.000Z",
    "firstMessage": "Fix the tests",
    "messageCount": 4,
    "tokens": 84300,
    "live": {"running": True, "waiting": False, "attached": 1},
    "pinned": False,
    "archived": False,
}


@pytest.fixture(autouse=True)
def token_file(tmp_path, monkeypatch):
    path = tmp_path / "pi-host.token"
    path.write_text(f"{TOKEN}\n")
    monkeypatch.setenv("PI_HOST_TOKEN_FILE", str(path))
    return path


@pytest.fixture
def repositories(tmp_path, monkeypatch):
    root = tmp_path / "git"
    (root / "app" / ".git").mkdir(parents=True)
    (root / "notes").mkdir()
    (tmp_path / "dev" / "tool").mkdir(parents=True)
    (tmp_path / "code" / "proj" / ".git").mkdir(parents=True)
    monkeypatch.setattr(pi_host, "HOME", tmp_path)
    monkeypatch.setattr(pi_host, "DEV_ROOT", tmp_path / "dev")
    monkeypatch.setattr(pi_host, "REPOSITORIES_ROOT", root)
    return root


@dataclass
class FakeHost:
    """Answers `METHOD /path` from `routes`, anything else is a connection failure. Every request is kept."""

    routes: dict[str, tuple[int, object]]
    requests: list[httpx.Request] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f"{request.method} {request.url.path}"
        if key not in self.routes:
            raise httpx.ConnectError("refused", request=request)
        status, body = self.routes[key]
        return httpx.Response(status) if body is None else httpx.Response(status, content=json.dumps(body).encode())


def _serve(monkeypatch, routes: dict[str, tuple[int, object]]) -> FakeHost:
    host = FakeHost(routes)
    monkeypatch.setattr(pi_host, "_transport", httpx.MockTransport(host.handle))
    return host


def _client(has_host_access: bool = True, roots: FileRoots | None = None) -> TestClient:
    overrides = {get_request_identity: request_identity(has_host_access), get_file_roots: roots or mounted_roots()}
    return route_client(router, overrides=overrides)


def test_urls_follow_the_environment(monkeypatch):
    monkeypatch.setenv("GIGACHAD_PI_HOST_URL", "http://127.0.0.1:8311/")
    assert pi_host.base_url() == "http://127.0.0.1:8311"
    assert pi_host.gateway_url() == "http://127.0.0.1:8311/v1"
    assert pi_host.token() == TOKEN


async def test_requests_carry_the_bearer_token(monkeypatch):
    host = _serve(monkeypatch, {"GET /health": (200, {"ok": True, "piVersion": "1.5.2"})})

    health = await pi_host.health()

    assert health.pi_version == "1.5.2"
    assert host.requests[0].headers["authorization"] == f"Bearer {TOKEN}"
    assert "origin" not in host.requests[0].headers


async def test_a_missing_token_file_means_pi_host_is_unavailable(monkeypatch, token_file):
    token_file.unlink()
    _serve(monkeypatch, {"GET /health": (200, {"ok": True, "piVersion": "1.5.2"})})

    with pytest.raises(pi_host.PiHostUnavailable, match="token"):
        await pi_host.health()


def test_the_gateway_token_is_its_own_file(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.delenv("PI_GATEWAY_TOKEN_FILE", raising=False)
    (tmp_path / "gigachad-bot").mkdir()
    (tmp_path / "gigachad-bot" / "pi-gateway.token").write_text("gateway-only\n")
    assert pi_host.gateway_token() == "gateway-only"

    monkeypatch.setenv("PI_GATEWAY_TOKEN_FILE", str(tmp_path / "absent"))
    with pytest.raises(pi_host.PiHostUnavailable, match="gateway token"):
        pi_host.gateway_token()


def test_sessions_pass_through_camel_case_with_the_repo_resolved(monkeypatch, repositories):
    host = _serve(monkeypatch, {"GET /sessions": (200, {"sessions": [SESSION]})})

    response = _client().get("/api/pi/sessions", params={"repo": "app", "archived": "true"})

    assert response.status_code == 200, response.text
    expected = {**SESSION, "parentSession": None, "profile": "coding", "factory": None, "usage": None, "cachedUntil": None}
    assert response.json() == {"sessions": [expected]}
    assert dict(host.requests[0].url.params) == {"cwd": str(repositories / "app"), "archived": "1"}


def test_sessions_without_a_repo_list_every_cwd(monkeypatch):
    host = _serve(monkeypatch, {"GET /sessions": (200, {"sessions": [SESSION, {**SESSION, "cwd": "/home/me/git/other"}]})})

    response = _client().get("/api/pi/sessions")

    assert response.status_code == 200, response.text
    assert [session["cwd"] for session in response.json()["sessions"]] == ["/home/me/git/app", "/home/me/git/other"]
    assert dict(host.requests[0].url.params) == {"archived": "0"}


def test_sessions_leave_factory_tasks_out(monkeypatch):
    running = {"state": "running", "round": 1, "reason": "", "at": "2026-11-19T10:00:00.000Z"}
    task = {**SESSION, "sessionId": "task", "factory": running}
    host = _serve(monkeypatch, {"GET /sessions": (200, {"sessions": [SESSION, task]})})

    picked = _client().get("/api/pi/sessions")

    assert [session["sessionId"] for session in picked.json()["sessions"]] == [SESSION["sessionId"]]
    assert [dict(request.url.params) for request in host.requests] == [{"archived": "0"}]


def test_search_passes_hits_through_with_the_query_and_limit(monkeypatch):
    found = {"entryId": "a1b2c3d4", "ordinal": 1, "quality": 1, "before": "…fix the ", "match": "tests", "after": "…"}
    hit = {"session": SESSION, "snippet": "…fix the tests…", "entryId": "a1b2c3d4", "matches": [found]}
    host = _serve(monkeypatch, {"GET /search": (200, {"hits": [hit]})})

    response = _client().get("/api/pi/search", params={"q": "fix tests"})

    assert response.status_code == 200, response.text
    session = {**SESSION, "parentSession": None, "profile": "coding", "factory": None, "usage": None, "cachedUntil": None}
    assert response.json() == {"hits": [{**hit, "session": session}]}
    assert dict(host.requests[0].url.params) == {"q": "fix tests", "limit": "20"}
    assert _client().get("/api/pi/search").status_code == 422


def test_session_matches_pass_ranked_hits_through_with_the_query_and_limit(monkeypatch):
    found = {"entryId": "a1b2c3d4", "ordinal": 2, "quality": 0.8, "before": "run the ", "match": "test", "after": "s"}
    page = {"matches": [found], "total": 7}
    host = _serve(monkeypatch, {f"GET /sessions/{SESSION_ID}/matches": (200, page)})

    response = _client().get(f"/api/pi/sessions/{SESSION_ID}/matches", params={"q": "test", "limit": 5})

    assert (response.status_code, response.json()) == (200, page)
    assert dict(host.requests[0].url.params) == {"q": "test", "limit": "5"}
    assert _client().get(f"/api/pi/sessions/{SESSION_ID}/matches").status_code == 422
    assert _client().get(f"/api/pi/sessions/{SESSION_ID}/matches", params={"q": "x", "limit": 0}).status_code == 422


def test_session_matches_of_an_unknown_session_are_404(monkeypatch):
    _serve(monkeypatch, {f"GET /sessions/{SESSION_ID}/matches": (404, {"error": "not_found"})})

    assert _client().get(f"/api/pi/sessions/{SESSION_ID}/matches", params={"q": "x"}).status_code == 404


def test_entries_pass_pi_shapes_through_untouched(monkeypatch):
    entries = {"entries": [{"type": "message", "id": "e1", "parentId": None, "message": {"role": "user"}}], "leafId": "e1"}
    host = _serve(monkeypatch, {f"GET /sessions/{SESSION_ID}/entries": (200, entries)})

    response = _client().get(f"/api/pi/sessions/{SESSION_ID}/entries")

    assert response.status_code == 200, response.text
    assert response.json() == entries
    assert _client().get(f"/api/pi/sessions/{SESSION_ID}/entries", params={"turns": 3}).status_code == 200
    assert dict(host.requests[-1].url.params) == {"turns": "3"}
    assert _client().get(f"/api/pi/sessions/{SESSION_ID}/entries", params={"turns": 0}).status_code == 422


def test_entries_forward_the_custom_type_and_since_of_a_factory_timeline(monkeypatch):
    entry = {"type": "custom", "id": "e2", "customType": "factory-state", "data": {"state": "running"}}
    host = _serve(monkeypatch, {f"GET /sessions/{SESSION_ID}/entries": (200, {"entries": [entry], "leafId": "e9"})})

    response = _client().get(f"/api/pi/sessions/{SESSION_ID}/entries", params={"customType": "factory-", "since": "e1"})

    assert (response.status_code, response.json()) == (200, {"entries": [entry], "leafId": "e9"})
    assert dict(host.requests[-1].url.params) == {"customType": "factory-", "since": "e1"}
    _client().get(f"/api/pi/sessions/{SESSION_ID}/entries", params={"customType": "factory-"})
    assert dict(host.requests[-1].url.params) == {"customType": "factory-"}


@pytest.mark.parametrize(
    ("params", "error"),
    [({"customType": "factory-", "since": "gone"}, "invalid_since"), ({"turns": 2, "since": "e1"}, "invalid_query")],
)
def test_entries_pass_the_refusal_of_a_bad_filter_on_as_400(monkeypatch, params, error):
    _serve(monkeypatch, {f"GET /sessions/{SESSION_ID}/entries": (400, {"error": error})})

    refused = _client().get(f"/api/pi/sessions/{SESSION_ID}/entries", params=params)

    assert (refused.status_code, refused.json()) == (400, {"detail": error, "body": {"error": error}})


async def test_factory_entries_are_filtered_by_custom_type_and_since(monkeypatch):
    entry = {"type": "custom", "id": "e2", "customType": "factory-state", "data": {"state": "running", "round": 1, "reason": ""}}
    host = _serve(monkeypatch, {f"GET /sessions/{SESSION_ID}/entries": (200, {"entries": [entry], "leafId": "e9"})})

    answer = await pi_host.session_entries(SESSION_ID, custom_type="factory-", since="e1")
    await pi_host.session_entries(SESSION_ID, custom_type="factory-")

    assert (answer.entries, answer.leaf_id) == ([entry], "e9")
    assert dict(host.requests[0].url.params) == {"customType": "factory-", "since": "e1"}
    assert dict(host.requests[1].url.params) == {"customType": "factory-"}


async def test_a_missing_factory_plugin_is_reported_with_where_it_comes_from(monkeypatch):
    message = f"The factory is not installed. Install it from {pi_host.FACTORY_PLUGIN_SOURCE}"
    _serve(monkeypatch, {"POST /factory/call": (503, {"error": "factory_plugin_missing", "message": message})})

    with pytest.raises(pi_host.FactoryPluginMissing) as missing:
        await pi_host.factory_call("repos.list", {})

    assert "github.com/patrickab/agent-factory" in str(missing.value)


async def test_a_factory_call_answers_the_result_and_a_refusal_its_code(monkeypatch):
    host = _serve(monkeypatch, {"POST /factory/call": (200, {"result": {"repos": ["/repo"]}})})

    assert await pi_host.factory_call("repos.list", {}) == {"repos": ["/repo"]}
    assert json.loads(host.requests[0].content) == {"action": "repos.list", "params": {}}
    assert host.requests[0].extensions["timeout"]["read"] == pi_host.FACTORY_CALL_TIMEOUT_SECONDS

    _serve(monkeypatch, {"POST /factory/call": (409, {"error": "refused", "message": "task 6f1c2a3b is running, cancel it first"})})
    with pytest.raises(pi_host.FactoryCallError) as refused:
        await pi_host.factory_call("tasks.remove", {"id": "6f1c"})
    assert (refused.value.code, str(refused.value)) == ("refused", "task 6f1c2a3b is running, cancel it first")

    _serve(monkeypatch, {"POST /factory/call": (200, {"answer": 1})})
    with pytest.raises(pi_host.PiHostBadResponse):
        await pi_host.factory_call("repos.list", {})


async def test_delete_removes_the_session_and_an_unknown_one_is_refused(monkeypatch):
    routes = {f"DELETE /sessions/{SESSION_ID}": (204, None), "DELETE /sessions/absent": (404, {"error": "not_found"})}
    host = _serve(monkeypatch, routes)

    assert await pi_host.delete_session(SESSION_ID) is None
    with pytest.raises(pi_host.PiHostRejected) as refused:
        await pi_host.delete_session("absent")

    assert refused.value.status == 404
    assert [request.method for request in host.requests] == ["DELETE", "DELETE"]


def test_a_factory_task_session_carries_its_state():
    state = {"state": "waiting_approve", "round": 2, "reason": "", "at": "2026-11-19T10:05:00.000Z"}

    session = pi_host.SessionInfo.model_validate({**SESSION, "factory": state})

    assert session.factory == pi_host.FactoryState(state="waiting_approve", round=2, reason="", at="2026-11-19T10:05:00.000Z")
    assert session.model_dump(by_alias=True)["factory"] == state
    assert pi_host.SessionInfo.model_validate(SESSION).factory is None


def test_open_session_sends_the_coding_profile_and_resume(monkeypatch, repositories):
    host = _serve(monkeypatch, {"POST /sessions": (200, {"session": SESSION})})
    model = {"provider": "anthropic", "id": "claude-haiku-4-5"}
    body = {"repo": "app", "resume": {"sessionPath": SESSION["sessionPath"]}, "model": model}

    response = _client().post("/api/pi/sessions", json=body)

    assert response.status_code == 200, response.text
    assert response.json()["session"]["sessionId"] == SESSION_ID
    assert json.loads(host.requests[0].content) == {
        "cwd": str(repositories / "app"),
        "profile": "coding",
        "resume": {"sessionPath": SESSION["sessionPath"]},
        "model": {"provider": "anthropic", "id": "claude-haiku-4-5"},
    }


async def test_calls_that_wait_on_a_pi_child_outlast_pi_hosts_own_limit_reads_keep_the_short_one(monkeypatch):
    host = _serve(
        monkeypatch,
        {
            "POST /sessions": (200, {"session": SESSION}),
            f"POST /sessions/{SESSION_ID}/close": (204, None),
            "GET /health": (200, {"ok": True}),
        },
    )

    await pi_host.open_session("/tmp", None)
    await pi_host.close_session(SESSION_ID)
    with suppress(pi_host.PiHostBadResponse):
        await pi_host.health()

    limits = [request.extensions["timeout"]["read"] for request in host.requests]
    # pi-host gives each RPC to a child 30 s, a start runs several.
    assert limits[0] == pi_host.CHILD_TIMEOUT_SECONDS > 30
    assert limits[1] == pi_host.CLOSE_TIMEOUT_SECONDS
    assert limits[2] == pi_host.TIMEOUT_SECONDS


async def test_a_pi_host_that_does_not_answer_is_logged_with_the_call_and_the_exception(monkeypatch, caplog):
    _serve(monkeypatch, {})

    with pytest.raises(pi_host.PiHostUnavailable, match=r"does not answer POST /sessions after \d+ s \(ConnectError"):
        await pi_host.open_session("/tmp", None)

    assert "POST /sessions failed after" in caplog.text
    assert "ConnectError" in caplog.text


def test_open_session_passes_a_fork_through(monkeypatch, repositories):
    host = _serve(monkeypatch, {"POST /sessions": (200, {"session": {**SESSION, "parentSession": "/s/source.jsonl"}})})
    fork_of = {"sessionPath": "/s/source.jsonl", "entryId": "a1b2c3d4"}

    response = _client().post("/api/pi/sessions", json={"repo": "app", "resume": {"forkOf": fork_of}})

    assert response.status_code == 200, response.text
    assert response.json()["session"]["parentSession"] == "/s/source.jsonl"
    sent = json.loads(host.requests[0].content)
    assert sent == {"cwd": str(repositories / "app"), "profile": "coding", "resume": {"forkOf": fork_of}}
    invalid = {"repo": "app", "resume": {"forkOf": {"sessionPath": "/s/source.jsonl"}}}
    assert _client().post("/api/pi/sessions", json=invalid).status_code == 422


async def test_document_sessions_carry_their_prompt_and_resume_by_id(monkeypatch):
    host = _serve(monkeypatch, {"POST /sessions": (200, {"session": {**SESSION, "profile": "document"}})})

    session = await pi_host.open_session(
        "/w/markdown",
        pi_host.ResumeById(session_id=SESSION_ID),
        profile="document",
        system_prompt="Edit it.",
        tools=["read"],
    )

    assert session.profile == "document"
    assert json.loads(host.requests[0].content) == {
        "cwd": "/w/markdown",
        "profile": "document",
        "resume": {"sessionId": SESSION_ID},
        "systemPrompt": "Edit it.",
        "tools": ["read"],
    }


def test_sessions_without_a_profile_are_coding_sessions():
    assert pi_host.SessionInfo.model_validate(SESSION).profile == "coding"


def test_files_of_an_absolute_cwd_inside_the_file_roots_or_the_workbench(monkeypatch, tmp_path):
    host = _serve(monkeypatch, {"GET /files": (200, {"files": [{"path": "notes.md", "score": 1}]})})
    vault, workbench, outside = tmp_path / "vault", tmp_path / "workbench", tmp_path / "outside"
    for folder in (vault / "docs", workbench / "markdown", outside):
        folder.mkdir(parents=True)
    monkeypatch.setattr(config, "WORKBENCH", workbench)
    client = _client(roots=mounted_roots(vault))

    found = {"files": [{"path": "notes.md", "score": 1}]}
    for folder in (vault / "docs", workbench / "markdown"):
        assert client.get("/api/pi/files", params={"cwd": str(folder)}).json() == found
    assert [dict(request.url.params)["cwd"] for request in host.requests] == [str(vault / "docs"), str(workbench / "markdown")]

    assert client.get("/api/pi/files", params={"cwd": str(outside)}).status_code == 404
    assert client.get("/api/pi/files", params={"cwd": str(vault / "docs" / ".." / ".." / "outside")}).status_code == 404
    assert client.get("/api/pi/files", params={"cwd": str(vault / "missing")}).status_code == 404
    assert client.get("/api/pi/files", params={"cwd": "vault/docs"}).status_code == 400
    assert client.get("/api/pi/files").status_code == 400
    assert client.get("/api/pi/files", params={"cwd": str(vault), "repo": "app"}).status_code == 400
    assert len(host.requests) == 2


def test_unknown_repositories_are_refused_before_pi_host(monkeypatch, repositories):
    host = _serve(monkeypatch, {})

    assert _client().post("/api/pi/sessions", json={"repo": "missing", "resume": None}).status_code == 404
    assert _client().get("/api/pi/files", params={"repo": "../etc"}).status_code == 404
    assert host.requests == []


def test_repositories_are_listed(monkeypatch, repositories):
    _serve(monkeypatch, {"GET /sessions": (200, {"sessions": []})})
    names = [repo["name"] for repo in _client().get("/api/pi/repos").json()]
    assert names == ["app", "notes", "tool", "code/proj"]


def test_repositories_are_sorted_by_most_recent_use(monkeypatch, repositories):
    older = {**SESSION, "cwd": str(repositories / "app"), "modifiedAt": "2025-01-01T00:00:00Z"}
    newer = {**SESSION, "cwd": str(repositories.parent / "dev" / "tool"), "modifiedAt": "2025-02-01T00:00:00Z"}
    _serve(monkeypatch, {"GET /sessions": (200, {"sessions": [older, newer]})})
    names = [repo["name"] for repo in _client().get("/api/pi/repos").json()]
    assert names == ["tool", "app", "notes", "code/proj"]


def test_pi_host_down_answers_503(monkeypatch):
    _serve(monkeypatch, {})

    response = _client().get("/api/pi/sessions")

    assert response.status_code == 503
    assert list(response.json()) == ["detail"] and "pi-host" in response.json()["detail"]


def test_a_changed_shape_answers_502_naming_the_field(monkeypatch):
    broken = {key: value for key, value in SESSION.items() if key != "messageCount"}
    _serve(monkeypatch, {"GET /sessions": (200, {"sessions": [broken]})})

    response = _client().get("/api/pi/sessions")

    assert response.status_code == 502 and list(response.json()) == ["detail"]
    assert "`sessions.0.messageCount`" in response.json()["detail"]


def test_a_refusal_keeps_its_status_and_body(monkeypatch, repositories):
    live = {"error": "live", "holder": "tui"}
    _serve(monkeypatch, {"POST /sessions": (409, {"error": "locked"}), f"POST /sessions/{SESSION_ID}/archive": (409, live)})
    client = _client()

    locked = client.post("/api/pi/sessions", json={"repo": "app", "resume": None})
    archived = client.post(f"/api/pi/sessions/{SESSION_ID}/archive")

    assert (locked.status_code, locked.json()) == (409, {"detail": "locked", "body": {"error": "locked"}})
    assert archived.status_code == 409
    assert archived.json()["body"] == live


def test_session_actions_reach_their_pi_host_routes(monkeypatch, repositories):
    host = _serve(
        monkeypatch,
        {
            f"POST /sessions/{SESSION_ID}/archive": (200, {"session": {**SESSION, "live": None, "archived": True}}),
            f"POST /sessions/{SESSION_ID}/restore": (200, {"session": {**SESSION, "live": None}}),
            f"PUT /sessions/{SESSION_ID}/pin": (200, {"session": {**SESSION, "pinned": True}}),
            "GET /files": (200, {"files": [{"path": "src/app.py", "score": 0.9}]}),
        },
    )
    client = _client()

    assert client.post(f"/api/pi/sessions/{SESSION_ID}/archive").json()["session"]["archived"] is True
    assert client.post(f"/api/pi/sessions/{SESSION_ID}/restore").json()["session"]["live"] is None
    assert client.put(f"/api/pi/sessions/{SESSION_ID}/pin", json={"pinned": True}).json()["session"]["pinned"] is True
    assert client.get("/api/pi/files", params={"repo": "app"}).json() == {"files": [{"path": "src/app.py", "score": 0.9}]}
    assert json.loads(host.requests[2].content) == {"pinned": True}
    # K6: pi-host answers the whole cached list, the UI ranks it.
    assert dict(host.requests[3].url.params) == {"cwd": str(repositories / "app")}


def test_session_ids_outside_the_uuid_alphabet_are_refused(monkeypatch):
    host = _serve(monkeypatch, {})
    assert _client().post("/api/pi/sessions/..%2Fhealth/archive").status_code in (404, 422)
    assert host.requests == []


def test_logins_without_host_access_are_refused(monkeypatch):
    _serve(monkeypatch, {"GET /sessions": (200, {"sessions": []})})
    assert _client(has_host_access=False).get("/api/pi/sessions").status_code == 403


@dataclass
class FakeSocketHost:
    """A pi-host `/ws` stand-in on its own loop: echoes frames, closes on `"bye"`, records the upgrade headers and its own close."""

    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    closed: threading.Event = field(default_factory=threading.Event)

    async def handle(self, connection: ServerConnection) -> None:
        self.headers = {name.lower(): value for name, value in connection.request.headers.raw_items()}
        try:
            async for message in connection:
                if message == "bye":
                    return
                await connection.send(b"echo:" + message if isinstance(message, bytes) else f"echo:{message}")
        except ConnectionClosed:
            pass
        finally:
            self.closed.set()


@contextmanager
def fake_socket_host(monkeypatch) -> Iterator[FakeSocketHost]:
    fake = FakeSocketHost()
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    stop = loop.create_future()

    async def main() -> None:
        # A peer the test client dropped without a closing handshake must not hold the shutdown past the join below.
        async with serve(fake.handle, "127.0.0.1", 0, close_timeout=0.5) as server:
            port = next(iter(server.sockets)).getsockname()[1]
            fake.url = f"http://127.0.0.1:{port}"
            ready.set()
            await stop

    thread = threading.Thread(target=loop.run_until_complete, args=(main(),), daemon=True)
    thread.start()
    ready.wait(5)
    monkeypatch.setenv("GIGACHAD_PI_HOST_URL", fake.url)
    try:
        yield fake
    finally:
        loop.call_soon_threadsafe(stop.set_result, None)
        thread.join(10)
        loop.close()


@pytest.fixture
def trusted_loopback(monkeypatch):
    monkeypatch.setattr(websocket_origin, "CORS_ORIGINS", ["*"])


@pytest.mark.usefixtures("trusted_loopback")
def test_ws_relays_frames_both_ways_with_the_token_and_no_origin(monkeypatch):
    with fake_socket_host(monkeypatch) as fake, _client().websocket_connect("/api/pi/ws", headers={"origin": ORIGIN}) as socket:
        socket.send_text('{"type":"attach","sessionId":"x"}')
        assert socket.receive_text() == 'echo:{"type":"attach","sessionId":"x"}'
        socket.send_bytes(b"\x00\x01")
        assert socket.receive_bytes() == b"echo:\x00\x01"
        assert fake.headers["authorization"] == f"Bearer {TOKEN}"
        assert "origin" not in fake.headers


@pytest.mark.usefixtures("trusted_loopback")
def test_ws_closes_the_browser_when_pi_host_ends(monkeypatch):
    with fake_socket_host(monkeypatch), _client().websocket_connect("/api/pi/ws", headers={"origin": ORIGIN}) as socket:
        socket.send_text("bye")
        with pytest.raises(WebSocketDisconnect):
            socket.receive_text()


@pytest.mark.usefixtures("trusted_loopback")
def test_ws_closes_pi_host_when_the_browser_leaves(monkeypatch):
    # Leaving the TestClient context cancels the app at once, so the browser disconnects inside it and the close is awaited there.
    with fake_socket_host(monkeypatch) as fake, _client().websocket_connect("/api/pi/ws", headers={"origin": ORIGIN}) as socket:
        socket.send_text("ping")
        assert socket.receive_text() == "echo:ping"
        socket.close()
        assert fake.closed.wait(5)


@pytest.mark.usefixtures("trusted_loopback")
def test_ws_refuses_foreign_origins_and_logins_without_host_access(monkeypatch):
    # The handshake succeeds and the close carries the code: a close before accept would reach the browser as 1006.
    with fake_socket_host(monkeypatch) as fake:
        for client, origin in ((_client(), "https://evil.example"), (_client(has_host_access=False), ORIGIN)):
            with (
                client.websocket_connect("/api/pi/ws", headers={"origin": origin}) as socket,
                pytest.raises(WebSocketDisconnect) as refused,
            ):
                socket.receive_text()
            assert refused.value.code == websocket_origin.POLICY_VIOLATION
        assert fake.headers == {}


@pytest.mark.usefixtures("trusted_loopback")
def test_ws_answers_try_again_later_when_pi_host_is_down(monkeypatch):
    monkeypatch.setenv("GIGACHAD_PI_HOST_URL", "http://127.0.0.1:9")
    with (
        _client().websocket_connect("/api/pi/ws", headers={"origin": ORIGIN}) as socket,
        pytest.raises(WebSocketDisconnect) as refused,
    ):
        socket.receive_text()
    assert refused.value.code == 1013


def test_auto_resume_setting_passes_through_for_host_logins_only(monkeypatch):
    host = _serve(monkeypatch, {"GET /settings": (200, {"autoResume": False}), "PUT /settings": (200, {"autoResume": True})})

    assert _client().get("/api/pi/settings").json() == {"autoResume": False}
    assert _client().put("/api/pi/settings", json={"autoResume": True}).json() == {"autoResume": True}
    assert json.loads(host.requests[1].content) == {"autoResume": True}
    assert _client(has_host_access=False).get("/api/pi/settings").status_code == 403
    assert len(host.requests) == 2


def test_auto_resume_setting_is_unavailable_without_pi_host(monkeypatch):
    _serve(monkeypatch, {})

    assert _client().get("/api/pi/settings").status_code == 503


TURN_DELETE = f"POST /sessions/{SESSION_ID}/turns/ab12cd34/delete"
TURN_URL = f"/api/pi/sessions/{SESSION_ID}/turns/ab12cd34/delete"
TRUSTED = {"origin": "http://localhost:2999"}


def test_a_turn_delete_passes_through_and_drops_the_thumbnails_of_removed_images(monkeypatch):
    image = "a" * 64
    pi_media._cache[image] = b"thumb"
    pi_media._cache_bytes = 5
    host = _serve(monkeypatch, {TURN_DELETE: (200, {"ok": True, "removedEntries": 4, "removedMedia": [image]})})

    response = _client().post(TURN_URL, json={"through": "end"}, headers=TRUSTED)

    assert response.status_code == 200
    assert response.json() == {"removedEntries": 4, "removedMedia": [image], "notRemoved": []}
    assert json.loads(host.requests[0].content) == {"through": "end"}
    assert pi_media.thumbnail_for(image) is None


def test_a_turn_delete_names_what_its_cleanup_left_on_disk(monkeypatch):
    paste = ".pi/pastes/ab.txt"
    _serve(monkeypatch, {TURN_DELETE: (200, {"ok": True, "removedEntries": 2, "removedMedia": [], "notRemoved": [paste]})})

    response = _client().post(TURN_URL, json={"through": "turn"}, headers=TRUSTED)

    assert response.status_code == 200
    assert response.json() == {"removedEntries": 2, "removedMedia": [], "notRemoved": [paste]}


@pytest.mark.parametrize("reason", ["running", "compacted", "locked", "changed"])
def test_a_turn_delete_refusal_keeps_its_status_and_reason(monkeypatch, reason):
    _serve(monkeypatch, {TURN_DELETE: (409, {"error": reason})})

    response = _client().post(TURN_URL, json={"through": "turn"}, headers=TRUSTED)

    assert response.status_code == 409
    assert response.json() == {"detail": reason, "body": {"error": reason}}


def test_a_turn_delete_from_an_untrusted_request_or_without_host_access_is_refused(monkeypatch):
    host = _serve(monkeypatch, {TURN_DELETE: (200, {"ok": True, "removedEntries": 1, "removedMedia": []})})

    assert _client().post(TURN_URL, json={"through": "turn"}, headers={"origin": "https://evil.example"}).status_code == 403
    assert _client().post(TURN_URL, json={"through": "turn"}).status_code == 403
    assert _client(has_host_access=False).post(TURN_URL, json={"through": "turn"}, headers=TRUSTED).status_code == 403
    assert _client().post(TURN_URL, json={"through": "all"}, headers=TRUSTED).status_code == 422
    assert host.requests == []


def test_opening_a_session_with_resume_continue_is_refused(monkeypatch, repositories):
    _serve(monkeypatch, {})
    assert _client().post("/api/pi/sessions", json={"repo": "app", "resume": "continue"}).status_code == 422
