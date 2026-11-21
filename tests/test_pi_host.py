"""pi-host answers reach the browser unchanged, a missing host, a changed shape or a refusal is reported, and the WS proxy relays."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
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
from lib import pi_host
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
    assert response.json() == {"sessions": [{**SESSION, "parentSession": None, "profile": "coding", "factory": None}]}
    assert dict(host.requests[0].url.params) == {"cwd": str(repositories / "app"), "archived": "1"}


def test_sessions_without_a_repo_list_every_cwd(monkeypatch):
    host = _serve(monkeypatch, {"GET /sessions": (200, {"sessions": [SESSION, {**SESSION, "cwd": "/home/me/git/other"}]})})

    response = _client().get("/api/pi/sessions")

    assert response.status_code == 200, response.text
    assert [session["cwd"] for session in response.json()["sessions"]] == ["/home/me/git/app", "/home/me/git/other"]
    assert dict(host.requests[0].url.params) == {"archived": "0"}


def test_search_passes_hits_through_with_the_query_and_limit(monkeypatch):
    hit = {"session": SESSION, "snippet": "…fix the tests…", "entryId": "a1b2c3d4"}
    host = _serve(monkeypatch, {"GET /search": (200, {"hits": [hit]})})

    response = _client().get("/api/pi/search", params={"q": "fix tests"})

    assert response.status_code == 200, response.text
    assert response.json() == {"hits": [{**hit, "session": {**SESSION, "parentSession": None, "profile": "coding", "factory": None}}]}
    assert dict(host.requests[0].url.params) == {"q": "fix tests", "limit": "20"}
    assert _client().get("/api/pi/search").status_code == 422


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


async def test_a_prompt_reaches_the_session_and_answers_its_disposition(monkeypatch):
    host = _serve(monkeypatch, {f"POST /sessions/{SESSION_ID}/prompt": (200, {"disposition": "handled"})})

    assert await pi_host.prompt_session(SESSION_ID, "/factory approve") == "handled"
    assert await pi_host.prompt_session(SESSION_ID, "Also the icon", "followUp") == "handled"

    assert json.loads(host.requests[0].content) == {"message": "/factory approve"}
    assert json.loads(host.requests[1].content) == {"message": "Also the icon", "streamingBehavior": "followUp"}


async def test_a_prompt_to_a_stopped_session_is_refused_with_pi_hosts_status(monkeypatch):
    _serve(monkeypatch, {f"POST /sessions/{SESSION_ID}/prompt": (409, {"error": "not_live"})})

    with pytest.raises(pi_host.PiHostRejected) as refused:
        await pi_host.prompt_session(SESSION_ID, "/factory approve")

    assert (refused.value.status, refused.value.body) == (409, {"error": "not_live"})


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
        "/w/markdown", pi_host.ResumeById(session_id=SESSION_ID), profile="document", system_prompt="Edit it.", tools=["read"],
    )

    assert session.profile == "document"
    assert json.loads(host.requests[0].content) == {
        "cwd": "/w/markdown", "profile": "document", "resume": {"sessionId": SESSION_ID},
        "systemPrompt": "Edit it.", "tools": ["read"],
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
        assert client.get("/api/pi/files", params={"cwd": str(folder), "q": "n"}).json() == found
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

    assert _client().post("/api/pi/sessions", json={"repo": "notes", "resume": None}).status_code == 404
    assert _client().get("/api/pi/files", params={"repo": "../etc"}).status_code == 404
    assert host.requests == []


def test_repositories_are_listed(repositories):
    assert _client().get("/api/pi/repos").json() == [{"name": "app", "path": str(repositories / "app")}]


def test_pi_host_down_answers_503(monkeypatch):
    _serve(monkeypatch, {})

    response = _client().get("/api/pi/sessions")

    assert response.status_code == 503
    assert "pi-host" in response.json()["detail"]


def test_a_changed_shape_answers_502_naming_the_field(monkeypatch):
    broken = {key: value for key, value in SESSION.items() if key != "messageCount"}
    _serve(monkeypatch, {"GET /sessions": (200, {"sessions": [broken]})})

    response = _client().get("/api/pi/sessions")

    assert response.status_code == 502
    assert "`sessions.0.messageCount`" in response.json()["detail"]


def test_a_refusal_keeps_its_status_and_body(monkeypatch, repositories):
    current = {"revision": 3, "data": {"title": "newer"}}
    _serve(monkeypatch, {"POST /sessions": (409, {"error": "locked"}), f"PUT /sessions/{SESSION_ID}/state": (409, current)})
    client = _client()

    locked = client.post("/api/pi/sessions", json={"repo": "app", "resume": "continue"})
    stale = client.put(f"/api/pi/sessions/{SESSION_ID}/state", json={"revision": 2, "data": {"title": "mine"}})

    assert (locked.status_code, locked.json()) == (409, {"detail": "locked", "body": {"error": "locked"}})
    assert stale.status_code == 409
    assert stale.json()["body"] == current


def test_session_actions_reach_their_pi_host_routes(monkeypatch, repositories):
    host = _serve(
        monkeypatch,
        {
            f"POST /sessions/{SESSION_ID}/close": (204, None),
            f"POST /sessions/{SESSION_ID}/archive": (200, {"session": {**SESSION, "live": None, "archived": True}}),
            f"POST /sessions/{SESSION_ID}/restore": (200, {"session": {**SESSION, "live": None}}),
            f"PUT /sessions/{SESSION_ID}/pin": (200, {"session": {**SESSION, "pinned": True}}),
            f"GET /sessions/{SESSION_ID}/state": (200, {"revision": 1, "data": {}}),
            f"PUT /sessions/{SESSION_ID}/state": (200, {"revision": 2}),
            "GET /files": (200, {"files": [{"path": "src/app.py", "score": 0.9}]}),
        },
    )
    client = _client()

    assert client.post(f"/api/pi/sessions/{SESSION_ID}/close").status_code == 204
    assert client.post(f"/api/pi/sessions/{SESSION_ID}/archive").json()["session"]["archived"] is True
    assert client.post(f"/api/pi/sessions/{SESSION_ID}/restore").json()["session"]["live"] is None
    assert client.put(f"/api/pi/sessions/{SESSION_ID}/pin", json={"pinned": True}).json()["session"]["pinned"] is True
    assert client.get(f"/api/pi/sessions/{SESSION_ID}/state").json() == {"revision": 1, "data": {}}
    assert client.put(f"/api/pi/sessions/{SESSION_ID}/state", json={"revision": 1, "data": {"a": 1}}).json() == {"revision": 2}
    assert client.get("/api/pi/files", params={"repo": "app", "q": "app"}).json() == {"files": [{"path": "src/app.py", "score": 0.9}]}
    assert client.get("/api/pi/files", params={"repo": "app"}).status_code == 200
    assert json.loads(host.requests[3].content) == {"pinned": True}
    assert json.loads(host.requests[5].content) == {"revision": 1, "data": {"a": 1}}
    assert dict(host.requests[6].url.params) == {"cwd": str(repositories / "app"), "q": "app"}
    # K6: without a query pi-host answers the whole cached list, the UI ranks it.
    assert dict(host.requests[7].url.params) == {"cwd": str(repositories / "app"), "q": ""}


def test_session_ids_outside_the_uuid_alphabet_are_refused(monkeypatch):
    host = _serve(monkeypatch, {})
    assert _client().post("/api/pi/sessions/..%2Fhealth/close").status_code in (404, 422)
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
