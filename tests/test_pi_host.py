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
from backend.routes.pi import router
from lib import pi_host
from tests.conftest import request_identity, route_client

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


def _client(has_host_access: bool = True) -> TestClient:
    return route_client(router, overrides={get_request_identity: request_identity(has_host_access)})


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


def test_sessions_pass_through_camel_case_with_the_repo_resolved(monkeypatch, repositories):
    host = _serve(monkeypatch, {"GET /sessions": (200, {"sessions": [SESSION]})})

    response = _client().get("/api/pi/sessions", params={"repo": "app", "archived": "true"})

    assert response.status_code == 200, response.text
    assert response.json() == {"sessions": [{**SESSION, "parentSession": None}]}
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
    assert response.json() == {"hits": [{**hit, "session": {**SESSION, "parentSession": None}}]}
    assert dict(host.requests[0].url.params) == {"q": "fix tests", "limit": "20"}
    assert _client().get("/api/pi/search").status_code == 422


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
            with client.websocket_connect("/api/pi/ws", headers={"origin": origin}) as socket:
                with pytest.raises(WebSocketDisconnect) as refused:
                    socket.receive_text()
            assert refused.value.code == websocket_origin.POLICY_VIOLATION
        assert fake.headers == {}


@pytest.mark.usefixtures("trusted_loopback")
def test_ws_answers_try_again_later_when_pi_host_is_down(monkeypatch):
    monkeypatch.setenv("GIGACHAD_PI_HOST_URL", "http://127.0.0.1:9")
    with _client().websocket_connect("/api/pi/ws", headers={"origin": ORIGIN}) as socket:
        with pytest.raises(WebSocketDisconnect) as refused:
            socket.receive_text()
    assert refused.value.code == 1013
