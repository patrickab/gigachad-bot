"""The preview gate lets only host logins through and reaches each preview's frontend and backend by its port in the path."""

import asyncio
import threading

import httpx
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from websockets.asyncio.server import ServerConnection, serve

from backend import preview_gate
from backend.identity import HOST_CAPABILITY
from lib import factory

OWNER = "owner@example.com"
GRANTED = f'{{"{HOST_CAPABILITY}":[{{}}]}}'
HOST = "gigachad-backend.tail.ts.net:8445"
TAILNET = {"host": HOST, "origin": f"https://{HOST}", "tailscale-user-login": OWNER, "tailscale-app-capabilities": GRANTED}


class _StreamedBody(httpx.AsyncByteStream):
    """A body a preview streams, the way the gate reads it (`aiter_raw`)."""

    async def __aiter__(self):
        yield b"ok"


@pytest.fixture
def upstream(monkeypatch) -> list[httpx.Request]:
    """Record what reaches the previews."""
    monkeypatch.delenv("GIGACHAD_ENV", raising=False)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, stream=_StreamedBody())

    monkeypatch.setattr(preview_gate, "_transport", httpx.MockTransport(handler))
    return seen


def _request(method: str, path: str, headers: dict[str, str], **kwargs) -> httpx.Response:
    with TestClient(preview_gate.app) as client:
        return client.request(method, path, headers=headers, **kwargs)


@pytest.mark.parametrize("dropped", ["tailscale-user-login", "tailscale-app-capabilities"])
def test_refuses_tailnet_requests_without_a_host_login(upstream, dropped):
    response = _request("GET", "/preview/8101/", {name: value for name, value in TAILNET.items() if name != dropped})

    assert response.status_code == 403
    assert upstream == []


def test_lets_the_local_dev_login_through_without_tailnet_headers(upstream, monkeypatch):
    monkeypatch.setenv("GIGACHAD_ENV", "development")
    monkeypatch.setenv("GIGACHAD_DEV_USER", "dev")

    assert _request("GET", "/preview/8101/", {}).status_code == 200


def test_frontend_paths_keep_their_base_path_and_reach_the_preview_as_its_own_origin(upstream):
    response = _request("GET", "/preview/8101/_next/a%2Fb.js?v=1", TAILNET)

    assert response.text == "ok"
    forwarded = upstream[0]
    assert forwarded.url.netloc == b"127.0.0.1:8101"
    assert forwarded.url.raw_path == b"/preview/8101/_next/a%2Fb.js?v=1"
    assert forwarded.headers["host"] == "127.0.0.1:8101"
    assert forwarded.headers["origin"] == "http://127.0.0.1:8101"
    assert "tailscale-user-login" not in forwarded.headers
    assert "tailscale-app-capabilities" not in forwarded.headers


def test_backend_paths_lose_their_prefix(upstream):
    _request("POST", "/preview-api/8100/api/chats?x=1", TAILNET, json={"title": "Fourier"})

    forwarded = upstream[0]
    assert forwarded.url.netloc == b"127.0.0.1:8100"
    assert forwarded.url.raw_path == b"/api/chats?x=1"
    assert forwarded.content == b'{"title":"Fourier"}'


@pytest.mark.parametrize("path", ["/", "/preview/8001/api/config", "/preview-api/22/", "/preview/8101x/", "/api/config"])
def test_reaches_nothing_but_preview_ports(upstream, path):
    response = _request("GET", path, TAILNET)

    assert response.status_code == 404
    assert upstream == []


def test_reports_a_stopped_preview_as_bad_gateway(monkeypatch):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.delenv("GIGACHAD_ENV", raising=False)
    monkeypatch.setattr(preview_gate, "_transport", httpx.MockTransport(refuse))

    assert _request("GET", "/preview/8101/", TAILNET).status_code == 502


@pytest.fixture
def echo_preview(monkeypatch):
    """A preview backend whose WebSocket answers each message with its path, origin and the message. Yields its port."""
    monkeypatch.delenv("GIGACHAD_ENV", raising=False)
    loop = asyncio.new_event_loop()
    stopped = asyncio.Event()
    started: list[int] = []
    ready = threading.Event()

    async def echo(connection: ServerConnection) -> None:
        request = connection.request
        async for message in connection:
            await connection.send(f"{request.path} {request.headers['origin']} {message}")

    async def run() -> None:
        async with serve(echo, "127.0.0.1", 0) as server:
            started.append(server.sockets[0].getsockname()[1])
            ready.set()
            await stopped.wait()

    thread = threading.Thread(target=loop.run_until_complete, args=(run(),), daemon=True)
    thread.start()
    ready.wait(5)
    monkeypatch.setattr(preview_gate, "PREVIEW_PORTS", range(started[0], started[0] + 1))
    yield started[0]
    loop.call_soon_threadsafe(stopped.set)
    thread.join(5)


def test_relays_websockets_both_ways(echo_preview):
    with (
        TestClient(preview_gate.app) as client,
        client.websocket_connect(f"/preview-api/{echo_preview}/api/terminal/session?x=1", headers=TAILNET) as websocket,
    ):
        websocket.send_text("hello")

        assert websocket.receive_text() == f"/api/terminal/session?x=1 http://127.0.0.1:{echo_preview} hello"


def test_refuses_websockets_without_a_host_login(echo_preview):
    with (
        TestClient(preview_gate.app) as client,
        pytest.raises(WebSocketDisconnect) as refused,
        client.websocket_connect(f"/preview-api/{echo_preview}/api/terminal/session", headers={"host": HOST}),
    ):
        pass

    assert refused.value.code == 1008


def test_the_engines_preview_urls_move_onto_the_gate(monkeypatch):
    monkeypatch.setenv("GIGACHAD_PREVIEW_URL", f"https://{HOST}/")

    preview = factory.Preview(
        status="ready", url="http://127.0.0.1:8101/preview/8101", open_url="http://127.0.0.1:8101/preview/8101/?open=abc", error=None
    )

    assert preview.url == f"https://{HOST}/preview/8101"
    assert preview.open_url == f"https://{HOST}/preview/8101/?open=abc"
    # Not a preview port, and a preview that never got a URL.
    assert factory.through_gate("http://127.0.0.1:8001/api") == "http://127.0.0.1:8001/api"
    assert factory.through_gate("") == ""
