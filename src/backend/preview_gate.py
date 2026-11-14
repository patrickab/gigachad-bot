"""Identity gate in front of every factory preview: one port for all of them, picked by the URL path.

- Consumes the previews the factory engine starts on loopback ports `lib.factory.PREVIEW_PORTS`, provides them to the browser
  at `lib.factory.preview_gate_url()` (Tailscale Serve terminates HTTPS in production)
- `/preview/<port>/...` reaches the preview's frontend with the path kept, which Next serves under that base path.
  `/preview-api/<port>/...` reaches its backend with the prefix stripped
- A preview runs the task agent's code as the host user: only the local dev login and Tailnet logins granted
  `HOST_CAPABILITY` pass, everyone else gets 403
- An origin of its own, so preview code never shares one with the real backend API
- The upstream sees its own loopback `Host` and `Origin` (Next's dev server and the backend's WebSocket routes refuse
  others) and no Tailnet identity headers. Streams both ways, WebSockets included (hot reload, terminals)

Run: `uvicorn backend.preview_gate:app --app-dir src --host 127.0.0.1 --port 30185`
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import re

import httpx
from starlette.applications import Starlette
from starlette.requests import HTTPConnection, Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidHandshake

from backend.identity import CAPABILITIES_HEADER, development_login, has_host_access, host_denied
from backend.omp_web_gate import LOGIN_HEADER, METHODS
from backend.reverse_proxy import relay_websocket, streaming_response
from lib.factory import PREVIEW_PORTS
from lib.upstream import forwardable_headers

# `preview` keeps the whole path for the upstream, `preview-api` strips its prefix.
_ROUTE = re.compile(rb"^/(preview|preview-api)/(\d{1,5})(/.*)?$")
_REPLACED_HEADERS = frozenset({"content-length", "host", "origin", LOGIN_HEADER, CAPABILITIES_HEADER.lower()})
# Handshake fields the WebSocket client writes itself.
_WEBSOCKET_HANDSHAKE_HEADERS = frozenset(
    {"sec-websocket-key", "sec-websocket-version", "sec-websocket-extensions", "sec-websocket-protocol"}
)
_POLICY_VIOLATION = 1008
_UPSTREAM_ERROR = 1011

# Tests swap in an `httpx.MockTransport`, None is the real network.
_transport: httpx.AsyncBaseTransport | None = None


def is_allowed(connection: HTTPConnection) -> bool:
    """Whether the request comes from the local dev login or a Tailnet login holding the host capability."""
    login = connection.headers.get(LOGIN_HEADER) or development_login()
    return bool(login) and has_host_access(login, connection.headers.get(CAPABILITIES_HEADER))


def upstream_target(connection: HTTPConnection) -> tuple[str, str] | None:
    """Loopback authority (`127.0.0.1:<port>`) and request target of the preview a path names, None for any other path.

    Examples
    --------
    `/preview/8101/_next/x.js?v=1` -> `("127.0.0.1:8101", "/preview/8101/_next/x.js?v=1")`
    `/preview-api/8100/api/config` -> `("127.0.0.1:8100", "/api/config")`
    """
    raw_path = connection.scope.get("raw_path") or connection.url.path.encode()
    match = _ROUTE.match(raw_path)
    if match is None or int(match[2]) not in PREVIEW_PORTS:
        return None
    path = raw_path if match[1] == b"preview" else match[3] or b"/"
    query = connection.scope.get("query_string", b"")
    return f"127.0.0.1:{int(match[2])}", (path + (b"?" + query if query else b"")).decode("latin-1")


def _upstream_headers(connection: HTTPConnection, authority: str, dropped: frozenset[str] = frozenset()) -> list[tuple[str, str]]:
    """The browser's headers with `Host` and `Origin` set to the upstream's own."""
    headers = [*forwardable_headers(connection.headers.items(), _REPLACED_HEADERS | dropped), ("host", authority)]
    if "origin" in connection.headers:
        headers.append(("origin", f"http://{authority}"))
    return headers


async def proxy(request: Request) -> Response:
    if not is_allowed(request):
        return PlainTextResponse(f"{host_denied('Factory previews')}\n", status_code=403)
    target = upstream_target(request)
    if target is None:
        return PlainTextResponse("No preview at this path\n", status_code=404)
    authority, path = target
    client: httpx.AsyncClient = request.app.state.client
    upstream_request = client.build_request(
        request.method, f"http://{authority}{path}", headers=_upstream_headers(request, authority), content=request.stream()
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return PlainTextResponse(f"The preview does not answer: {exc}\n", status_code=502)
    return streaming_response(upstream)


async def proxy_websocket(websocket: WebSocket) -> None:
    target = upstream_target(websocket)
    if not is_allowed(websocket) or target is None:
        await websocket.close(code=_POLICY_VIOLATION)
        return
    authority, path = target
    try:
        upstream = await connect(
            f"ws://{authority}{path}",
            additional_headers=_upstream_headers(websocket, authority, _WEBSOCKET_HANDSHAKE_HEADERS),
            subprotocols=websocket.scope["subprotocols"] or None,
            max_size=None,
        )
    except (OSError, InvalidHandshake):
        await websocket.close(code=_UPSTREAM_ERROR)
        return
    await relay_websocket(websocket, upstream)


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    # No read timeout: a preview streams chat answers and compiles pages on first request.
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None), transport=_transport) as client:
        app.state.client = client
        yield


app = Starlette(
    routes=[Route("/{path:path}", proxy, methods=METHODS), WebSocketRoute("/{path:path}", proxy_websocket)],
    lifespan=lifespan,
)
