"""Reverse-proxy plumbing shared by the Tailnet gates (omp-web, LearnHarness, factory previews) and the Jupyter proxy route.

- `streaming_response` relays an upstream answer unbuffered, hop-by-hop fields dropped (`lib.upstream.forwardable_headers`)
- `forward` is the whole request path of the omp-web and LearnHarness gates, standalone Starlette apps whose
  `app.state.client` has the upstream as `base_url`
- `relay_websocket` joins a browser WebSocket to an upstream one the caller opened
"""

import asyncio
from collections.abc import Callable
import contextlib

import httpx
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response, StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

from lib.upstream import forwardable_headers


async def _browser_to_upstream(websocket: WebSocket, upstream: ClientConnection) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        data = message.get("bytes")
        await upstream.send(data if data is not None else message.get("text") or "")


async def _upstream_to_browser(websocket: WebSocket, upstream: ClientConnection) -> None:
    async for message in upstream:
        if isinstance(message, bytes):
            await websocket.send_bytes(message)
        else:
            await websocket.send_text(message)


async def relay_websocket(websocket: WebSocket, upstream: ClientConnection) -> None:
    """Accept *websocket* with the subprotocol the open *upstream* chose and relay both ways until either side closes, then close both."""
    try:
        await websocket.accept(subprotocol=upstream.subprotocol)
        relays = [
            asyncio.create_task(_browser_to_upstream(websocket, upstream)),
            asyncio.create_task(_upstream_to_browser(websocket, upstream)),
        ]
        try:
            await asyncio.wait(relays, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for relay in relays:
                relay.cancel()
            await asyncio.gather(*relays, return_exceptions=True)
    except (WebSocketDisconnect, ConnectionClosed, RuntimeError):
        pass
    finally:
        await upstream.close()
        if websocket.client_state == WebSocketState.CONNECTED:
            with contextlib.suppress(RuntimeError):
                await websocket.close()


def streaming_response(upstream: httpx.Response, drop: Callable[[str], bool] | None = None) -> StreamingResponse:
    """Relay the open streaming *upstream* answer, closed when the response is done. *drop* refuses more lowercase header names."""
    response = StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code, background=BackgroundTask(upstream.aclose))
    # Raw headers keep repeated fields (several `set-cookie`) that a mapping would merge.
    response.raw_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in forwardable_headers(upstream.headers.multi_items())
        if drop is None or not drop(name.lower())
    ]
    return response


def _gate_headers(request: Request) -> list[tuple[str, str]]:
    """Request headers for the upstream behind a Tailnet gate: the browser's `Host` kept, the scheme marked HTTPS."""
    return [
        *forwardable_headers(request.headers.items(), frozenset({"content-length", "x-forwarded-proto"})),
        ("x-forwarded-proto", "https"),
    ]


def _gate_target(request: Request) -> str:
    """Path and query exactly as the browser sent them, so encoded segments (`%2F` in file paths) stay encoded."""
    raw_path = request.scope.get("raw_path") or request.url.path.encode()
    query = request.scope.get("query_string", b"")
    return (raw_path + (b"?" + query if query else b"")).decode("latin-1")


async def forward(request: Request, label: str) -> Response:
    """Stream *request* to `request.app.state.client` and its answer back, 502 naming *label* when the upstream does not answer."""
    client: httpx.AsyncClient = request.app.state.client
    upstream_request = client.build_request(
        request.method, _gate_target(request), headers=_gate_headers(request), content=request.stream()
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return PlainTextResponse(f"{label} does not answer: {exc}\n", status_code=502)
    return streaming_response(upstream)
