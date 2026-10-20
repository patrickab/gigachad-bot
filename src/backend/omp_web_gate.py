"""Tailnet identity gate in front of omp-web, for browsers on other devices.

- Tailscale Serve terminates HTTPS, injects `Tailscale-User-Login` and `Tailscale-App-Capabilities` and proxies here (loopback only)
- Forwards only logins granted `HOST_CAPABILITY` in the tailnet policy: omp-web runs shell commands as the host user, no grant refuses everyone
- Keeps `Host` and marks the request HTTPS, so omp-web's same-origin API guard compares the browser's own origin
- Streams both ways, omp-web's live chat is a long-lived SSE response

Run: `uvicorn backend.omp_web_gate:app --app-dir src --host 127.0.0.1 --port 30179`
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

from backend.identity import CAPABILITIES_HEADER, HOST_CAPABILITY, is_host_capability_granted
from lib.omp_web import omp_web_url

LOGIN_HEADER = "tailscale-user-login"
# RFC 9110 connection-specific fields. httpx sets `content-length` for the request itself.
HOP_BY_HOP_HEADERS = frozenset(
    {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"}
)
METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

# Tests swap in an `httpx.MockTransport`, None is the real network.
_transport: httpx.AsyncBaseTransport | None = None


def is_allowed(request: Request) -> bool:
    """Whether Tailscale Serve vouched for a login holding the host capability."""
    return bool(request.headers.get(LOGIN_HEADER)) and is_host_capability_granted(request.headers.get(CAPABILITIES_HEADER))


def upstream_headers(request: Request) -> list[tuple[str, str]]:
    """Request headers for omp-web: hop-by-hop fields dropped, the browser's `Host` kept, scheme marked HTTPS."""
    dropped = HOP_BY_HOP_HEADERS | {"content-length", "x-forwarded-proto"}
    return [*((name, value) for name, value in request.headers.items() if name not in dropped), ("x-forwarded-proto", "https")]


def upstream_target(request: Request) -> str:
    """Path and query exactly as the browser sent them, so encoded segments (`%2F` in file paths) stay encoded."""
    raw_path = request.scope.get("raw_path") or request.url.path.encode()
    query = request.scope.get("query_string", b"")
    return (raw_path + (b"?" + query if query else b"")).decode("latin-1")


async def proxy(request: Request) -> Response:
    if not is_allowed(request):
        return PlainTextResponse(f"omp-web is limited to logins granted {HOST_CAPABILITY}\n", status_code=403)
    client: httpx.AsyncClient = request.app.state.client
    upstream_request = client.build_request(
        request.method, upstream_target(request), headers=upstream_headers(request), content=request.stream()
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return PlainTextResponse(f"omp-web does not answer: {exc}\n", status_code=502)

    async def body() -> AsyncIterator[bytes]:
        async for chunk in upstream.aiter_raw():
            yield chunk

    response = StreamingResponse(body(), status_code=upstream.status_code, background=BackgroundTask(upstream.aclose))
    # Raw headers keep repeated fields (several `set-cookie`) that a mapping would merge.
    response.raw_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in upstream.headers.multi_items()
        if name.lower() not in HOP_BY_HOP_HEADERS
    ]
    return response


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    # No read timeout: SSE streams stay open for a whole agent run.
    async with httpx.AsyncClient(base_url=omp_web_url(), timeout=httpx.Timeout(10.0, read=None), transport=_transport) as client:
        app.state.client = client
        yield


app = Starlette(routes=[Route("/{path:path}", proxy, methods=METHODS)], lifespan=lifespan)
