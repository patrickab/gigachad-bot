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
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

from backend.identity import CAPABILITIES_HEADER, host_denied, is_host_capability_granted
from backend.reverse_proxy import forward
from lib.omp_web import omp_web_url

LOGIN_HEADER = "tailscale-user-login"
METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

# Tests swap in an `httpx.MockTransport`, None is the real network.
_transport: httpx.AsyncBaseTransport | None = None


def is_allowed(request: Request) -> bool:
    """Whether Tailscale Serve vouched for a login holding the host capability."""
    return bool(request.headers.get(LOGIN_HEADER)) and is_host_capability_granted(request.headers.get(CAPABILITIES_HEADER))


async def proxy(request: Request) -> Response:
    if not is_allowed(request):
        return PlainTextResponse(f"{host_denied('omp-web')}\n", status_code=403)
    return await forward(request, "omp-web")


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    # No read timeout: SSE streams stay open for a whole agent run.
    async with httpx.AsyncClient(base_url=omp_web_url(), timeout=httpx.Timeout(10.0, read=None), transport=_transport) as client:
        app.state.client = client
        yield


app = Starlette(routes=[Route("/{path:path}", proxy, methods=METHODS)], lifespan=lifespan)
