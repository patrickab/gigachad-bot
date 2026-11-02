"""Tailnet identity gate in front of LearnHarness, for browsers on other devices.

- Same contract as `omp_web_gate` (whose login check it reuses, `reverse_proxy.forward` does the proxying): Tailscale Serve
  terminates HTTPS and vouches for the login, only logins granted `HOST_CAPABILITY` pass, everyone else gets 403
- A plain identity-checking proxy: keeps `Host`, marks the request HTTPS, streams both ways (LearnHarness streams its turns).
  No LearnHarness knowledge lives here, `lib/learnharness.py` owns that, except one policy: LearnHarness takes no provider
  key, custom endpoint or subscription (models come from gigachad-bot through the LiteLLM proxy), so writes to those
  routes are refused with 403. The overlay hides their UI, this keeps a hand-made request out too

Run: `uvicorn backend.learnharness_gate:app --app-dir src --host 127.0.0.1 --port 30182`
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import posixpath

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

from backend.identity import host_denied
from backend.omp_web_gate import METHODS, is_allowed
from backend.reverse_proxy import forward
from lib.learnharness import learnharness_url

# Tests swap in an `httpx.MockTransport`, None is the real network.
_transport: httpx.AsyncBaseTransport | None = None

# Read-only for the browser: provider keys (`PUT`/`DELETE /api/keys/<id>`), the custom endpoint (`POST`/`PATCH`/`DELETE`, the
# proxy's base URL and key binding) and subscription sign-in (`POST /api/subscriptions`).
_CREDENTIAL_ROUTES = ("/api/keys", "/api/custom-provider", "/api/subscriptions")
# Subscription sign-in's browser step, every method of it.
_SIGN_IN_ROUTE = "/api/subscriptions/browser"
_READ_METHODS = frozenset({"GET", "HEAD"})
_CREDENTIALS_REFUSED = "LearnHarness takes no provider keys or subscriptions: gigachad-bot provides its models"


def _under(route: str, prefix: str) -> bool:
    return route == prefix or route.startswith(prefix + "/")


def is_credential_write(method: str, path: str) -> bool:
    """Whether *method* on the decoded *path* would change LearnHarness's provider keys, custom endpoint or subscriptions."""
    # Collapse `//`, `.`, `..` and a trailing slash, ignore case: no spelling of the route slips through.
    route = posixpath.normpath("/" + path.lstrip("/")).lower()
    if _under(route, _SIGN_IN_ROUTE):
        return True
    return method.upper() not in _READ_METHODS and any(_under(route, prefix) for prefix in _CREDENTIAL_ROUTES)


async def proxy(request: Request) -> Response:
    if not is_allowed(request):
        return PlainTextResponse(f"{host_denied('LearnHarness')}\n", status_code=403)
    if is_credential_write(request.method, request.url.path):
        return PlainTextResponse(f"{_CREDENTIALS_REFUSED}\n", status_code=403)
    return await forward(request, "LearnHarness")


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    # No read timeout: a learning turn streams for as long as the model works.
    async with httpx.AsyncClient(base_url=learnharness_url(), timeout=httpx.Timeout(10.0, read=None), transport=_transport) as client:
        app.state.client = client
        yield


app = Starlette(routes=[Route("/{path:path}", proxy, methods=METHODS)], lifespan=lifespan)
