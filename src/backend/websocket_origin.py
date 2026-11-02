"""Which browser origins may reach the host (terminals, file watching, Jupyter kernels): WebSocket upgrades and changing requests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from fastapi import HTTPException, Request

from config import CORS_ORIGINS

if TYPE_CHECKING:
    from fastapi import WebSocket

    from backend.identity import RequestIdentity

# Close code of a refused upgrade: the origin or the login may not open the socket.
POLICY_VIOLATION = 1008
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def is_trusted_origin(origin: str | None) -> bool:
    """Allow-listed origins, or loopback ones while CORS is the `*` default.

    CORS does not cover WebSockets, and a terminal is code execution on this host,
    so any page a user visits must not be able to open one.
    """
    if not origin:
        return False
    if "*" not in CORS_ORIGINS:
        return origin in CORS_ORIGINS
    return urlsplit(origin).hostname in _LOOPBACK_HOSTS


def is_allowed_websocket(websocket: WebSocket, identity: RequestIdentity | None = None) -> bool:
    """Whether *websocket* may be accepted: its `Origin` is trusted and, when given, the *identity* holds host access."""
    return is_trusted_origin(websocket.headers.get("origin")) and (identity is None or identity.has_host_access)


def is_trusted_request(request: Request) -> bool:
    """Whether *request* comes from the app's own pages: a trusted `Origin`, or for a read without one a request no other site sent."""
    # Identity comes from the tailnet, not a cookie, so any page the user opens carries their login. CORS stops such a page from reading
    # the answer, not from sending it: a change needs a trusted `Origin` (browsers send one on every POST, PATCH and DELETE), a read
    # without `Origin` (a plain `<img src>`) must not be flagged cross-site by `Sec-Fetch-Site`.
    origin = request.headers.get("origin")
    if origin is not None:
        return is_trusted_origin(origin)
    return request.method in {"GET", "HEAD"} and request.headers.get("sec-fetch-site", "same-origin") in {"same-origin", "none"}


def require_trusted_request(request: Request) -> None:
    """Dependency answering 403 to a request that is not `is_trusted_request`."""
    if not is_trusted_request(request):
        raise HTTPException(status_code=403, detail="Cross-site request refused")
