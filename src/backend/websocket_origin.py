"""Which browser origins may open the WebSockets that reach the host (terminals, file watching)."""

from __future__ import annotations

from urllib.parse import urlsplit

from config import CORS_ORIGINS

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
