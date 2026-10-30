"""HTTP plumbing shared by the routes: exception-to-status mapping, SSE responses, inline image decoding and response transfer."""

import asyncio
import base64
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
import gzip
import hashlib
import json
import re
from typing import Any

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from llm_baseclient.client import LLMClient
from sse_starlette.sse import EventSourceResponse
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from lib.async_utils import aiter_in_thread
from lib.llm_resilience import api_query_resilient

# `(exception or exceptions, status, detail)`, a `None` detail uses `str(exc)`.
Rule = tuple[type[Exception] | tuple[type[Exception], ...], int, str | None]

_DATA_URI_RE = re.compile(r"data:image/\w+;base64,(.+)")


@contextmanager
def http_errors(*rules: Rule) -> Iterator[None]:
    """Re-raise as `HTTPException` per the first matching `(exception, status, detail)` rule, a `None` detail uses `str(exc)`."""
    try:
        yield
    except Exception as exc:
        for kinds, status, detail in rules:
            if isinstance(exc, kinds):
                raise HTTPException(status_code=status, detail=str(exc) if detail is None else detail) from exc
        raise


def decode_image(base64_data: str | None) -> bytes | None:
    """Decode a raw or `data:image/...;base64,` string to bytes."""
    if not base64_data:
        return None
    match = _DATA_URI_RE.match(base64_data)
    if match:
        return base64.b64decode(match.group(1))
    return base64.b64decode(base64_data)


_HEARTBEAT_SECONDS = 15
_SSE_HEADERS = {
    # Some reverse proxies otherwise buffer a small SSE response until the model
    # finishes. Vision providers commonly have larger first-token gaps, exposing
    # that buffering as an apparently non-streaming reply.
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
}


def _sse(events: AsyncIterator[tuple[str, str]]) -> EventSourceResponse:
    """Forward `(event, data)` pairs, then `done`, or an `error` event when the stream fails."""

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        try:
            async for event, data in events:
                yield {"event": event, "data": data}
            yield {"event": "done", "data": ""}
        except Exception as e:  # noqa: BLE001 - a failed turn must reach the UI as an error event
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_stream(), headers=_SSE_HEADERS)


async def stream_completion(client: LLMClient, **query: Any) -> EventSourceResponse:
    """Stream a completion as SSE: string chunks become `token` events, dict chunks `usage` events."""
    chunks = await run_in_threadpool(api_query_resilient, client, stream=True, **query)

    async def events() -> AsyncIterator[tuple[str, str]]:
        async for chunk in aiter_in_thread(chunks):
            yield ("usage", json.dumps(chunk)) if isinstance(chunk, dict) else ("token", chunk)

    return _sse(events())


def sse_tool_event_stream(events: AsyncIterator[tuple[str, Any]]) -> EventSourceResponse:
    """Forward a tool-loop event stream. `token` data stays raw text, everything else is JSON."""

    async def tagged() -> AsyncIterator[tuple[str, str]]:
        async for name, data in events:
            yield name, data if isinstance(data, str) else json.dumps(data)

    return _sse(tagged())


def live_stream(events: AsyncIterator[dict[str, str]]) -> EventSourceResponse:
    """Serve long-lived SSE events, pinging idle connections so proxies keep them open."""
    return EventSourceResponse(events, ping=_HEARTBEAT_SECONDS)


# Cacheable per browser, but revalidated on every use, so a change shows on the next request.
REVALIDATE = "private, no-cache"
_TRANSFER_TYPES = ("application/json", "text/")
_GZIP_MIN_BYTES = 1024
# Level 1: the 3.5 MB ink canvas compresses to 1.38 MB in 48 ms, level 6 only reaches 1.20 MB in 173 ms.
_GZIP_LEVEL = 1
_OFF_LOOP_BYTES = 256 * 1024


def etag_matches(if_none_match: str | None, etag: str) -> bool:
    """Whether an ``If-None-Match`` header lists *etag* (weak comparison)."""
    if if_none_match is None:
        return False
    tags = {tag.strip().removeprefix("W/") for tag in if_none_match.split(",")}
    return "*" in tags or etag.removeprefix("W/") in tags


def _is_transfer_optimizable(status: int, headers: Headers) -> bool:
    """Whole 200 JSON/text bodies the route left without its own validator or encoding."""
    content_type = headers.get("content-type", "")
    return (
        status == 200
        and content_type.startswith(_TRANSFER_TYPES)
        and not content_type.startswith("text/event-stream")
        and "etag" not in headers
        and "content-encoding" not in headers
    )


async def _off_loop_if_large[T](function: Callable[[bytes], T], body: bytes) -> T:
    """Hash or compress small bodies inline, large ones in a thread so the event loop keeps serving streams."""
    if len(body) < _OFF_LOOP_BYTES:
        return function(body)
    return await asyncio.to_thread(function, body)


def _body_etag(body: bytes) -> str:
    return f'W/"{hashlib.blake2b(body, digest_size=16).hexdigest()}"'


def _gzip(body: bytes) -> bytes:
    return gzip.compress(body, _GZIP_LEVEL, mtime=0)


class TransferMiddleware:
    """Validate and gzip whole JSON/text responses.

    - GET: weak ETag of the body plus ``private, no-cache``, a matching ``If-None-Match`` gets 304 without the body
    - Gzip when the client accepts it and the body has at least ``_GZIP_MIN_BYTES``
    - Streamed bodies (SSE, proxies, files), ranges and responses with their own ETag or encoding pass through
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_headers = Headers(scope=scope)
        deferred_start: Message | None = None

        async def send_optimized(message: Message) -> None:
            nonlocal deferred_start
            if message["type"] == "http.response.start" and _is_transfer_optimizable(message["status"], Headers(raw=message["headers"])):
                deferred_start = message
                return
            if deferred_start is None or message["type"] != "http.response.body":
                await send(message)
                return
            start, deferred_start = deferred_start, None
            if message.get("more_body", False):
                await send(start)
                await send(message)
                return
            await self._send_whole(scope["method"], request_headers, start, message.get("body", b""), send)

        await self.app(scope, receive, send_optimized)

    @staticmethod
    async def _send_whole(method: str, request_headers: Headers, start: Message, body: bytes, send: Send) -> None:
        headers = MutableHeaders(raw=start["headers"])
        headers.add_vary_header("Accept-Encoding")
        if method == "GET":
            etag = await _off_loop_if_large(_body_etag, body)
            headers["ETag"] = etag
            headers.setdefault("Cache-Control", REVALIDATE)
            if etag_matches(request_headers.get("if-none-match"), etag):
                del headers["Content-Length"]
                del headers["Content-Type"]
                await send({**start, "status": 304})
                await send({"type": "http.response.body", "body": b""})
                return
        if len(body) >= _GZIP_MIN_BYTES and "gzip" in request_headers.get("accept-encoding", ""):
            body = await _off_loop_if_large(_gzip, body)
            headers["Content-Encoding"] = "gzip"
            headers["Content-Length"] = str(len(body))
        await send(start)
        await send({"type": "http.response.body", "body": body})
