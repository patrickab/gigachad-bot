"""HTTP plumbing shared by the routes: exception-to-status mapping, SSE responses and inline image decoding."""

import base64
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
import json
import re
from typing import Any

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from llm_baseclient.client import LLMClient
from sse_starlette.sse import EventSourceResponse

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
