"""Server-sent delivery of the Postgres change log.

`GET /api/sync/stream?since=<seq>` emits one `change` event per resource the requesting user
changed, first replaying each resource's latest change with `seq > since`, then
streaming live rows as they are committed. Event data is JSON::

    event: change
    data: {"seq": 41, "resource_kind": "document", "resource_key": "notes/a.md",
           "version": 3, "device_id": "0f1e...-uuid" | null}

`seq` is monotonic, so a client persists the highest value it has applied and resumes
with `?since=<seq>` after a disconnect. Events carry metadata only: the content is
fetched through the resource's own API, and `device_id` lets a client ignore echoes of
its own writes. Heartbeats are SSE comments every 15 seconds.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Query
from sse_starlette.sse import EventSourceResponse

from backend.http_plumbing import live_stream
from backend.identity import RequestIdentity, get_request_identity
from backend.sync import get_change_broker
from lib.changes import EVENT_KEYS, latest_changes
from lib.postgres_pool import get_postgres_pool

router = APIRouter(prefix="/api/sync", tags=["sync"])


def _event(payload: dict) -> dict[str, str]:
    return {"event": "change", "data": json.dumps({key: payload.get(key) for key in EVENT_KEYS}, default=str)}


@router.get("/stream")
async def stream_changes(
    since: int = Query(default=0, ge=0),
    identity: RequestIdentity = Depends(get_request_identity),
) -> EventSourceResponse:
    """Replay missed changes, then stream live ones, for the requesting user only."""
    user_id = identity.user_id
    broker = get_change_broker()

    async def events() -> AsyncIterator[dict[str, str]]:
        # Subscribe before replaying so a change committed mid-replay is not lost.
        # The seq watermark then suppresses the overlap.
        async with broker.subscribe(user_id) as live:
            delivered = since
            for row in await asyncio.to_thread(latest_changes, get_postgres_pool(), user_id, since):
                delivered = row["seq"]
                yield _event(row)
            async for change in live:
                if change["seq"] <= delivered:
                    continue
                delivered = change["seq"]
                yield _event(change)

    return live_stream(events())
