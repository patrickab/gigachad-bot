"""HTTP and SSE adapters for transactional collaborative canvas mutations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
import json
from typing import Any, Literal, TypeVar
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.canvas import CanvasMutation, CanvasMutationStore, CanvasNotFound, InvalidCanvasMutation, InvalidCanvasSnapshot, normalize_canvas_key
from backend.identity import RequestIdentity, get_request_identity
from backend.sync import get_change_broker
from config import get_postgres_pool

router = APIRouter(prefix="/api/canvases", tags=["canvases"])
HEARTBEAT_SECONDS = 15
T = TypeVar("T")


class CanvasMutationRequest(BaseModel):
    mutationId: UUID
    kind: Literal["upsert", "delete"]
    entity: Literal["stroke", "frame", "attachment", "text"]
    id: str = Field(min_length=1)
    value: dict[str, Any] | None = None


class CanvasMutationBatchRequest(BaseModel):
    mutations: list[CanvasMutationRequest]


def _store(identity: RequestIdentity) -> CanvasMutationStore:
    return CanvasMutationStore(get_postgres_pool(), identity.user_id, identity.device_id)


async def _call(function: Callable[..., T], *args: Any) -> T:
    try:
        return await asyncio.to_thread(function, *args)
    except CanvasNotFound as exc:
        raise HTTPException(status_code=404, detail="Canvas not found") from exc
    except (InvalidCanvasMutation, InvalidCanvasSnapshot) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{canvas_key:path}/mutations")
async def submit_mutations(
    canvas_key: str,
    request: CanvasMutationBatchRequest,
    identity: RequestIdentity = Depends(get_request_identity),
) -> dict[str, int]:
    """Atomically apply unseen ID-keyed mutations; acknowledge the revision holding them."""
    mutations = [CanvasMutation(m.mutationId, m.kind, m.entity, m.id, m.value) for m in request.mutations]
    return {"revision": await _call(_store(identity).apply, canvas_key, mutations)}


@router.get("/{canvas_key:path}/stream")
async def stream_canvas(
    canvas_key: str,
    sinceRevision: int = Query(default=0, ge=0),
    identity: RequestIdentity = Depends(get_request_identity),
) -> EventSourceResponse:
    """Replay durable batches after ``sinceRevision``, then again on every change to this canvas."""
    store = _store(identity)
    first = await _call(store.replay, canvas_key, sinceRevision)
    key = normalize_canvas_key(canvas_key)

    async def events() -> AsyncIterator[dict[str, str]]:
        async with get_change_broker().subscribe(identity.user_id) as live:
            delivered, batches = sinceRevision, first
            while True:
                for batch in batches:
                    delivered = batch.revision
                    yield {"event": "batch", "data": json.dumps(batch.event_data())}
                # Replaying after subscribing closes the gap to the previous replay.
                batches = await asyncio.to_thread(store.replay, key, delivered)
                if batches:
                    continue
                async for change in live:
                    if change.get("resource_kind") == "document" and change.get("resource_key") == key:
                        break
                else:
                    return

    return EventSourceResponse(events(), ping=HEARTBEAT_SECONDS)


@router.get("/{canvas_key:path}")
async def read_canvas_snapshot(
    canvas_key: str,
    identity: RequestIdentity = Depends(get_request_identity),
) -> dict[str, Any]:
    """Return the canonical materialized canvas with the revision that produced it."""
    document, revision = await _call(_store(identity).snapshot, canvas_key)
    return {"revision": revision, "document": document}
