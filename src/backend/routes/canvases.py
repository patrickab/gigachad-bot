"""HTTP and SSE adapters for transactional collaborative canvas mutations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
import json
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.http_plumbing import http_errors, live_stream
from backend.identity import RequestIdentity, get_request_identity
from backend.routes.deps import get_canvas_store
from backend.sync import get_change_broker
from lib.canvas_store import (
    CanvasMutation,
    CanvasMutationStore,
    CanvasNotFound,
    Entity,
    InvalidCanvasMutation,
    InvalidCanvasSnapshot,
    MutationKind,
    normalize_canvas_key,
)

router = APIRouter(prefix="/api/canvases", tags=["canvases"])


class CanvasMutationRequest(BaseModel):
    mutationId: UUID
    kind: MutationKind
    entity: Entity
    id: str = Field(min_length=1)
    value: dict[str, Any] | None = None


class CanvasMutationBatchRequest(BaseModel):
    mutations: list[CanvasMutationRequest]


async def _call[T](function: Callable[..., T], *args: Any) -> T:
    with http_errors((CanvasNotFound, 404, "Canvas not found"), ((InvalidCanvasMutation, InvalidCanvasSnapshot), 422, None)):
        return await asyncio.to_thread(function, *args)


@router.post("/{canvas_key:path}/mutations")
async def submit_mutations(
    canvas_key: str,
    request: CanvasMutationBatchRequest,
    store: CanvasMutationStore = Depends(get_canvas_store),
) -> dict[str, int]:
    """Atomically apply unseen ID-keyed mutations; acknowledge the revision holding them."""
    mutations = [CanvasMutation(m.mutationId, m.kind, m.entity, m.id, m.value) for m in request.mutations]
    return {"revision": await _call(store.apply, canvas_key, mutations)}


@router.get("/{canvas_key:path}/stream")
async def stream_canvas(
    canvas_key: str,
    sinceRevision: int = Query(default=0, ge=0),
    identity: RequestIdentity = Depends(get_request_identity),
    store: CanvasMutationStore = Depends(get_canvas_store),
) -> EventSourceResponse:
    """Replay durable batches after ``sinceRevision``, then again on every change to this canvas."""
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
                # Block until this canvas changes. The loop ends only when the broker closes the subscription, which ends the stream.
                async for change in live:
                    if change.get("resource_kind") == "document" and change.get("resource_key") == key:
                        break
                else:
                    return

    return live_stream(events())


@router.get("/{canvas_key:path}")
async def read_canvas_snapshot(
    canvas_key: str,
    store: CanvasMutationStore = Depends(get_canvas_store),
) -> dict[str, Any]:
    """Return the canonical materialized canvas with the revision that produced it."""
    document, revision = await _call(store.snapshot, canvas_key)
    return {"revision": revision, "document": document}
