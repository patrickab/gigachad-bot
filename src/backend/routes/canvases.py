"""HTTP and SSE adapters for transactional collaborative canvas mutations."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
import json
from typing import TYPE_CHECKING, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.http_plumbing import http_errors, live_stream
from backend.routes.deps import get_canvas_store, get_user_scope
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
from lib.changes import UserScope

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable

    from lib.canvas_store import CanvasBatch

router = APIRouter(prefix="/api/canvases", tags=["canvases"])


class CanvasMutationRequest(BaseModel):
    mutationId: UUID
    kind: MutationKind
    entity: Entity
    id: str = Field(min_length=1)
    value: dict[str, Any] | None = None


class CanvasMutationBatchRequest(BaseModel):
    mutations: list[CanvasMutationRequest]


async def _call[T](function: Callable[..., T], *args: object) -> T:
    with http_errors((CanvasNotFound, 404, "Canvas not found"), ((InvalidCanvasMutation, InvalidCanvasSnapshot), 422, None)):
        return await asyncio.to_thread(function, *args)


async def apply_mutations(store: CanvasMutationStore, canvas_key: str, request: CanvasMutationBatchRequest) -> dict[str, int]:
    """Atomically apply unseen ID-keyed mutations; acknowledge the revision holding them."""
    mutations = [CanvasMutation(m.mutationId, m.kind, m.entity, m.id, m.value) for m in request.mutations]
    return {"revision": await _call(store.apply, canvas_key, mutations)}


async def canvas_snapshot(store: CanvasMutationStore, canvas_key: str) -> dict[str, Any]:
    """The canonical materialized canvas with the revision that produced it."""
    document, revision = await _call(store.snapshot, canvas_key)
    return {"revision": revision, "document": document}


async def follow_canvas(scope: UserScope, canvas_key: str, since_revision: int) -> AsyncGenerator[list[CanvasBatch]]:
    """Lists of new batches: those after ``since_revision``, then the new ones on every change to this canvas.

    The first replay runs before this returns, so a missing canvas fails the request instead of the stream.
    """
    store = CanvasMutationStore(scope)
    first = await _call(store.replay, canvas_key, since_revision)
    key = normalize_canvas_key(canvas_key)

    async def updates() -> AsyncGenerator[list[CanvasBatch]]:
        async with get_change_broker().subscribe(scope.user_id) as live:
            delivered, batches = since_revision, first
            while True:
                if batches:
                    yield batches
                    delivered = batches[-1].revision
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

    return updates()


def stream_batches(updates: AsyncGenerator[list[CanvasBatch]]) -> EventSourceResponse:
    async def events() -> AsyncIterator[dict[str, str]]:
        async with aclosing(updates):
            async for batches in updates:
                for batch in batches:
                    yield {"event": "batch", "data": json.dumps(batch.event_data())}

    return live_stream(events())


@router.post("/{canvas_key:path}/mutations")
async def submit_mutations(
    canvas_key: str,
    request: CanvasMutationBatchRequest,
    store: CanvasMutationStore = Depends(get_canvas_store),
) -> dict[str, int]:
    return await apply_mutations(store, canvas_key, request)


@router.get("/{canvas_key:path}/stream")
async def stream_canvas(
    canvas_key: str,
    sinceRevision: int = Query(default=0, ge=0),
    scope: UserScope = Depends(get_user_scope),
) -> EventSourceResponse:
    return stream_batches(await follow_canvas(scope, canvas_key, sinceRevision))


@router.get("/{canvas_key:path}")
async def read_canvas_snapshot(
    canvas_key: str,
    store: CanvasMutationStore = Depends(get_canvas_store),
) -> dict[str, Any]:
    return await canvas_snapshot(store, canvas_key)
