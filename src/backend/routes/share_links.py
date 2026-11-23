"""Share links over HTTP (`lib.share_links`).

- `/api/shares`: the owner creates, inspects and revokes the link of one of their resources
- `/api/shared/{secret}`: whoever holds a secret reaches its one resource as its owner, without a login of their own.
  An unknown, rotated or revoked secret gets the same 404 everywhere
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.http_plumbing import http_errors
from backend.routes.canvases import CanvasMutationBatchRequest, apply_mutations, canvas_snapshot, follow_canvas, stream_batches
from backend.routes.deps import get_share_links
from lib.canvas_store import CanvasMutationStore
from lib.changes import UserScope
from lib.data_store import InvalidStorageKey, StorageNotFoundError, validate_key
from lib.postgres_pool import get_postgres_pool
from lib.share_links import Access, Kind, ShareGrant, ShareLinks, ShareTarget, ShareTargetNotFound, find_grant

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from lib.canvas_store import CanvasBatch

owner_router = APIRouter(prefix="/api/shares", tags=["shares"])
guest_router = APIRouter(prefix="/api/shared", tags=["shares"])

_NOT_FOUND = "Share link not found"
# Not stored, so revoking a link takes effect at once. Never sniffed, so bytes keep their declared type.
_CONTENT_HEADERS = {"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"}


def _content_policy(mime: str) -> dict[str, str]:
    """Owner bytes render on the API origin, so they are sandboxed: shared HTML or SVG cannot run script there.

    PDFs stay unsandboxed: browsers refuse to show them in a sandboxed document, and their viewer runs apart from the page.
    """
    return {} if mime == "application/pdf" else {"Content-Security-Policy": "sandbox"}


class ShareTargetRequest(BaseModel):
    kind: Kind
    key: str = Field(min_length=1)


def _parse_target(kind: Kind, key: str) -> ShareTarget:
    with http_errors((InvalidStorageKey, 404, "Nothing to share")):
        return ShareTarget(kind, validate_key(key))


@owner_router.get("")
def read_share(kind: Kind, key: str, links: ShareLinks = Depends(get_share_links)) -> dict[str, Access | None]:
    target = _parse_target(kind, key)
    return {"access": target.access if links.is_shared(target) else None}


@owner_router.post("")
def create_share(request: ShareTargetRequest, links: ShareLinks = Depends(get_share_links)) -> dict[str, str]:
    """Issue the resource's link secret, replacing any previous one. The secret is never shown again."""
    target = _parse_target(request.kind, request.key)
    with http_errors((ShareTargetNotFound, 404, None)):
        return {"secret": links.create(target), "access": target.access}


@owner_router.delete("")
def revoke_share(kind: Kind, key: str, links: ShareLinks = Depends(get_share_links)) -> None:
    links.revoke(_parse_target(kind, key))


def get_grant(secret: str) -> ShareGrant:
    grant = find_grant(get_postgres_pool(), secret)
    if grant is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return grant


def get_canvas_grant(grant: ShareGrant = Depends(get_grant)) -> ShareGrant:
    if grant.target.view != "canvas":
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return grant


def _owner_scope(grant: ShareGrant) -> UserScope:
    # No device: a guest's writes reach every device of the owner, including the one that shared.
    return UserScope(get_postgres_pool(), grant.owner)


@guest_router.get("/{secret}")
def describe_shared(grant: ShareGrant = Depends(get_grant)) -> dict[str, Any]:
    """What the link opens and how. The owner's full key stays private, only its name is shown."""
    target = grant.target
    return {"name": PurePosixPath(target.key).name, "view": target.view}


@guest_router.get("/{secret}/content")
def read_shared_content(grant: ShareGrant = Depends(get_grant)) -> Response:
    target = grant.target
    with http_errors(((InvalidStorageKey, StorageNotFoundError), 404, _NOT_FOUND)):
        content, mime = target.read_content(_owner_scope(grant))
    disposition = f"inline; filename*=UTF-8''{quote(PurePosixPath(target.key).name)}"
    headers = {**_CONTENT_HEADERS, **_content_policy(mime), "Content-Disposition": disposition}
    return Response(content=content, media_type=mime, headers=headers)


@guest_router.get("/{secret}/canvas")
async def read_shared_canvas(grant: ShareGrant = Depends(get_canvas_grant)) -> dict[str, Any]:
    return await canvas_snapshot(CanvasMutationStore(_owner_scope(grant)), grant.target.key)


@guest_router.post("/{secret}/canvas/mutations")
async def submit_shared_mutations(
    request: CanvasMutationBatchRequest, grant: ShareGrant = Depends(get_canvas_grant),
) -> dict[str, int]:
    return await apply_mutations(CanvasMutationStore(_owner_scope(grant)), grant.target.key, request)


@guest_router.get("/{secret}/canvas/stream")
async def stream_shared_canvas(
    secret: str, sinceRevision: int = Query(default=0, ge=0), grant: ShareGrant = Depends(get_canvas_grant),
) -> EventSourceResponse:
    updates = await follow_canvas(_owner_scope(grant), grant.target.key, sinceRevision)
    return stream_batches(_until_revoked(updates, secret, grant))


async def _until_revoked(
    updates: AsyncGenerator[list[CanvasBatch]], secret: str, grant: ShareGrant,
) -> AsyncGenerator[list[CanvasBatch]]:
    """Pass updates on while the secret still grants this canvas: a revoked or rotated link receives nothing more.

    Revoking publishes no change, so an idle stream ends at the canvas's next change, before delivering it.
    """
    pool = get_postgres_pool()
    async with aclosing(updates):
        async for batches in updates:
            if await asyncio.to_thread(find_grant, pool, secret) != grant:
                return
            yield batches
