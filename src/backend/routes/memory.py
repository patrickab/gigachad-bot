from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.routes.deps import get_memory_store
from lib.memory_store import MemoryStore

router = APIRouter(prefix="/api/memory", tags=["memory"])

MemoryStoreDep = Annotated[MemoryStore, Depends(get_memory_store)]


class MemoryExtractRequest(BaseModel):
    messages: list[dict[str, str]]
    project_slug: str | None = None


class MemoryReviewRequest(BaseModel):
    review_id: str
    project_slug: str | None = None


class MemoryItemReviewRequest(BaseModel):
    review_id: str
    memory_id: str


@router.post("/extract")
async def extract_memories(
    req: MemoryExtractRequest,
    store: MemoryStoreDep,
) -> dict:
    result = await store.extract(req.messages, project_slug=req.project_slug)
    return {
        "review_id": result.review_id,
        "global": [
            {"id": m.id, "memory": m.memory, "categories": m.categories}
            for m in result.global_memories
        ],
        "project": [
            {"id": m.id, "memory": m.memory, "categories": m.categories}
            for m in result.project_memories
        ] if result.project_memories else None,
    }


@router.post("/accept")
async def accept_memories(
    req: MemoryReviewRequest,
    store: MemoryStoreDep,
) -> dict:
    await store.accept(review_id=req.review_id, project_slug=req.project_slug)
    return {"status": "accepted"}


@router.post("/accept-one")
async def accept_one_memory(
    req: MemoryItemReviewRequest,
    store: MemoryStoreDep,
) -> dict:
    await store.accept_one(review_id=req.review_id, memory_id=req.memory_id)
    return {"status": "accepted"}


@router.post("/cancel")
async def cancel_memories(
    req: MemoryReviewRequest,
    store: MemoryStoreDep,
) -> dict:
    await store.cancel(review_id=req.review_id, project_slug=req.project_slug)
    return {"status": "cancelled"}


@router.post("/cancel-one")
async def cancel_one_memory(
    req: MemoryItemReviewRequest,
    store: MemoryStoreDep,
) -> dict:
    await store.cancel_one(review_id=req.review_id, memory_id=req.memory_id)
    return {"status": "cancelled"}
