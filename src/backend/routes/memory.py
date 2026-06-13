from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.routes.deps import get_memory_store, request_client
from lib.memory_store import MemoryStore

router = APIRouter(prefix="/api/memory", tags=["memory"])

MemoryStoreDep = Annotated[MemoryStore, Depends(get_memory_store)]


class MemoryExtractRequest(BaseModel):
    messages: list[dict[str, str]]
    project_slug: str | None = None


class MemoryReviewRequest(BaseModel):
    review_id: str
    project_slug: str | None = None


class ManualMemory(BaseModel):
    id: str
    memory: str
    scope: str
    kind: str | None = None
    reason: str | None = None
    categories: list[str] | None = None


class MemoryComposeRequest(BaseModel):
    review_id: str
    accepted_ids: list[str]
    manual_memories: list[ManualMemory] = Field(default_factory=list)
    project_slug: str | None = None


class MemoryCommitRequest(BaseModel):
    review_id: str
    global_document: str | None = None
    project_document: str | None = None
    project_slug: str | None = None


@router.post("/extract")
async def extract_memories(
    req: MemoryExtractRequest,
    store: MemoryStoreDep,
) -> dict:
    try:
        with request_client() as client:
            result = await store.extract(client, req.messages, project_slug=req.project_slug)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {
        "review_id": result.review_id,
        "global": [
            {"id": m.id, "memory": m.memory, "scope": m.scope, "kind": m.kind, "reason": m.reason, "categories": m.categories}
            for m in result.global_memories
        ],
        "project": [
            {"id": m.id, "memory": m.memory, "scope": m.scope, "kind": m.kind, "reason": m.reason, "categories": m.categories}
            for m in result.project_memories
        ] if result.project_memories else None,
    }


@router.post("/compose")
async def compose_memory_docs(
    req: MemoryComposeRequest,
    store: MemoryStoreDep,
) -> dict:
    try:
        with request_client() as client:
            result = await store.compose(
                client,
                review_id=req.review_id,
                accepted_ids=req.accepted_ids,
                project_slug=req.project_slug,
                manual_memories=[m.model_dump() for m in req.manual_memories],
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {
        "review_id": result.review_id,
        "global_document": result.global_document,
        "project_document": result.project_document,
        "global_diff": result.global_diff,
        "project_diff": result.project_diff,
    }


@router.post("/commit")
async def commit_memory_docs(
    req: MemoryCommitRequest,
    store: MemoryStoreDep,
) -> dict:
    try:
        await store.commit_async(
            review_id=req.review_id,
            global_document=req.global_document,
            project_document=req.project_document,
            project_slug=req.project_slug,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"status": "committed"}


@router.post("/cancel")
async def cancel_memories(
    req: MemoryReviewRequest,
    store: MemoryStoreDep,
) -> dict:
    try:
        await store.cancel_async(review_id=req.review_id, project_slug=req.project_slug)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"status": "cancelled"}
