from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.routes.deps import get_memory_store, request_client
from lib.memory_store import MemoryStore, ProposedMemory as MemoryProposedMemory

router = APIRouter(prefix="/api/memory", tags=["memory"])

MemoryStoreDep = Annotated[MemoryStore, Depends(get_memory_store)]


class MemoryExtractRequest(BaseModel):
    messages: list[dict[str, str]]
    project_slug: str | None = None
    chat_id: str | None = None
    scope: str | None = None  # "global" | "project" | None (both)


class MemoryReviewRequest(BaseModel):
    review_id: str
    project_slug: str | None = None


class ProposedMemoryModel(BaseModel):
    id: str
    memory: str
    scope: str
    category: str | None = None


class MemoryCommitRequest(BaseModel):
    scope: str  # "global" | "project"
    accepted_memories: list[ProposedMemoryModel]
    project_slug: str | None = None
    review_id: str | None = None
    rejected_memories: list[ProposedMemoryModel] | None = None
    revised_memories: list[dict] | None = None


class MemoryPreviewRequest(BaseModel):
    scope: str
    accepted_memories: list[ProposedMemoryModel]
    project_slug: str | None = None


@router.post("/extract")
async def extract_memories(
    req: MemoryExtractRequest,
    store: MemoryStoreDep,
) -> dict:
    try:
        with request_client() as client:
            result = await store.extract(
                client, req.messages, project_slug=req.project_slug, chat_id=req.chat_id, scope=req.scope
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {
        "review_id": result.review_id,
        "global": [{"id": m.id, "memory": m.memory, "scope": m.scope, "category": m.category} for m in result.global_memories],
        "project": [{"id": m.id, "memory": m.memory, "scope": m.scope, "category": m.category} for m in result.project_memories]
        if result.project_memories
        else None,
    }


@router.post("/commit")
async def commit_memory_docs(
    req: MemoryCommitRequest,
    store: MemoryStoreDep,
) -> dict:
    try:
        with request_client() as client:
            accepted = [MemoryProposedMemory(m.id, m.memory, m.scope, m.category or "note") for m in req.accepted_memories]
            await store.commit_async(
                llm=client,
                scope=req.scope,
                accepted_memories=accepted,
                project_slug=req.project_slug,
                review_id=req.review_id,
                rejected_memories=[m.model_dump() for m in req.rejected_memories] if req.rejected_memories else None,
                revised_memories=req.revised_memories,
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"status": "committed"}


@router.post("/preview")
async def preview_memory_docs(
    req: MemoryPreviewRequest,
    store: MemoryStoreDep,
) -> dict:
    try:
        with request_client() as client:
            accepted = [MemoryProposedMemory(m.id, m.memory, m.scope, m.category or "note") for m in req.accepted_memories]
            result = await store.preview(
                llm=client,
                scope=req.scope,
                accepted_memories=accepted,
                project_slug=req.project_slug,
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return result


@router.post("/cancel")
async def cancel_memories(
    req: MemoryReviewRequest,
    store: MemoryStoreDep,
) -> dict:
    try:
        await store.cancel_async(review_id=req.review_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"status": "cancelled"}


@router.get("/memories")
def list_memories(
    store: MemoryStoreDep,
    scope: str = Query(...),
    project_slug: str | None = Query(None, alias="project_slug"),
) -> dict:
    try:
        memories = store.list_memories(scope=scope, project_slug=project_slug)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"memories": memories}


@router.get("/categories")
def get_categories(
    store: MemoryStoreDep,
    scope: str = Query(...),
    project_slug: str | None = Query(None, alias="project_slug"),
) -> dict:
    try:
        categories = store.get_categories(scope=scope, project_slug=project_slug)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"categories": categories}


class CategorySetRequest(BaseModel):
    scope: str
    categories: list[dict]
    project_slug: str | None = None


@router.put("/categories")
def set_categories(req: CategorySetRequest, store: MemoryStoreDep) -> dict:
    try:
        store.set_categories(scope=req.scope, categories=req.categories, project_slug=req.project_slug)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"status": "saved"}


class RemapRequest(BaseModel):
    scope: str
    orphaned_memories: list[dict]
    remaining_categories: list[dict]
    project_slug: str | None = None


@router.post("/remap-category")
async def remap_category(req: RemapRequest, store: MemoryStoreDep) -> dict:
    try:
        with request_client() as client:
            remapped = await store.remap_orphaned(
                llm=client,
                orphaned_memories=req.orphaned_memories,
                remaining_categories=req.remaining_categories,
                scope=req.scope,
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"memories": remapped}


class MemoryMoveRequest(BaseModel):
    memory_id: str
    from_scope: str
    to_scope: str
    from_project_slug: str | None = None
    to_project_slug: str | None = None


@router.post("/move")
def move_memory(req: MemoryMoveRequest, store: MemoryStoreDep) -> dict:
    try:
        store.move_memory(
            memory_id=req.memory_id,
            from_scope=req.from_scope,
            to_scope=req.to_scope,
            from_project_slug=req.from_project_slug,
            to_project_slug=req.to_project_slug,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"status": "moved"}
