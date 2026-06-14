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


class MemoryReviewRequest(BaseModel):
    review_id: str
    project_slug: str | None = None


class ProposedMemoryModel(BaseModel):
    id: str
    memory: str
    scope: str
    kind: str | None = None


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
            result = await store.extract(client, req.messages, project_slug=req.project_slug)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {
        "review_id": result.review_id,
        "global": [
            {"id": m.id, "memory": m.memory, "scope": m.scope, "kind": m.kind}
            for m in result.global_memories
        ],
        "project": [
            {"id": m.id, "memory": m.memory, "scope": m.scope, "kind": m.kind}
            for m in result.project_memories
        ] if result.project_memories else None,
    }


@router.post("/commit")
async def commit_memory_docs(
    req: MemoryCommitRequest,
    store: MemoryStoreDep,
) -> dict:
    try:
        with request_client() as client:
            accepted = [
                MemoryProposedMemory(m.id, m.memory, m.scope, m.kind or "note")
                for m in req.accepted_memories
            ]
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
            accepted = [
                MemoryProposedMemory(m.id, m.memory, m.scope, m.kind or "note")
                for m in req.accepted_memories
            ]
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
        await store.cancel_async(review_id=req.review_id, project_slug=req.project_slug)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"status": "cancelled"}


@router.get("/profiles")
def list_memory_profiles(
    store: MemoryStoreDep,
    project_slug: str | None = Query(None, alias="project_slug"),
) -> dict:
    try:
        profiles = store.list_profiles(project_slug=project_slug)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"profiles": profiles}


@router.get("/content")
def get_memory_profile_content(
    store: MemoryStoreDep,
    filepath: str = Query(..., alias="filepath"),
) -> dict:
    try:
        content = store.get_profile_content(filepath=filepath)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"content": content, "filepath": filepath}
