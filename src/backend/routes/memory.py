from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.http_plumbing import http_errors
from backend.routes.deps import get_memory_store, get_model_defaults, request_client
from lib.memory_extract import MemoryExtractor, ScopeProposal
from lib.memory_store import MemoryOperation, MemoryScope, MemoryStore, ScopeKind

router = APIRouter(prefix="/api/memory", tags=["memory"])

MemoryStoreDep = Annotated[MemoryStore, Depends(get_memory_store)]
ModelDefaultsDep = Annotated[dict[str, str], Depends(get_model_defaults)]
# The memory store reports every failure as a plain exception, the UI shows it as an unavailable backend.
_STORE_FAILURE = (Exception, 502, None)


class MemoryExtractRequest(BaseModel):
    messages: list[dict[str, str]]
    project_slug: str | None = None
    chat_id: str | None = None


class MemoryCancelRequest(BaseModel):
    review_id: str


class OperationModel(BaseModel):
    id: str
    op: Literal["add", "update", "delete"]
    topic: str
    text: str = ""
    target_id: str | None = None
    index: int | None = None


class TopicModel(BaseModel):
    name: str
    summary: str


class ScopedRequest(BaseModel):
    scope: ScopeKind
    project_slug: str | None = None

    def memory_scope(self) -> MemoryScope:
        return MemoryScope.of(self.scope, self.project_slug)


class MemoryCommitRequest(ScopedRequest):
    operations: list[OperationModel]
    topics: list[TopicModel] = []
    review_id: str | None = None


class TopicSetRequest(ScopedRequest):
    topics: list[TopicModel]


class MemoriesSetRequest(ScopedRequest):
    memories: list[dict]


def _proposal_json(proposal: ScopeProposal | None) -> dict | None:
    if not proposal:
        return None
    return {"operations": [asdict(o) for o in proposal.operations], "topics": [asdict(t) for t in proposal.topics]}


@router.post("/extract")
async def extract_memories(req: MemoryExtractRequest, store: MemoryStoreDep, defaults: ModelDefaultsDep) -> dict:
    """Propose memory operations from a conversation for user review, a scope is None when not extracted."""
    with http_errors(_STORE_FAILURE), request_client() as client:
        extractor = MemoryExtractor(store, client, defaults["memory_model"])
        result = await extractor.extract(req.messages, project_slug=req.project_slug, chat_id=req.chat_id)
    return {
        "review_id": result.review_id,
        "global": _proposal_json(result.global_proposal),
        "project": _proposal_json(result.project_proposal),
    }


@router.post("/commit")
def commit_memories(req: MemoryCommitRequest, store: MemoryStoreDep) -> dict:
    """Apply accepted operations and topics, then advance the reviewed chat's watermark."""
    operations = [MemoryOperation(o.id, o.op, req.scope, o.topic, o.text, o.target_id, index=o.index) for o in req.operations]
    with http_errors(_STORE_FAILURE):
        store.commit(req.memory_scope(), operations, [t.model_dump() for t in req.topics], req.review_id)
    return {"status": "committed"}


@router.post("/cancel")
def cancel_memories(req: MemoryCancelRequest, store: MemoryStoreDep) -> dict:
    """Discard a pending review without advancing its watermark."""
    store.cancel(req.review_id)
    return {"status": "cancelled"}


@router.get("/memories")
def list_memories(store: MemoryStoreDep, scope: ScopeKind, project_slug: str | None = None) -> dict:
    with http_errors(_STORE_FAILURE):
        return {"memories": store.list_memories(MemoryScope.of(scope, project_slug))}


@router.put("/memories")
def set_memories(req: MemoriesSetRequest, store: MemoryStoreDep) -> dict:
    """Replace a scope's stored memories, used by the memory window's direct edits."""
    with http_errors(_STORE_FAILURE):
        store.write_memories(req.memory_scope(), req.memories)
    return {"status": "saved"}


@router.get("/topics")
def get_topics(store: MemoryStoreDep, scope: ScopeKind, project_slug: str | None = None) -> dict:
    with http_errors(_STORE_FAILURE):
        return {"topics": store.get_topics(MemoryScope.of(scope, project_slug))}


@router.put("/topics")
def set_topics(req: TopicSetRequest, store: MemoryStoreDep) -> dict:
    with http_errors(_STORE_FAILURE):
        store.set_topics(req.memory_scope(), [t.model_dump() for t in req.topics])
    return {"status": "saved"}


@router.post("/regroup")
async def regroup_memories(req: ScopedRequest, store: MemoryStoreDep, defaults: ModelDefaultsDep) -> dict:
    """Propose subject topics for a scope's facts as `{topics, memories}`, nothing is written."""
    with http_errors(_STORE_FAILURE), request_client() as client:
        return await MemoryExtractor(store, client, defaults["memory_model"]).propose_regroup(req.memory_scope())
