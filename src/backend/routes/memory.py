"""The memory window's REST surface: list, read and change the login's memory-store root through `memory_store.store`."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from memory_store import store
from memory_store.paths import inside_root, is_document, split_path, store_dir, target_file
from pydantic import BaseModel

from backend.identity import RequestIdentity, get_request_identity
from lib.memory_context import memory_root

if TYPE_CHECKING:
    from collections.abc import Iterator

router = APIRouter(prefix="/api/memory", tags=["memory"])

_NOT_FOUND_CODES = frozenset({"not_found", "memory_store_missing"})
_FACT_FIELDS = ("id", "text", "created", "updated")


def get_memory_root(identity: Annotated[RequestIdentity, Depends(get_request_identity)]) -> Path:
    return memory_root(identity.user_id, identity.has_host_access)


MemoryRootDep = Annotated[Path, Depends(get_memory_root)]


class MemoryApplyRequest(BaseModel):
    memory_store: str
    ops: list[dict[str, Any]]


@contextmanager
def _store_errors() -> Iterator[None]:
    """`MemoryStoreError` as 404 (missing memory_store or entry) or 400, the detail `{code, message}`."""
    try:
        yield
    except store.MemoryStoreError as exc:
        status = 404 if exc.code in _NOT_FOUND_CODES else 400
        raise HTTPException(status_code=status, detail={"code": exc.code, "message": exc.message}) from exc


def _read_document(root: Path, memory_store: str, path: str) -> str:
    """A document's text, held to the root like the library's reads: a symlink out of it is `invalid_path`."""
    file = inside_root(root, target_file(store_dir(root, memory_store), split_path(path)))
    try:
        return file.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        raise store.MemoryStoreError("not_found", f"no document {path}") from None


# Plain `def` routes: FastAPI runs them in its thread pool, so the blocking file I/O stays off the event loop.
@router.get("/list")
def list_memories(root: MemoryRootDep, memory_store: str = "", path: str = "") -> dict:
    with _store_errors():
        return store.list_memories(memory_store, path, root=root)


@router.get("/read")
def read_memory(root: MemoryRootDep, memory_store: str, path: str) -> dict:
    with _store_errors():
        if is_document(path):
            return {"kind": "document", "text": _read_document(root, memory_store, path)}
        memory = store.read_memory(memory_store, path, root=root)
    facts = [{key: fact.get(key) for key in _FACT_FIELDS} for fact in memory["facts"]]
    return {"kind": "memory", "description": memory["description"], "facts": facts}


@router.post("/apply")
def apply_memory_ops(req: MemoryApplyRequest, root: MemoryRootDep) -> dict:
    with _store_errors():
        return {"results": store.apply_memory_ops(req.memory_store, req.ops, root=root)}
