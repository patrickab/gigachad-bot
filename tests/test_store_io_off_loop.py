"""Async services run their blocking store and asset IO in worker threads, never on the event loop."""

import asyncio
from collections.abc import Callable
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from lib import architecture_service, c4_service
from lib.architecture_service import Change
from lib.architecture_workspace import Snapshot
from lib.memory_extract import MemoryExtractor
from lib.memory_store import MemoryScope, MemoryStore
from lib.sandbox_store import SandboxStore
from lib.storage_namespace import sandbox_asset
from tests.fakes import MemoryAssetStore, MemoryDataStore, memory_sandbox_service
from tests.test_memory_store import GLOBAL, ScriptedLLM, seed


def on_loop_calls(monkeypatch: pytest.MonkeyPatch, cls: type, *names: str) -> list[tuple[str, bool]]:
    """Record `(method, ran on the event loop thread)` for every call of *names* on *cls*."""
    calls: list[tuple[str, bool]] = []

    def recorded(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            calls.append((name, threading.current_thread() is threading.main_thread()))
            return original(*args, **kwargs)

        return wrapper

    for name in names:
        monkeypatch.setattr(cls, name, recorded(name, getattr(cls, name)))
    return calls


class FakeArchitectureStore:
    """Two reads (the first commit loses a race) and the commits, each noting its thread."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.conflicts = 1

    def _note(self, name: str) -> None:
        self.calls.append((name, threading.current_thread() is threading.main_thread()))

    def source_key(self, relative: str) -> str:
        return relative

    def read(self, *, allow_empty: bool = False) -> Snapshot:
        self._note("read")
        return Snapshot({}, {}, {})

    def commit(self, snapshot: Snapshot, sources: dict[str, str], snapshots: dict[str, str], deleted: Any = ()) -> None:
        self._note("commit")
        if self.conflicts:
            self.conflicts -= 1
            raise architecture_service.StorageConflictError("raced")


def test_architecture_apply_reads_checks_and_commits_off_the_loop(monkeypatch):
    result = SimpleNamespace(sources={"a.c4": "model {}"}, snapshots={}, model=SimpleNamespace(errors=[]))

    async def apply_operations(*args: Any) -> SimpleNamespace:
        return result

    monkeypatch.setattr(c4_service, "apply_operations", apply_operations)
    store = FakeArchitectureStore()

    applied = asyncio.run(architecture_service.apply(
        store, Change([]), check_project=lambda: store._note("check_project"),  # type: ignore[arg-type]
    ))

    assert applied is result
    assert store.calls == [("read", False), ("check_project", False), ("commit", False)] * 2


def test_memory_extract_and_regroup_touch_the_store_off_the_loop(data_store, monkeypatch):
    store = MemoryStore(data_store)
    seed(store)
    calls = on_loop_calls(
        monkeypatch, MemoryStore, "ensure_pending_dir", "watermark", "get_topics", "read_memories", "write_pending"
    )
    add = {"operations": [{"op": "add", "topic": "profile", "text": "Runs iwd"}], "topics": []}
    regroup = {"topics": [{"name": "profile", "summary": "Who the user is"}], "assignments": [{"id": "m1", "topic": "profile"}]}
    extractor = MemoryExtractor(store, ScriptedLLM(add, regroup), "test")

    result = asyncio.run(extractor.extract([{"role": "user", "content": "hi"}], chat_id="c1"))
    regrouped = asyncio.run(extractor.propose_regroup(GLOBAL))

    assert [o.text for o in result.global_proposal.operations] == ["Runs iwd"]
    assert {m["topic"] for m in regrouped["memories"]} == {"profile", "preferences"}
    assert [name for name, _ in calls] == [
        "ensure_pending_dir", "watermark", "get_topics", "read_memories", "write_pending",
        "read_memories", "get_topics",
    ]
    assert not any(on_loop for _, on_loop in calls)


def test_memory_regroup_of_an_empty_scope_reads_off_the_loop(data_store, monkeypatch):
    store = MemoryStore(data_store)
    calls = on_loop_calls(monkeypatch, MemoryStore, "get_topics", "read_memories")

    regrouped = asyncio.run(MemoryExtractor(store, ScriptedLLM(), "test").propose_regroup(MemoryScope("project", "shop")))

    assert regrouped == {"topics": store.get_topics(MemoryScope("project", "shop")), "memories": []}
    assert calls[:2] == [("read_memories", False), ("get_topics", False)]


@pytest.mark.asyncio
async def test_sandbox_checkpoint_and_delete_touch_storage_off_the_loop(monkeypatch):
    data_store, assets = MemoryDataStore(), MemoryAssetStore()
    service = memory_sandbox_service(data_store=data_store, assets=assets)
    await service.invoke(chat_id="chat-1", tool_call_id="call-1", prompt="create")
    store_calls = on_loop_calls(monkeypatch, SandboxStore, "checkpoint", "delete_chat_state", "referenced_shas")
    asset_writes = on_loop_calls(monkeypatch, MemoryAssetStore, "write")

    await service.checkpoint(chat_id="chat-1")
    active = SandboxStore(data_store).read_active("chat-1")
    assert active is not None
    workspace_sha = SandboxStore(data_store).read_manifest("chat-1", active.manifest_id).workspace.sha256

    await service.delete_chat_state(chat_id="chat-1")

    assert SandboxStore(data_store).read_active("chat-1") is None
    assert not assets.list(sandbox_asset(workspace_sha))
    # Deleting without a scope visits every known scope; only the one holding state checks references.
    assert [name for name, _ in store_calls][:3] == ["checkpoint", "delete_chat_state", "referenced_shas"]
    assert not any(on_loop for _, on_loop in store_calls)
    assert asset_writes and not any(on_loop for _, on_loop in asset_writes)
