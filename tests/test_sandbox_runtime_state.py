"""Staged sandbox state must outlive one request but stay off durable storage until save."""

import pytest

from lib.sandbox_service import SandboxRuntimeState, SandboxService
from lib.sandbox_store import SandboxStore
from lib.storage_namespace import sandbox_asset
from tests.fakes import MemoryAssetStore, MemoryDataStore, memory_sandbox_service


@pytest.mark.asyncio
async def test_request_scoped_services_share_one_runtime_state_until_checkpoint() -> None:
    data_store, assets, runtime = MemoryDataStore(), MemoryAssetStore(), SandboxRuntimeState()
    chat_id = "chat-shared-runtime"

    def facade() -> SandboxService:
        return memory_sandbox_service(data_store=data_store, assets=assets, runtime=runtime, slot_prefix="user-hash")

    first = await facade().invoke(chat_id=chat_id, tool_call_id="call-1", prompt="create")
    assert SandboxStore(data_store).read_active(chat_id) is None

    # A later request gets a new facade, so continuing the workspace proves shared runtime state.
    second_facade = facade()
    second = await second_facade.invoke(chat_id=chat_id, tool_call_id="call-2", prompt="revise")
    assert second.manifest_id != first.manifest_id
    assert second.workspace_changed is True

    await second_facade.checkpoint(chat_id=chat_id)

    active = SandboxStore(data_store).read_active(chat_id)
    assert active is not None
    assert active.manifest_id == second.manifest_id
    manifest = SandboxStore(data_store).read_manifest(chat_id, active.manifest_id)
    assert manifest.workspace is not None
    assert assets.read(sandbox_asset(manifest.workspace.sha256)).content
    # Checkpoint discards the staged slot, leaving durable records as the only source.
    assert runtime.get("user-hash:workspace:chat-shared-runtime") is None
