"""A workspace-agent run is durable when it returns, whichever worker serves the chat save."""

import pytest

from lib.sandbox_service import SandboxRuntimeState, SandboxService
from lib.sandbox_store import SandboxStore
from tests.fakes import FakeSandboxRunner, MemoryAssetStore, MemoryDataStore, memory_sandbox_service


@pytest.mark.asyncio
async def test_a_run_survives_a_save_on_another_worker() -> None:
    data_store, assets = MemoryDataStore(), MemoryAssetStore()

    def worker() -> SandboxService:  # each worker has its own process state
        return memory_sandbox_service(data_store=data_store, assets=assets, runtime=SandboxRuntimeState())

    first = await worker().invoke(chat_id="chat-1", tool_call_id="call-1", prompt="create")

    active = SandboxStore(data_store).read_active("chat-1")
    assert active is not None and active.manifest_id == first.manifest_id

    runner = FakeSandboxRunner()
    await memory_sandbox_service(runner, data_store=data_store, assets=assets, runtime=SandboxRuntimeState()).invoke(
        chat_id="chat-1", tool_call_id="call-2", prompt="revise"
    )
    assert runner.calls[-1].active_manifest.manifest_id == first.manifest_id
    assert runner.calls[-1].workspace is not None  # the stored tar was restored
