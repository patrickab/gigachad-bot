from types import SimpleNamespace
from uuid import uuid4

import pytest

from backend.routes import deps
from lib.fake_sandbox_runner import FakeSandboxRunner
from tests.fakes import MemoryAssetStore, MemoryDataStore


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored", "gateway_id"),
    [
        ("litellm_proxy/anthropic/claude-haiku-5-5", "anthropic/claude-haiku-5-5"),
        ("gemini/gemini-3.5-flash", "google/gemini-3.5-flash"),
        ("ollama/gemma3:4b", "ollama/gemma3:4b"),
    ],
)
async def test_sandbox_runs_the_stored_default_as_a_gateway_model_id(monkeypatch, stored, gateway_id):
    """The stored default reaches the sandbox's Pi as its gateway id, like every backend completion."""
    runner = FakeSandboxRunner()
    monkeypatch.setattr(deps, "_sandbox_runner", runner)
    service = deps.get_sandbox_service(
        identity=SimpleNamespace(user_id=uuid4()),
        docs=MemoryDataStore(),
        assets=MemoryAssetStore(),
        defaults={"sandbox_model": stored},
    )

    await service.invoke(chat_id="chat-model", tool_call_id="call-1", prompt="Reply ok")

    assert runner.calls[-1].task.model == gateway_id
