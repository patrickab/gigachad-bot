"""A native sandbox run restores the staged workspace, persists outputs and always releases its files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_sandbox import AgentSandboxError, SandboxInvocation, SandboxRun, WorkspaceArchive
import pytest

from tests.fakes import memory_sandbox_service, native_manifest, tar_archive

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_native_runner_failures_stay_model_safe() -> None:
    class FailingRunner:
        async def run(self, invocation: SandboxInvocation) -> SandboxRun:
            del invocation
            raise AgentSandboxError("runtime detail")

    result = await memory_sandbox_service(FailingRunner()).invoke(chat_id="chat-fail", tool_call_id="call-fail", prompt="fail")

    assert result.status == "failed"
    assert result.error == "runner_error"
    assert "runtime detail" not in result.summary


@pytest.mark.asyncio
async def test_invalid_native_capture_is_released_after_failure(tmp_path: Path) -> None:
    capture_path = tmp_path / "invalid.tar"
    capture_path.write_bytes(tar_archive(b"capture"))

    class InvalidCaptureRunner:
        async def run(self, invocation: SandboxInvocation) -> SandboxRun:
            return SandboxRun(
                status="completed",
                summary="done",
                next_manifest=native_manifest("bad-capture", "0" * 64, run_id=invocation.run_id, profile=invocation.profile_name),
                outputs=(),
                workspace_changed=True,
                workspace=WorkspaceArchive.from_file(str(capture_path)),
            )

    service = memory_sandbox_service(InvalidCaptureRunner())

    result = await service.invoke(chat_id="chat-invalid", tool_call_id="call-invalid", prompt="continue")

    assert result.error == "invalid_manifest"
    assert not capture_path.exists()
