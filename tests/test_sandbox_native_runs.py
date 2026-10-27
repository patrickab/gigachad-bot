"""A native sandbox run restores the staged workspace, persists outputs and always releases its files."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from agent_sandbox import AgentSandboxError, SandboxInvocation, SandboxManifest, SandboxRun, StreamEvent, WorkspaceArchive
import pytest

from lib.sandbox_service import SandboxRuntimeState
from lib.storage_namespace import sandbox_asset
from tests.fakes import MemoryAssetStore, memory_sandbox_service, native_manifest, tar_archive


class NativeRunnerStub:
    def __init__(self, capture_path: Path, manifest: SandboxManifest) -> None:
        self.capture_path = capture_path
        self.manifest = manifest
        self.invocation: SandboxInvocation | None = None
        self.restore_bytes: bytes | None = None

    async def run(self, invocation: SandboxInvocation) -> SandboxRun:
        self.invocation = invocation
        assert invocation.workspace is not None
        self.restore_bytes = Path(invocation.workspace.path).read_bytes()
        return SandboxRun(
            status="completed",
            summary="done",
            next_manifest=self.manifest,
            outputs=(StreamEvent(name="stdout", text="durable output"),),
            workspace_changed=True,
            workspace=WorkspaceArchive.from_file(str(self.capture_path)),
        )


@pytest.mark.asyncio
async def test_native_run_materializes_restore_persists_outputs_and_releases_leases(tmp_path: Path) -> None:
    assets, runtime = MemoryAssetStore(), SandboxRuntimeState()
    prior_bytes = tar_archive(b"before\n")
    prior_sha256 = hashlib.sha256(prior_bytes).hexdigest()
    next_bytes = tar_archive(b"after\n")
    next_sha256 = hashlib.sha256(next_bytes).hexdigest()
    capture_path = tmp_path / "next.tar"
    capture_path.write_bytes(next_bytes)
    prior_manifest = native_manifest("prior", prior_sha256, run_id="prior-call", snapshot_asset_id=sandbox_asset(prior_sha256))
    runner = NativeRunnerStub(capture_path, native_manifest("next", next_sha256, run_id="next-call"))
    service = memory_sandbox_service(runner, assets=assets, runtime=runtime)
    staged = runtime.stage("workspace:chat-native")
    staged.manifest = prior_manifest
    staged.assets[prior_sha256] = prior_bytes

    result = await service.invoke(chat_id="chat-native", tool_call_id="next-call", prompt="continue")

    assert runner.invocation is not None
    assert runner.invocation.active_manifest == prior_manifest
    assert runner.restore_bytes == prior_bytes
    assert not Path(runner.invocation.workspace.path).exists()
    assert not capture_path.exists()
    assert assets.read(result.outputs[0].mime_bundle["text/plain"]["asset_path"]).content == b"durable output"
    staged = runtime.get("workspace:chat-native")
    assert staged is not None and staged.manifest is not None
    assert staged.manifest.workspace.snapshot_asset_id == sandbox_asset(next_sha256)
    assert sandbox_asset(next_sha256) not in assets.assets


@pytest.mark.asyncio
async def test_cancellation_defers_resumed_workspace_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prior_bytes = tar_archive(b"before\n")
    prior_sha256 = hashlib.sha256(prior_bytes).hexdigest()
    next_bytes = tar_archive(b"after\n")
    next_sha256 = hashlib.sha256(next_bytes).hexdigest()
    capture_path = tmp_path / "next.tar"
    capture_path.write_bytes(next_bytes)

    started = asyncio.Event()
    finish = asyncio.Event()

    class BlockingNativeRunner(NativeRunnerStub):
        async def run(self, invocation: SandboxInvocation) -> SandboxRun:
            self.invocation = invocation
            started.set()
            await finish.wait()
            return await super().run(invocation)

    runner = BlockingNativeRunner(capture_path, native_manifest("next", next_sha256, run_id="cancelled-call"))
    runtime = SandboxRuntimeState()
    staged = runtime.stage("workspace:chat-cancelled")
    staged.manifest = native_manifest("prior", prior_sha256, run_id="prior-call")
    staged.assets[prior_sha256] = prior_bytes
    released = asyncio.Event()
    release = SandboxRun.release
    release_calls: list[SandboxRun] = []

    def record_release(run: SandboxRun) -> None:
        release_calls.append(run)
        release(run)
        released.set()

    monkeypatch.setattr(SandboxRun, "release", record_release)
    service = memory_sandbox_service(runner, runtime=runtime)
    task = asyncio.create_task(service.invoke(chat_id="chat-cancelled", tool_call_id="cancelled-call", prompt="continue"))
    try:
        await asyncio.wait_for(started.wait(), 1.0)
        assert runner.invocation is not None and runner.invocation.workspace is not None
        restore_path = Path(runner.invocation.workspace.path)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert restore_path.exists()
    finally:
        finish.set()

    await asyncio.wait_for(released.wait(), 1.0)
    assert len(release_calls) == 1
    assert runner.restore_bytes == prior_bytes
    assert not restore_path.exists()
    assert not capture_path.exists()


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


@pytest.mark.asyncio
async def test_native_release_failure_logs_without_invalidating_staged_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    capture_bytes = tar_archive(b"capture")
    capture_path = tmp_path / "release.tar"
    capture_path.write_bytes(capture_bytes)
    native_run = SandboxRun(
        status="completed",
        summary="done",
        next_manifest=native_manifest("release-failure", hashlib.sha256(capture_bytes).hexdigest(), run_id="call-release"),
        outputs=(StreamEvent(name="stdout", text="output"),),
        workspace_changed=True,
        workspace=WorkspaceArchive.from_file(str(capture_path)),
    )

    class ReleaseFailingRunner:
        async def run(self, invocation: SandboxInvocation) -> SandboxRun:
            del invocation
            return native_run

    release_calls: list[SandboxRun] = []

    def fail_release(run: SandboxRun) -> None:
        release_calls.append(run)
        raise OSError("release failed")

    monkeypatch.setattr(SandboxRun, "release", fail_release)
    runtime = SandboxRuntimeState()
    service = memory_sandbox_service(ReleaseFailingRunner(), runtime=runtime)

    result = await service.invoke(chat_id="chat-release", tool_call_id="call-release", prompt="continue")

    staged = runtime.get("workspace:chat-release")
    assert result.status == "completed"
    assert result.error is None
    assert result.manifest_id == "release-failure"
    assert release_calls == [native_run]
    assert staged is not None and staged.manifest is not None
    assert staged.manifest.manifest_id == "release-failure"
    assert staged.runs["call-release"].result is not None
    assert staged.runs["call-release"].result["error"] is None
    release_logs = [
        record
        for record in caplog.records
        if record.getMessage() == "sandbox run call-release release failed for slot workspace:chat-release"
    ]
    assert len(release_logs) == 1
    capture_path.unlink()
