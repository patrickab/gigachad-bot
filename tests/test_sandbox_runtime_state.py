"""Staged sandbox state must outlive one request but stay off durable storage until save."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import tarfile

from agent_sandbox import AgentSandboxError, PromptImage, SandboxInvocation, SandboxManifest, SandboxRun
from agent_sandbox.manifest import Execution, Outputs, WorkspaceRef
from agent_sandbox.outputs import (
    DisplayDataEvent,
    ErrorEvent,
    ExecuteResultEvent,
    MimeBundle,
    StreamEvent,
    UpdateDisplayDataEvent,
)
import pytest

from config import TEST_MODEL
from lib.agent_sandbox_adapter import FakeSandboxRunner
from lib.data_store import Entry, Revision, StorageNotFoundError
from lib.sandbox_service import MAX_OUTPUTS, SandboxRuntimeState, SandboxService
from lib.sandbox_store import ActivePointer, SandboxStore
from lib.storage_namespace import sandbox_asset


class MemoryDataStore:
    def __init__(self) -> None:
        self._documents: dict[str, bytes] = {}

    def read_bytes(self, key: str) -> tuple[bytes, Revision]:
        try:
            content = self._documents[key]
        except KeyError as exc:
            raise StorageNotFoundError(key) from exc
        return content, Revision(hashlib.sha256(content).hexdigest())

    def write_bytes(self, key: str, content: bytes, *, expected: Revision | None = None) -> Revision:
        del expected
        self._documents[key] = content
        return Revision(hashlib.sha256(content).hexdigest())

    def list(self, prefix: str = "", *, recursive: bool = False) -> list[Entry]:
        del recursive
        scope = f"{prefix}/" if prefix else ""
        return [Entry(key=key, is_dir=False, size=len(value)) for key, value in self._documents.items() if key.startswith(scope)]

    def exists(self, key: str) -> bool:
        return key in self._documents

    def mkdir(self, key: str) -> None:
        del key

    def delete(self, key: str, *, recursive: bool = False) -> None:
        del recursive
        for existing in [k for k in self._documents if k == key or k.startswith(f"{key}/")]:
            del self._documents[existing]

    def move(self, source: str, destination: str) -> None:
        self._documents[destination] = self._documents.pop(source)


@dataclass(frozen=True)
class _Asset:
    content: bytes


class MemoryAssetStore:
    def __init__(self) -> None:
        self._assets: dict[str, bytes] = {}

    def read(self, key: str) -> _Asset:
        try:
            return _Asset(self._assets[key])
        except KeyError as exc:
            raise StorageNotFoundError(key) from exc

    def list(self, key: str) -> list[str]:
        return [key] if key in self._assets else []

    def write(self, kind: str, key: str, content: bytes, *, mime: str) -> None:
        del kind, mime
        self._assets[key] = content

    def delete(self, key: str) -> None:
        self._assets.pop(key, None)


def _service(data_store: MemoryDataStore, assets: MemoryAssetStore, runtime: SandboxRuntimeState) -> SandboxService:
    return SandboxService(
        data_store=data_store,
        asset_store=assets,
        runner=FakeSandboxRunner(),
        runtime_state=runtime,
        model=TEST_MODEL,
        profile="gigachad",
        slot_prefix="user-hash",
    )


@pytest.mark.asyncio
async def test_request_scoped_services_share_one_runtime_state_until_checkpoint() -> None:
    data_store, assets, runtime = MemoryDataStore(), MemoryAssetStore(), SandboxRuntimeState()
    chat_id = "chat-shared-runtime"

    first = await _service(data_store, assets, runtime).invoke(chat_id=chat_id, tool_call_id="call-1", prompt="create")
    assert SandboxStore(data_store).read_active(chat_id) is None

    # A later request gets a new facade, so continuing the workspace proves shared runtime state.
    second_facade = _service(data_store, assets, runtime)
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


class WorkspaceFileRun:
    def __init__(self, content: bytes | Exception) -> None:
        self.content = content
        self.requests: list[str] = []

    def read_workspace_file(self, relative_path: str) -> bytes:
        self.requests.append(relative_path)
        if isinstance(self.content, Exception):
            raise self.content
        return self.content


def test_plot_workspace_artifact_is_promoted_from_the_native_reader() -> None:
    content = b'{"data": [], "layout": {}}'
    run = WorkspaceFileRun(content)

    assert SandboxService._plot_bytes(run) == content  # type: ignore[arg-type]
    assert run.requests == ["plot.json"]


@pytest.mark.parametrize(
    "artifact",
    [
        pytest.param(AgentSandboxError("plot.json is missing"), id="missing"),
        pytest.param(b"{not json", id="malformed"),
        pytest.param(b"[]", id="non-object"),
        pytest.param(AgentSandboxError("plot.json exceeds 4194304 byte limit"), id="oversized"),
        pytest.param(AgentSandboxError("unsafe member path"), id="unsafe"),
    ],
)
def test_invalid_plot_workspace_artifacts_are_ignored(artifact: bytes | Exception) -> None:
    run = WorkspaceFileRun(artifact)

    assert SandboxService._plot_bytes(run) is None  # type: ignore[arg-type]
    assert run.requests == ["plot.json"]


def test_native_output_projection_preserves_order_ids_and_original_bytes() -> None:
    text = "π" * 8001
    outputs, assets = SandboxService._project_outputs(
        (
            StreamEvent(name="stdout", text=text),
            DisplayDataEvent(MimeBundle({"text/html": "<b>display</b>"}), display_id="display-1"),
            ExecuteResultEvent(MimeBundle({"text/plain": "result"})),
            UpdateDisplayDataEvent(MimeBundle({"text/plain": "update"}), display_id="display-1"),
            ErrorEvent("ValueError", "bad input"),
        )
    )

    assert [output.display_id for output in outputs] == ["output-0", "display-1", "output-2", "display-1", "output-4"]
    assert [output.title for output in outputs] == [
        "stdout",
        "Display output",
        "Execution result",
        "Display update",
        "Execution error",
    ]
    text_ref = outputs[0].mime_bundle["text/plain"]
    assert text_ref["size_bytes"] == len(text.encode("utf-8"))
    assert assets[text_ref["sha256"]] == text.encode("utf-8")
    assert outputs[0].text == text[:8000] + "\u2026 [truncated]"
    assert outputs[-1].text == "ValueError: bad input"


@pytest.mark.asyncio
async def test_invoke_forwards_native_prompt_images_unchanged() -> None:
    data_store, assets, runtime = MemoryDataStore(), MemoryAssetStore(), SandboxRuntimeState()
    runner = FakeSandboxRunner()
    service = SandboxService(
        data_store=data_store,
        asset_store=assets,
        runner=runner,
        runtime_state=runtime,
        model=TEST_MODEL,
        profile="gigachad",
    )
    image = PromptImage("chart.png", b"image")

    await service.invoke(chat_id="chat-image", tool_call_id="image-call", prompt="describe", prompt_images=(image,))

    assert runner.calls[-1].prompt_images == (image,)
    assert isinstance(runner.calls[-1].prompt_images[0], PromptImage)
    assert image.content not in assets._assets.values()


@pytest.mark.asyncio
async def test_separate_runtime_states_do_not_share_staged_workspaces() -> None:
    data_store, assets = MemoryDataStore(), MemoryAssetStore()
    chat_id = "chat-isolated-runtime"

    await _service(data_store, assets, SandboxRuntimeState()).invoke(chat_id=chat_id, tool_call_id="call-1", prompt="create")
    runner = FakeSandboxRunner()
    isolated = SandboxService(
        data_store=data_store,
        asset_store=assets,
        runner=runner,
        runtime_state=SandboxRuntimeState(),
        model=TEST_MODEL,
        slot_prefix="user-hash",
    )
    await isolated.invoke(chat_id=chat_id, tool_call_id="call-2", prompt="revise")

    assert runner.calls[-1].active_manifest is None


def _archive(content: bytes) -> bytes:
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        member = tarfile.TarInfo("workspace.txt")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    return archive_bytes.getvalue()


def _manifest(manifest_id: str, workspace_bytes: bytes) -> SandboxManifest:
    sha256 = hashlib.sha256(workspace_bytes).hexdigest()
    return SandboxManifest(
        schema_version=1,
        manifest_id=manifest_id,
        profile="gigachad",
        runtime_fingerprint="test",
        workspace=WorkspaceRef(snapshot_asset_id=None, sha256=sha256),
        outputs=Outputs(events_asset_id=None, artifacts=()),
        omp_sessions={},
        execution=Execution(run_id=manifest_id, status="completed", exit_code=0, created_at=""),
    )


def test_accepting_many_outputs_persists_all_assets_and_artifacts(tmp_path: Path) -> None:
    workspace_bytes = _archive(b"after\n")
    capture_path = tmp_path / "many.tar"
    capture_path.write_bytes(workspace_bytes)
    run = SandboxRun(
        status="completed",
        summary="done",
        next_manifest=_manifest("many", workspace_bytes),
        outputs=tuple(StreamEvent(name="stdout", text=f"output {index}") for index in range(MAX_OUTPUTS + 1)),
        workspace_changed=True,
        capture_path=str(capture_path),
    )
    assets, runtime = MemoryAssetStore(), SandboxRuntimeState()
    service = _service(MemoryDataStore(), assets, runtime)
    staged = runtime.stage("workspace:many")

    result = service._accept_run(run, staged, tool_call_id="many")

    assert len(result.outputs) == MAX_OUTPUTS
    assert len(staged.output_shas) == MAX_OUTPUTS + 1
    assert len(staged.manifest.outputs.artifacts) == MAX_OUTPUTS + 1  # type: ignore[union-attr]
    assert {ref["asset_path"] for output in result.outputs for ref in output.mime_bundle.values()} <= set(assets._assets)
    assert len(assets._assets) == MAX_OUTPUTS + 1
    run.release()


def test_asset_write_failure_does_not_mutate_staged_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("write failed")

    previous_bytes = _archive(b"before\n")
    next_bytes = _archive(b"after\n")
    capture_path = tmp_path / "failure.tar"
    capture_path.write_bytes(next_bytes)
    run = SandboxRun(
        status="completed",
        summary="done",
        next_manifest=_manifest("next", next_bytes),
        outputs=(StreamEvent(name="stdout", text="output"),),
        workspace_changed=True,
        capture_path=str(capture_path),
    )
    runtime = SandboxRuntimeState()
    staged = runtime.stage("workspace:failure")
    staged.manifest = _manifest("previous", previous_bytes)
    staged.assets = {staged.manifest.workspace.sha256: previous_bytes}
    staged.output_shas = {"previous-output"}
    staged.last_completed_run_id = "previous"
    state = (staged.manifest, staged.assets.copy(), staged.output_shas.copy(), staged.last_completed_run_id)
    assets = MemoryAssetStore()
    monkeypatch.setattr(assets, "write", fail_write)
    service = _service(MemoryDataStore(), assets, runtime)

    with pytest.raises(OSError, match="write failed"):
        service._accept_run(run, staged, tool_call_id="next")

    assert (staged.manifest, staged.assets, staged.output_shas, staged.last_completed_run_id) == state
    run.release()


class NativeRunnerStub:
    def __init__(self, capture_path: Path, manifest: SandboxManifest) -> None:
        self.capture_path = capture_path
        self.manifest = manifest
        self.invocation: SandboxInvocation | None = None
        self.restore_bytes: bytes | None = None

    async def run(self, invocation: SandboxInvocation) -> SandboxRun:
        self.invocation = invocation
        assert invocation.workspace_archive_path is not None
        self.restore_bytes = Path(invocation.workspace_archive_path).read_bytes()
        return SandboxRun(
            status="completed",
            summary="done",
            next_manifest=self.manifest,
            outputs=(StreamEvent(name="stdout", text="durable output"),),
            workspace_changed=True,
            capture_path=str(self.capture_path),
        )


@pytest.mark.asyncio
async def test_native_run_materializes_restore_persists_outputs_and_releases_leases(tmp_path: Path) -> None:
    data_store, assets, runtime = MemoryDataStore(), MemoryAssetStore(), SandboxRuntimeState()
    prior_bytes = _archive(b"before\n")
    prior_sha256 = hashlib.sha256(prior_bytes).hexdigest()
    next_bytes = _archive(b"after\n")
    next_sha256 = hashlib.sha256(next_bytes).hexdigest()
    capture_path = tmp_path / "next.tar"
    capture_path.write_bytes(next_bytes)
    prior_manifest = SandboxManifest(
        schema_version=1,
        manifest_id="prior",
        profile="gigachad",
        runtime_fingerprint="test",
        workspace=WorkspaceRef(snapshot_asset_id=sandbox_asset(prior_sha256), sha256=prior_sha256),
        outputs=Outputs(events_asset_id=None, artifacts=()),
        omp_sessions={},
        execution=Execution(run_id="prior-call", status="completed", exit_code=0, created_at=""),
    )
    next_manifest = SandboxManifest(
        schema_version=1,
        manifest_id="next",
        profile="gigachad",
        runtime_fingerprint="test",
        workspace=WorkspaceRef(snapshot_asset_id=None, sha256=next_sha256),
        outputs=Outputs(events_asset_id=None, artifacts=()),
        omp_sessions={},
        execution=Execution(run_id="next-call", status="completed", exit_code=0, created_at=""),
    )
    runner = NativeRunnerStub(capture_path, next_manifest)
    service = SandboxService(
        data_store=data_store,
        asset_store=assets,
        runner=runner,  # type: ignore[arg-type]
        runtime_state=runtime,
        model=TEST_MODEL,
        profile="gigachad",
    )
    staged = runtime.stage("workspace:chat-native")
    staged.manifest = prior_manifest
    staged.assets[prior_sha256] = prior_bytes

    result = await service.invoke(chat_id="chat-native", tool_call_id="next-call", prompt="continue")

    assert runner.invocation is not None
    assert runner.invocation.active_manifest == prior_manifest
    assert runner.restore_bytes == prior_bytes
    assert not Path(runner.invocation.workspace_archive_path).exists()
    assert not capture_path.exists()
    assert assets.read(result.outputs[0].mime_bundle["text/plain"]["asset_path"]).content == b"durable output"
    staged = runtime.get("workspace:chat-native")
    assert staged is not None and staged.manifest is not None
    assert staged.manifest.workspace.snapshot_asset_id == sandbox_asset(next_sha256)
    assert sandbox_asset(next_sha256) not in assets._assets


@pytest.mark.asyncio
async def test_cancellation_defers_resumed_workspace_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prior_bytes = _archive(b"before\n")
    prior_sha256 = hashlib.sha256(prior_bytes).hexdigest()
    next_bytes = _archive(b"after\n")
    next_sha256 = hashlib.sha256(next_bytes).hexdigest()
    capture_path = tmp_path / "next.tar"
    capture_path.write_bytes(next_bytes)

    def manifest(manifest_id: str, sha256: str, run_id: str) -> SandboxManifest:
        return SandboxManifest(
            schema_version=1,
            manifest_id=manifest_id,
            profile="gigachad",
            runtime_fingerprint="test",
            workspace=WorkspaceRef(snapshot_asset_id=None, sha256=sha256),
            outputs=Outputs(events_asset_id=None, artifacts=()),
            omp_sessions={},
            execution=Execution(run_id=run_id, status="completed", exit_code=0, created_at=""),
        )

    started = asyncio.Event()
    finish = asyncio.Event()

    class BlockingNativeRunner(NativeRunnerStub):
        async def run(self, invocation: SandboxInvocation) -> SandboxRun:
            self.invocation = invocation
            started.set()
            await finish.wait()
            return await super().run(invocation)

    runner = BlockingNativeRunner(capture_path, manifest("next", next_sha256, "cancelled-call"))
    runtime = SandboxRuntimeState()
    staged = runtime.stage("workspace:chat-cancelled")
    staged.manifest = manifest("prior", prior_sha256, "prior-call")
    staged.assets[prior_sha256] = prior_bytes
    released = asyncio.Event()
    release = SandboxRun.release
    release_calls: list[SandboxRun] = []

    def record_release(run: SandboxRun) -> None:
        release_calls.append(run)
        release(run)
        released.set()

    monkeypatch.setattr(SandboxRun, "release", record_release)
    service = SandboxService(
        data_store=MemoryDataStore(),
        asset_store=MemoryAssetStore(),
        runner=runner,  # type: ignore[arg-type]
        runtime_state=runtime,
        model=TEST_MODEL,
    )
    task = asyncio.create_task(service.invoke(chat_id="chat-cancelled", tool_call_id="cancelled-call", prompt="continue"))
    try:
        await asyncio.wait_for(started.wait(), 1.0)
        assert runner.invocation is not None and runner.invocation.workspace_archive_path is not None
        restore_path = Path(runner.invocation.workspace_archive_path)
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


@pytest.mark.parametrize("failure", (StorageNotFoundError, ValueError))
@pytest.mark.asyncio
async def test_active_manifest_read_failures_are_invalid(monkeypatch: pytest.MonkeyPatch, failure: type[Exception]) -> None:
    data_store = MemoryDataStore()
    SandboxStore(data_store).write_active("chat-invalid-active", ActivePointer(manifest_id="invalid", updated_at=1.0))

    def fail_read(*_args: object, **_kwargs: object) -> SandboxManifest:
        raise failure("unavailable")

    monkeypatch.setattr(SandboxStore, "read_manifest", fail_read)
    result = await _service(data_store, MemoryAssetStore(), SandboxRuntimeState()).invoke(
        chat_id="chat-invalid-active", tool_call_id="invalid-active", prompt="continue"
    )

    assert result.status == "failed"
    assert result.error == "invalid_manifest"


@pytest.mark.asyncio
async def test_native_runner_failures_stay_model_safe() -> None:
    class FailingRunner:
        async def run(self, invocation: SandboxInvocation) -> SandboxRun:
            del invocation
            raise AgentSandboxError("runtime detail")

    service = SandboxService(
        data_store=MemoryDataStore(),
        asset_store=MemoryAssetStore(),
        runner=FailingRunner(),  # type: ignore[arg-type]
        runtime_state=SandboxRuntimeState(),
        model=TEST_MODEL,
    )

    result = await service.invoke(chat_id="chat-fail", tool_call_id="call-fail", prompt="fail")

    assert result.status == "failed"
    assert result.error == "runner_error"
    assert "runtime detail" not in result.summary


@pytest.mark.asyncio
async def test_invalid_native_capture_is_released_after_failure(tmp_path: Path) -> None:
    capture_path = tmp_path / "invalid.tar"
    capture_path.write_bytes(_archive(b"capture"))

    class InvalidCaptureRunner:
        async def run(self, invocation: SandboxInvocation) -> SandboxRun:
            return SandboxRun(
                status="completed",
                summary="done",
                next_manifest=SandboxManifest(
                    schema_version=1,
                    manifest_id="bad-capture",
                    profile=invocation.profile_name,
                    runtime_fingerprint="test",
                    workspace=WorkspaceRef(snapshot_asset_id=None, sha256="0" * 64),
                    outputs=Outputs(events_asset_id=None, artifacts=()),
                    omp_sessions={},
                    execution=Execution(run_id=invocation.run_id, status="completed", exit_code=0, created_at=""),
                ),
                outputs=(),
                workspace_changed=True,
                capture_path=str(capture_path),
            )

    service = SandboxService(
        data_store=MemoryDataStore(),
        asset_store=MemoryAssetStore(),
        runner=InvalidCaptureRunner(),  # type: ignore[arg-type]
        runtime_state=SandboxRuntimeState(),
        model=TEST_MODEL,
    )

    result = await service.invoke(chat_id="chat-invalid", tool_call_id="call-invalid", prompt="continue")

    assert result.error == "invalid_manifest"
    assert not capture_path.exists()


@pytest.mark.asyncio
async def test_native_release_failure_logs_without_invalidating_staged_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    capture_bytes = _archive(b"capture")
    capture_path = tmp_path / "release.tar"
    capture_path.write_bytes(capture_bytes)
    native_run = SandboxRun(
        status="completed",
        summary="done",
        next_manifest=SandboxManifest(
            schema_version=1,
            manifest_id="release-failure",
            profile="gigachad",
            runtime_fingerprint="test",
            workspace=WorkspaceRef(snapshot_asset_id=None, sha256=hashlib.sha256(capture_bytes).hexdigest()),
            outputs=Outputs(events_asset_id=None, artifacts=()),
            omp_sessions={},
            execution=Execution(run_id="call-release", status="completed", exit_code=0, created_at=""),
        ),
        outputs=(StreamEvent(name="stdout", text="output"),),
        workspace_changed=True,
        capture_path=str(capture_path),
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
    service = SandboxService(
        data_store=MemoryDataStore(),
        asset_store=MemoryAssetStore(),
        runner=ReleaseFailingRunner(),  # type: ignore[arg-type]
        runtime_state=runtime,
        model=TEST_MODEL,
    )

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
