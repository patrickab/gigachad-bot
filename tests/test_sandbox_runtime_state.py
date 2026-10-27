"""Staged sandbox state must outlive one request but stay off durable storage until save."""

from dataclasses import asdict
import hashlib
from pathlib import Path

from agent_sandbox import AgentSandboxError, ErrorEvent, SandboxManifest, SandboxRun, StreamEvent, WorkspaceArchive
import pytest

from lib.data_store import StorageNotFoundError
from lib.fake_sandbox_runner import FakeSandboxRunner
from lib.json_docs import JsonDocs
from lib.sandbox_service import MAX_OUTPUTS, SandboxRuntimeState, SandboxService
from lib.sandbox_store import ActivePointer, SandboxStore
from lib.storage_namespace import sandbox_active, sandbox_asset
from tests.fakes import MemoryAssetStore, MemoryDataStore, memory_sandbox_service, native_manifest, tar_archive


def _manifest(manifest_id: str, workspace_bytes: bytes) -> SandboxManifest:
    return native_manifest(manifest_id, hashlib.sha256(workspace_bytes).hexdigest())


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


@pytest.mark.asyncio
async def test_separate_runtime_states_do_not_share_staged_workspaces() -> None:
    data_store, assets = MemoryDataStore(), MemoryAssetStore()
    chat_id = "chat-isolated-runtime"

    await memory_sandbox_service(data_store=data_store, assets=assets, slot_prefix="user-hash").invoke(
        chat_id=chat_id, tool_call_id="call-1", prompt="create"
    )
    runner = FakeSandboxRunner()
    isolated = memory_sandbox_service(runner, data_store=data_store, assets=assets, slot_prefix="user-hash")
    await isolated.invoke(chat_id=chat_id, tool_call_id="call-2", prompt="revise")

    assert runner.calls[-1].active_manifest is None


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
            StreamEvent(name="stderr", text="warning"),
            ErrorEvent("ValueError", "bad input"),
        )
    )

    assert [output.display_id for output in outputs] == ["output-0", "output-1", "output-2"]
    assert [output.title for output in outputs] == ["stdout", "stderr", "Execution error"]
    text_ref = outputs[0].mime_bundle["text/plain"]
    assert text_ref["size_bytes"] == len(text.encode("utf-8"))
    assert assets[text_ref["sha256"]] == text.encode("utf-8")
    assert outputs[0].text == text[:8000] + "\u2026 [truncated]"
    assert outputs[-1].text == "ValueError: bad input"


def test_accepting_many_outputs_persists_all_assets_and_artifacts(tmp_path: Path) -> None:
    workspace_bytes = tar_archive(b"after\n")
    capture_path = tmp_path / "many.tar"
    capture_path.write_bytes(workspace_bytes)
    run = SandboxRun(
        status="completed",
        summary="done",
        next_manifest=_manifest("many", workspace_bytes),
        outputs=tuple(StreamEvent(name="stdout", text=f"output {index}") for index in range(MAX_OUTPUTS + 1)),
        workspace_changed=True,
        workspace=WorkspaceArchive.from_file(str(capture_path)),
    )
    assets, runtime = MemoryAssetStore(), SandboxRuntimeState()
    service = memory_sandbox_service(assets=assets, runtime=runtime, slot_prefix="user-hash")
    staged = runtime.stage("workspace:many")

    result = service._accept_run(run, staged, tool_call_id="many")

    assert len(result.outputs) == MAX_OUTPUTS
    assert len(staged.output_shas) == MAX_OUTPUTS + 1
    assert len(staged.manifest.outputs.artifacts) == MAX_OUTPUTS + 1  # type: ignore[union-attr]
    assert {ref["asset_path"] for output in result.outputs for ref in output.mime_bundle.values()} <= set(assets.assets)
    assert len(assets.assets) == MAX_OUTPUTS + 1
    run.release()


def test_asset_write_failure_does_not_mutate_staged_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("write failed")

    previous_bytes = tar_archive(b"before\n")
    next_bytes = tar_archive(b"after\n")
    capture_path = tmp_path / "failure.tar"
    capture_path.write_bytes(next_bytes)
    run = SandboxRun(
        status="completed",
        summary="done",
        next_manifest=_manifest("next", next_bytes),
        outputs=(StreamEvent(name="stdout", text="output"),),
        workspace_changed=True,
        workspace=WorkspaceArchive.from_file(str(capture_path)),
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
    service = memory_sandbox_service(assets=assets, runtime=runtime, slot_prefix="user-hash")

    with pytest.raises(OSError, match="write failed"):
        service._accept_run(run, staged, tool_call_id="next")

    assert (staged.manifest, staged.assets, staged.output_shas, staged.last_completed_run_id) == state
    run.release()


@pytest.mark.parametrize("failure", (StorageNotFoundError, ValueError))
@pytest.mark.asyncio
async def test_active_manifest_read_failures_are_invalid(monkeypatch: pytest.MonkeyPatch, failure: type[Exception]) -> None:
    data_store = MemoryDataStore()
    JsonDocs(data_store).write(sandbox_active("chat-invalid-active"), asdict(ActivePointer(manifest_id="invalid", updated_at=1.0)))

    def fail_read(*_args: object, **_kwargs: object) -> SandboxManifest:
        raise failure("unavailable")

    monkeypatch.setattr(SandboxStore, "read_manifest", fail_read)
    result = await memory_sandbox_service(data_store=data_store, slot_prefix="user-hash").invoke(
        chat_id="chat-invalid-active", tool_call_id="invalid-active", prompt="continue"
    )

    assert result.status == "failed"
    assert result.error == "invalid_manifest"
