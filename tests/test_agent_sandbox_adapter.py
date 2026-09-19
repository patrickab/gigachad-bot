import hashlib
import io
from pathlib import Path
import tarfile

import pytest

from config import TEST_MODEL
from lib.agent_sandbox_adapter import (
    AgentSandboxRunnerAdapter,
    AssetRef,
    FakeSandboxRunner,
    SandboxInvocation,
    SandboxStateRecord,
)
from lib.image_paths import PromptImage

# The model id is caller-supplied configuration; these tests only check it is forwarded.
SANDBOX_MODEL = TEST_MODEL


def test_real_adapter_promotes_plot_json_from_workspace_archive() -> None:
    content = b'{"data": [], "layout": {}}'
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        entry = tarfile.TarInfo("plot.json")
        entry.size = len(content)
        archive.addfile(entry, io.BytesIO(content))

    output, assets = AgentSandboxRunnerAdapter._plot_output(archive_bytes.getvalue())

    assert output is not None
    ref = output.mime_bundle["application/vnd.plotly.v1+json"]
    assert assets == {ref.sha256: content}


@pytest.mark.asyncio
async def test_real_adapter_restores_archive_captures_workspace_and_releases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_sandbox.invocation import SandboxRun as ExternalRun
    from agent_sandbox.manifest import Execution, Outputs, SandboxManifest, WorkspaceRef
    from agent_sandbox.outputs import StreamEvent

    prior_bytes = b"previous workspace"
    next_bytes = b"next workspace"
    capture = tmp_path / "captured.tar"
    capture.write_bytes(next_bytes)
    calls = []
    restored_bytes = []

    def fake_run(external_invocation):
        calls.append(external_invocation)
        restored_bytes.append(Path(external_invocation.workspace_archive_path).read_bytes())
        return ExternalRun(
            status="completed",
            summary="done",
            next_manifest=SandboxManifest(
                schema_version=1,
                manifest_id="next-manifest",
                profile="test",
                runtime_fingerprint="test",
                workspace=WorkspaceRef(snapshot_asset_id=None, sha256=hashlib.sha256(next_bytes).hexdigest()),
                outputs=Outputs(events_asset_id=None, artifacts=()),
                omp_sessions={},
                execution=Execution(run_id="call-2", status="completed", exit_code=0, created_at=""),
            ),
            outputs=(StreamEvent(name="stdout", text="workspace restored"),),
            workspace_changed=True,
            capture_path=str(capture),
        )

    monkeypatch.setattr("agent_sandbox.invocation.run", fake_run)
    adapter = AgentSandboxRunnerAdapter()
    prior_ref = AssetRef(hashlib.sha256(prior_bytes).hexdigest(), "application/x-tar", len(prior_bytes))
    result = await adapter.run(
        SandboxInvocation(
            slot_key="isolated-chat",
            run_id="call-2",
            profile="test",
            model=SANDBOX_MODEL,
            thinking="medium",
            prompt="trivial OMP command",
            prior_manifest=SandboxStateRecord(version=1, workspace=prior_ref),
            prior_assets={prior_ref.sha256: prior_bytes},
            prompt_images=(PromptImage("photo.png", b"image-bytes"),),
            append_system="follow the caller's policy",
        )
    )

    assert calls[0].workspace_archive_path is not None
    assert restored_bytes == [prior_bytes]
    assert calls[0].active_manifest is not None
    assert calls[0].prompt_images == (("photo.png", b"image-bytes"),)
    assert calls[0].model == SANDBOX_MODEL
    assert calls[0].thinking == "medium"
    assert calls[0].append_system == "follow the caller's policy"
    assert result.manifest.workspace is not None
    assert result.manifest.workspace.sha256 == hashlib.sha256(next_bytes).hexdigest()
    assert result.new_assets[result.manifest.workspace.sha256] == next_bytes
    assert result.outputs[0].text == "workspace restored"
    output_ref = result.outputs[0].mime_bundle["text/plain"]
    assert output_ref.sha256 == hashlib.sha256(b"workspace restored").hexdigest()
    assert result.new_assets[output_ref.sha256] == b"workspace restored"
    assert capture.exists()
    assert not Path(calls[0].workspace_archive_path).exists()
    assert result.release is not None
    result.release()
    assert not capture.exists()


def test_sandbox_dependency_uses_real_runner_unless_explicitly_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.routes import deps

    monkeypatch.setattr(deps, "_sandbox_runner", None)
    monkeypatch.delenv("GIGACHAD_SANDBOX_EXECUTION", raising=False)
    monkeypatch.delenv("GIGACHAD_SANDBOX_FAKE", raising=False)
    assert isinstance(deps.get_sandbox_runner(), AgentSandboxRunnerAdapter)

    monkeypatch.setattr(deps, "_sandbox_runner", None)
    monkeypatch.setenv("GIGACHAD_SANDBOX_FAKE", "true")
    assert isinstance(deps.get_sandbox_runner(), FakeSandboxRunner)
