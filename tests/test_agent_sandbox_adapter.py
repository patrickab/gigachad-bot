import hashlib
from pathlib import Path

from agent_sandbox import PythonScriptError, SandboxInvocation, SandboxManifest, SandboxRun
from agent_sandbox.manifest import Execution, Outputs, WorkspaceRef
import pytest

from config import TEST_MODEL
from lib.agent_sandbox_adapter import AgentSandboxRunnerAdapter, FakeSandboxRunner, SandboxScriptError

SANDBOX_MODEL = TEST_MODEL


def _manifest(workspace_sha256: str, *, manifest_id: str = "next-manifest") -> SandboxManifest:
    return SandboxManifest(
        schema_version=1,
        manifest_id=manifest_id,
        profile="test",
        runtime_fingerprint="test",
        workspace=WorkspaceRef(snapshot_asset_id=None, sha256=workspace_sha256),
        outputs=Outputs(events_asset_id=None, artifacts=()),
        omp_sessions={},
        execution=Execution(run_id="call-2", status="completed", exit_code=0, created_at=""),
    )


class SandboxStub:
    def __init__(self, result: SandboxRun) -> None:
        self.result = result
        self.invocations: list[SandboxInvocation] = []
        self.script_calls: list[tuple[str, str, float, str]] = []

    def run(self, invocation: SandboxInvocation) -> SandboxRun:
        self.invocations.append(invocation)
        return self.result

    def run_python_script(self, interpreter: str, script: str, timeout: float, *, profile_name: str) -> str:
        self.script_calls.append((interpreter, script, timeout, profile_name))
        return "script output"


@pytest.mark.asyncio
async def test_adapter_bridges_native_invocations_through_one_injected_sandbox(tmp_path: Path) -> None:
    capture = tmp_path / "capture.tar"
    capture.write_bytes(b"captured workspace")
    native_run = SandboxRun(
        status="completed",
        summary="done",
        next_manifest=_manifest(hashlib.sha256(capture.read_bytes()).hexdigest()),
        outputs=(),
        workspace_changed=True,
        capture_path=str(capture),
    )
    sandbox = SandboxStub(native_run)
    adapter = AgentSandboxRunnerAdapter(sandbox)  # type: ignore[arg-type]
    invocation = SandboxInvocation(
        slot_key="isolated-chat",
        profile_name="test",
        prompt="trivial OMP command",
        run_id="call-2",
        model=SANDBOX_MODEL,
        thinking="medium",
    )

    assert await adapter.run(invocation) is native_run
    assert await adapter.run_script("print('ok')", profile="test", interpreter="venv", timeout=12.0) == "script output"
    assert sandbox.invocations == [invocation]
    assert sandbox.script_calls == [("venv", "print('ok')", 12.0, "test")]

    native_run.release()
    assert not capture.exists()


@pytest.mark.asyncio
async def test_adapter_maps_native_script_errors(tmp_path: Path) -> None:
    capture = tmp_path / "capture.tar"
    capture.write_bytes(b"captured workspace")
    native_run = SandboxRun(
        status="completed",
        summary="done",
        next_manifest=_manifest(hashlib.sha256(capture.read_bytes()).hexdigest()),
        outputs=(),
        workspace_changed=True,
        capture_path=str(capture),
    )
    sandbox = SandboxStub(native_run)

    def fail_script(*_args, **_kwargs) -> str:
        raise PythonScriptError("native detail")

    sandbox.run_python_script = fail_script  # type: ignore[method-assign]
    with pytest.raises(SandboxScriptError, match="native detail"):
        await AgentSandboxRunnerAdapter(sandbox).run_script("bad", profile="test", interpreter="venv", timeout=1.0)  # type: ignore[arg-type]

    native_run.release()


@pytest.mark.asyncio
async def test_fake_runner_returns_a_native_capture_lease() -> None:
    runner = FakeSandboxRunner()
    result = await runner.run(
        SandboxInvocation(
            slot_key="fake-chat",
            profile_name="test",
            prompt="create",
            run_id="call-1",
            model=SANDBOX_MODEL,
            thinking="medium",
        )
    )

    capture = Path(result.capture_path)
    assert result.next_manifest.workspace.snapshot_asset_id is None
    assert capture.exists()
    assert result.read_workspace_file("workspace.txt") == b"> create\n"

    result.release()
    assert not capture.exists()


def test_sandbox_dependency_reuses_one_native_runner_until_application_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.routes import deps

    monkeypatch.setattr(deps, "_sandbox_runner", None)
    monkeypatch.delenv("GIGACHAD_SANDBOX_EXECUTION", raising=False)
    monkeypatch.delenv("GIGACHAD_SANDBOX_FAKE", raising=False)

    first = deps.get_sandbox_runner()
    assert isinstance(first, AgentSandboxRunnerAdapter)
    assert deps.get_sandbox_runner() is first
    assert first._sandbox is not None

    monkeypatch.setattr(deps, "_sandbox_runner", None)
    monkeypatch.setenv("GIGACHAD_SANDBOX_FAKE", "true")
    assert isinstance(deps.get_sandbox_runner(), FakeSandboxRunner)
