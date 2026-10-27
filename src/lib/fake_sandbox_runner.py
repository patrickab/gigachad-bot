"""A Docker-free sandbox runner for development and tests."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile

from agent_sandbox import SandboxInvocation, SandboxManifest, SandboxRun, StreamEvent, WorkspaceArchive
from agent_sandbox.manifest import Execution, Outputs, WorkspaceRef

from lib.sandbox_service import write_temp_tar


class FakeSandboxRunner:
    """Produce native sandbox leases without requiring Docker."""

    def __init__(self) -> None:
        self.calls: list[SandboxInvocation] = []
        self.scripts: list[str] = []

    async def run_script(self, script: str, *, profile: str, interpreter: str, timeout: float) -> str:
        """Echo a minimal Plotly figure so plot flows work without Docker."""
        del profile, interpreter, timeout
        self.scripts.append(script)
        return json.dumps({"data": [{"type": "scatter", "x": [1, 2], "y": [1, 2]}], "layout": {}})

    async def run(self, invocation: SandboxInvocation) -> SandboxRun:
        self.calls.append(invocation)
        prompt = invocation.task.prompt
        workspace_bytes = self._workspace_archive(f"> {prompt}\n")
        workspace_sha256 = hashlib.sha256(workspace_bytes).hexdigest()
        summary = f"Updated the workspace for: {prompt[:200]}"
        return SandboxRun(
            status="completed",
            summary=summary,
            next_manifest=SandboxManifest(
                schema_version=1,
                manifest_id=f"{invocation.run_id}-{workspace_sha256[:16]}",
                profile=invocation.profile_name,
                runtime_fingerprint="fake",
                workspace=WorkspaceRef(snapshot_asset_id=None, sha256=workspace_sha256),
                outputs=Outputs(events_asset_id=None, artifacts=()),
                omp_sessions={},
                execution=Execution(run_id=invocation.run_id, status="completed", exit_code=0, created_at=""),
            ),
            outputs=(StreamEvent(name="stdout", text=summary),),
            workspace_changed=invocation.workspace is None or invocation.workspace.sha256 != workspace_sha256,
            workspace=WorkspaceArchive(write_temp_tar(workspace_bytes, "gigachad-sandbox-fake-"), workspace_sha256),
        )

    @staticmethod
    def _workspace_archive(workspace_text: str) -> bytes:
        content = workspace_text.encode("utf-8")
        archive_bytes = io.BytesIO()
        with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
            member = tarfile.TarInfo("workspace.txt")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        return archive_bytes.getvalue()
