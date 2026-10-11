"""Adapt native sandbox calls to Gigachad's asynchronous runner seam."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import hashlib
import io
import json
import os
import tarfile
import tempfile
from typing import Protocol

from agent_sandbox import AgentSandboxError, Sandbox, SandboxInvocation, SandboxManifest, SandboxRun, StreamEvent, WorkspaceArchive
from agent_sandbox.manifest import Execution, Outputs, WorkspaceRef


class SandboxScriptError(RuntimeError):
    """Report disposable script failures without runtime detail."""


class SandboxRunner(Protocol):
    async def run(self, invocation: SandboxInvocation) -> SandboxRun: ...

    async def run_script(self, script: str, *, profile: str, interpreter: str, timeout: float) -> str: ...


class AgentSandboxRunnerAdapter:
    """Bridge one application-lifetime native sandbox into asyncio."""

    def __init__(self, sandbox: Sandbox | None = None) -> None:
        self._sandbox = sandbox or Sandbox()

    async def run(self, invocation: SandboxInvocation) -> SandboxRun:
        return await asyncio.to_thread(self._sandbox.run, invocation)

    async def run_script(self, script: str, *, profile: str, interpreter: str, timeout: float) -> str:
        """Execute one script through the shared native sandbox."""
        try:
            return await asyncio.to_thread(
                self._sandbox.run_python_script, interpreter, script, timeout, profile_name=profile
            )
        except AgentSandboxError as exc:
            raise SandboxScriptError(str(exc)) from exc


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
            workspace=WorkspaceArchive(self._capture(workspace_bytes), workspace_sha256),
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

    @staticmethod
    def _capture(content: bytes) -> str:
        fd, capture_path = tempfile.mkstemp(prefix="gigachad-sandbox-fake-", suffix=".tar")
        try:
            with os.fdopen(fd, "wb") as capture:
                capture.write(content)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(capture_path)
            raise
        return capture_path
