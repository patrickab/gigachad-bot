"""Adapt native sandbox calls to Gigachad's asynchronous runner seam."""

from __future__ import annotations

import asyncio
from typing import Protocol

from agent_sandbox import AgentSandboxError, Sandbox, SandboxInvocation, SandboxRun


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
