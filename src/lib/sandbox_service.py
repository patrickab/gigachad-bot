"""Coordinate staged workspace snapshots and durable rich sandbox outputs.

`invoke` holds a chat's workspace in this runtime. `checkpoint` promotes it only
after the corresponding chat history has been saved.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field, replace
import hashlib
import json
import logging
import os
import tempfile
import time
from typing import Any, Literal, cast

from agent_sandbox import AgentSandboxError, PromptImage, SandboxInvocation, SandboxManifest, SandboxRun
from agent_sandbox.manifest import Artifact, Outputs
from agent_sandbox.outputs import DisplayDataEvent, ErrorEvent, ExecuteResultEvent, OutputEvent, StreamEvent, UpdateDisplayDataEvent

from lib.agent_sandbox_adapter import SandboxRunner
from lib.asset_store import AssetStore
from lib.data_store import DataStore, StorageNotFoundError
from lib.sandbox_store import RunRecord, SandboxStore
from lib.storage_namespace import sandbox_asset, sandbox_scope

logger = logging.getLogger(__name__)
PROMPT_MAX_LEN = 12000
STALE_RUN_TIMEOUT_SECONDS = 300.0
SCRIPT_TIMEOUT_SECONDS = 60.0

MAX_OUTPUTS = 20
MAX_INLINE_TEXT_CHARS = 8000
DEFAULT_SANDBOX_SCOPE = "workspace"
WORKSPACE_AGENT_SCOPE = "workspace_agent"
SANDBOX_PLOT_SCOPE = "sandbox_plot"
KNOWN_SANDBOX_SCOPES = (DEFAULT_SANDBOX_SCOPE, WORKSPACE_AGENT_SCOPE, SANDBOX_PLOT_SCOPE)


class SandboxServiceError(RuntimeError):
    pass


class SandboxBusyError(SandboxServiceError):
    pass


@dataclass(frozen=True)
class SandboxOutputRecord:
    display_id: str | None
    title: str | None
    mime_bundle: dict[str, dict[str, Any]]
    text: str | None

    def to_json(self) -> dict[str, Any]:
        return {"display_id": self.display_id, "title": self.title, "mime_bundle": self.mime_bundle, "text": self.text}


@dataclass(frozen=True)
class SandboxToolResult:
    status: Literal["completed", "failed", "cancelled"]
    summary: str
    manifest_id: str | None
    workspace_changed: bool
    outputs: tuple[SandboxOutputRecord, ...] = ()
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "manifest_id": self.manifest_id,
            "workspace_changed": self.workspace_changed,
            "outputs": [output.to_json() for output in self.outputs],
            "error": self.error,
        }


@dataclass
class _StagedState:
    """Hold runtime-only sandbox state until checkpointing."""

    manifest: SandboxManifest | None = None
    assets: dict[str, bytes] = field(default_factory=dict)
    output_shas: set[str] = field(default_factory=set)
    runs: dict[str, RunRecord] = field(default_factory=dict)
    last_completed_run_id: str | None = None


class SandboxRuntimeState:
    """Architecture & Behavior.

    Lifetime:
        Retain chat slots across request-scoped service facades.
    Synchronization:
        Serialize one slot while allowing unrelated chats to proceed.
    """

    def __init__(self) -> None:
        self._states: dict[str, _StagedState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def lock(self, slot_key: str) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(slot_key, asyncio.Lock())

    def stage(self, slot_key: str) -> _StagedState:
        return self._states.setdefault(slot_key, _StagedState())

    def get(self, slot_key: str) -> _StagedState | None:
        return self._states.get(slot_key)

    def pop(self, slot_key: str) -> _StagedState | None:
        return self._states.pop(slot_key, None)

    def referenced_output_shas(self, *, exclude_slot: str) -> set[str]:
        return {sha256 for slot_key, state in self._states.items() if slot_key != exclude_slot for sha256 in state.output_shas}


def _normalize_prompt(prompt: str) -> str:
    normalized = prompt.strip()
    if not normalized:
        raise ValueError("prompt must not be empty")
    if len(normalized) > PROMPT_MAX_LEN:
        raise ValueError(f"prompt exceeds {PROMPT_MAX_LEN} characters")
    return normalized


def _validate_chat_id(chat_id: str) -> str:
    chat_id = chat_id.strip()
    if not chat_id:
        raise ValueError("chat_id must not be empty")
    return chat_id


def _result_from_record(record: RunRecord) -> SandboxToolResult:
    if record.result is None:
        return SandboxToolResult(
            status="failed" if record.status != "completed" else "completed",
            summary="Sandbox result unavailable.",
            manifest_id=record.manifest_id,
            workspace_changed=False,
            error="missing_result_record",
        )
    data = record.result
    outputs = tuple(
        SandboxOutputRecord(
            display_id=output.get("display_id"),
            title=output.get("title"),
            mime_bundle=output.get("mime_bundle", {}),
            text=output.get("text"),
        )
        for output in data.get("outputs", [])
    )
    return SandboxToolResult(
        status=data["status"],
        summary=data["summary"],
        manifest_id=data.get("manifest_id"),
        workspace_changed=data.get("workspace_changed", False),
        outputs=outputs,
        error=data.get("error"),
    )


class SandboxService:
    """Architecture & Behavior.

    Invocation:
        Materialize Gigachad workspace bytes, then call the native sandbox seam.
    Persistence:
        Persist projected output assets immediately and stage workspace bytes until checkpointing.
    Safety:
        Contain native failures, remove restore files, and release every native run.
    """

    def __init__(
        self,
        *,
        data_store: DataStore,
        asset_store: AssetStore,
        runner: SandboxRunner,
        runtime_state: SandboxRuntimeState,
        model: str,
        profile: str = "gigachad",
        slot_prefix: str = "",
    ) -> None:
        self._data_store = data_store
        self._assets = asset_store
        self._runner = runner
        self._runtime = runtime_state
        self._model = model
        self._profile = profile
        self._slot_prefix = slot_prefix

    def _slot_key(self, chat_id: str, scope: str) -> str:
        scoped_chat = f"{scope}:{chat_id}"
        return f"{self._slot_prefix}:{scoped_chat}" if self._slot_prefix else scoped_chat

    @property
    def model(self) -> str:
        """Name the configured sandbox model."""
        return self._model

    def _store_for(self, scope: str) -> SandboxStore:
        return SandboxStore(self._data_store, scope=scope)

    async def run_script(self, script: str, *, interpreter: str = "venv", timeout: float = SCRIPT_TIMEOUT_SECONDS) -> str:
        """Run one throwaway script with no workspace or secrets."""
        return await self._runner.run_script(script, profile=self._profile, interpreter=interpreter, timeout=timeout)

    async def invoke(
        self,
        *,
        chat_id: str,
        tool_call_id: str,
        prompt: str,
        prompt_images: tuple[PromptImage, ...] = (),
        append_system: str | None = None,
        scope: str = DEFAULT_SANDBOX_SCOPE,
        thinking: Literal["low", "medium"] = "medium",
        lean: bool = False,
    ) -> SandboxToolResult:
        chat_id = _validate_chat_id(chat_id)
        prompt = _normalize_prompt(prompt)
        scope = sandbox_scope(scope)
        store = self._store_for(scope)
        slot_key = self._slot_key(chat_id, scope)

        lock = await self._runtime.lock(slot_key)
        async with lock:
            staged = self._runtime.stage(slot_key)
            staged_record = staged.runs.get(tool_call_id)
            if staged_record is not None:
                return _result_from_record(staged_record)

            existing = store.read_run(chat_id, tool_call_id)
            if existing is not None and existing.status == "running":
                recovered = store.recover_completed_run(chat_id, tool_call_id)
                if recovered is not None:
                    return _result_from_record(recovered)
            if existing is not None and existing.status != "running":
                return _result_from_record(existing)
            if existing is not None and time.time() - existing.started_at < STALE_RUN_TIMEOUT_SECONDS:
                raise SandboxBusyError(f"sandbox run {tool_call_id} is already in progress")

            prior_manifest = staged.manifest
            started_at = time.time()
            restore_path: str | None = None
            run: SandboxRun | None = None
            runner_task: asyncio.Task[SandboxRun] | None = None
            try:
                if prior_manifest is None:
                    active = store.read_active(chat_id)
                    if active is not None:
                        try:
                            prior_manifest = store.read_manifest(chat_id, active.manifest_id)
                        except (StorageNotFoundError, ValueError) as exc:
                            raise SandboxServiceError("active sandbox manifest is unavailable") from exc
                restore_path = self._materialize_workspace_archive(prior_manifest, staged.assets)
                invocation = SandboxInvocation(
                    slot_key=slot_key,
                    profile_name=self._profile,
                    prompt=prompt,
                    run_id=tool_call_id,
                    active_manifest=prior_manifest,
                    workspace_archive_path=restore_path,
                    prompt_images=prompt_images,
                    model=self._model,
                    thinking=thinking,
                    append_system=append_system,
                    lean=lean,
                )
                runner_task = asyncio.create_task(self._runner.run(invocation))
                run = await asyncio.shield(runner_task)
                result = self._accept_run(run, staged, tool_call_id=tool_call_id)
            except asyncio.CancelledError:
                if runner_task is not None and run is None:
                    runner_task.add_done_callback(lambda task, path=restore_path: self._release_and_unlink_cancelled_run(task, path))
                    restore_path = None
                raise
            except SandboxServiceError:
                logger.exception("sandbox run %s produced an incoherent manifest for slot %s", tool_call_id, slot_key)
                result = SandboxToolResult(
                    "failed",
                    "The workspace agent could not complete this run.",
                    prior_manifest.manifest_id if prior_manifest else None,
                    False,
                    error="invalid_manifest",
                )
            except Exception:  # noqa: BLE001 - contain runner failures within this tool call
                logger.exception("sandbox runner failed for slot %s run %s", slot_key, tool_call_id)
                result = SandboxToolResult(
                    "failed",
                    "The workspace agent could not complete this run.",
                    prior_manifest.manifest_id if prior_manifest else None,
                    False,
                    error="runner_error",
                )
            finally:
                if restore_path is not None:
                    with suppress(FileNotFoundError):
                        os.unlink(restore_path)
                if run is not None:
                    try:
                        run.release()
                    except Exception:  # noqa: BLE001 - cleanup must not invalidate a staged run
                        logger.exception("sandbox run %s release failed for slot %s", tool_call_id, slot_key)

            staged.runs[tool_call_id] = RunRecord(
                tool_call_id=tool_call_id,
                status=result.status,
                started_at=started_at,
                finished_at=time.time(),
                manifest_id=result.manifest_id,
                result=result.to_json(),
            )
            return result

    @staticmethod
    def _release_and_unlink_cancelled_run(task: asyncio.Task[SandboxRun], restore_path: str | None) -> None:
        try:
            run = task.result()
        except BaseException:
            logger.exception("cancelled sandbox runner failed")
        else:
            try:
                run.release()
            except Exception:  # noqa: BLE001 - cleanup must not surface after cancellation
                logger.exception("cancelled sandbox run release failed")
        if restore_path is not None:
            with suppress(FileNotFoundError):
                os.unlink(restore_path)

    def _materialize_workspace_archive(self, prior_manifest: SandboxManifest | None, staged_assets: dict[str, bytes]) -> str | None:
        if prior_manifest is None:
            return None
        sha256 = prior_manifest.workspace.sha256
        content = staged_assets.get(sha256)
        if content is None:
            try:
                content = self._assets.read(sandbox_asset(sha256)).content
            except StorageNotFoundError as exc:
                raise SandboxServiceError(f"workspace snapshot {sha256} is unavailable") from exc
        if content is None:
            raise SandboxServiceError(f"workspace snapshot {sha256} is unavailable")
        fd, path = tempfile.mkstemp(prefix="gigachad-sandbox-restore-", suffix=".tar")
        try:
            with os.fdopen(fd, "wb") as archive:
                archive.write(content)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(path)
            raise
        return path

    def _accept_run(self, run: SandboxRun, staged: _StagedState, *, tool_call_id: str) -> SandboxToolResult:
        """Persist outputs now and stage workspace bytes until checkpointing."""
        try:
            with open(run.capture_path, "rb") as capture:
                workspace_bytes = capture.read()
        except OSError as exc:
            raise SandboxServiceError("sandbox workspace capture is unavailable") from exc
        workspace_sha256 = hashlib.sha256(workspace_bytes).hexdigest()
        if run.next_manifest.workspace.sha256 != workspace_sha256:
            raise SandboxServiceError("sandbox capture does not match its native workspace manifest")

        outputs, output_assets = self._project_outputs(cast("tuple[OutputEvent, ...]", run.outputs))
        plot_bytes = self._plot_bytes(run)
        if plot_bytes is not None:
            plot_output, plot_assets = self._make_output_record(
                display_id=None,
                title="Interactive plot",
                bundle={"application/vnd.plotly.v1+json": plot_bytes},
                text=None,
            )
            outputs = (*outputs, plot_output)
            output_assets.update(plot_assets)

        manifest = run.next_manifest.with_workspace_asset_id(sandbox_asset(workspace_sha256))
        artifacts = tuple(
            Artifact(asset_id=ref["asset_path"], mime_type=media_type)
            for output in outputs
            for media_type, ref in output.mime_bundle.items()
        )
        if artifacts:
            manifest = replace(
                manifest,
                outputs=Outputs(
                    events_asset_id=manifest.outputs.events_asset_id,
                    artifacts=(*manifest.outputs.artifacts, *artifacts),
                ),
            )
        media_types = {ref["sha256"]: media_type for output in outputs for media_type, ref in output.mime_bundle.items()}
        self._persist_assets(output_assets, media_types)

        staged.assets = {workspace_sha256: workspace_bytes}
        staged.output_shas.update(output_assets)
        staged.manifest = manifest
        staged.last_completed_run_id = tool_call_id
        return SandboxToolResult(
            status=cast('Literal["completed", "failed", "cancelled"]', run.status),
            summary=run.summary,
            manifest_id=manifest.manifest_id,
            workspace_changed=run.workspace_changed,
            outputs=outputs[:MAX_OUTPUTS],
        )

    @classmethod
    def _project_outputs(cls, events: tuple[OutputEvent, ...]) -> tuple[tuple[SandboxOutputRecord, ...], dict[str, bytes]]:
        outputs: list[SandboxOutputRecord] = []
        assets: dict[str, bytes] = {}
        for index, event in enumerate(events):
            if isinstance(event, StreamEvent):
                bundle = {"text/plain": event.text}
                display_id = None
                title = event.name
            elif isinstance(event, DisplayDataEvent):
                bundle = event.bundle.data
                display_id = event.display_id
                title = "Display output"
            elif isinstance(event, ExecuteResultEvent):
                bundle = event.bundle.data
                display_id = None
                title = "Execution result"
            elif isinstance(event, UpdateDisplayDataEvent):
                bundle = event.bundle.data
                display_id = event.display_id
                title = "Display update"
            elif isinstance(event, ErrorEvent):
                bundle = {"text/plain": f"{event.ename}: {event.evalue}"}
                display_id = None
                title = "Execution error"
            else:
                raise SandboxServiceError(f"unsupported sandbox output event: {type(event).__name__}")
            output, output_assets = cls._make_output_record(
                display_id=display_id or f"output-{index}",
                title=title,
                bundle={media_type: value.encode("utf-8") for media_type, value in bundle.items()},
                text=bundle.get("text/plain"),
            )
            outputs.append(output)
            assets.update(output_assets)
        return tuple(outputs), assets

    @staticmethod
    def _plot_bytes(run: SandboxRun) -> bytes | None:
        try:
            content = run.read_workspace_file("plot.json")
            return content if isinstance(json.loads(content), dict) else None
        except (AgentSandboxError, UnicodeDecodeError, ValueError):
            return None

    @staticmethod
    def _make_output_record(
        *, display_id: str | None, title: str | None, bundle: dict[str, bytes], text: str | None
    ) -> tuple[SandboxOutputRecord, dict[str, bytes]]:
        assets: dict[str, bytes] = {}
        mime_bundle: dict[str, dict[str, Any]] = {}
        for media_type, content in bundle.items():
            sha256 = hashlib.sha256(content).hexdigest()
            assets[sha256] = content
            mime_bundle[media_type] = {
                "sha256": sha256,
                "media_type": media_type,
                "size_bytes": len(content),
                "asset_path": sandbox_asset(sha256),
            }
        if text is not None and len(text) > MAX_INLINE_TEXT_CHARS:
            text = text[:MAX_INLINE_TEXT_CHARS] + "\u2026 [truncated]"
        return SandboxOutputRecord(display_id=display_id, title=title, mime_bundle=mime_bundle, text=text), assets

    async def checkpoint(self, *, chat_id: str, scope: str | None = None) -> None:
        """Promote the latest runtime-only workspace after its chat is saved."""
        chat_id = _validate_chat_id(chat_id)
        scopes = (sandbox_scope(scope),) if scope is not None else KNOWN_SANDBOX_SCOPES
        for state_scope in scopes:
            store = self._store_for(state_scope)
            slot_key = self._slot_key(chat_id, state_scope)
            lock = await self._runtime.lock(slot_key)
            async with lock:
                staged = self._runtime.get(slot_key)
                if staged is None:
                    continue
                if staged.manifest is not None:
                    self._persist_assets(staged.assets, {staged.manifest.workspace.sha256: "application/x-tar"})
                    store.checkpoint(
                        chat_id,
                        profile=self._profile,
                        manifest=staged.manifest,
                        runs=staged.runs,
                        latest_run_id=staged.last_completed_run_id,
                    )
                self._runtime.pop(slot_key)

    def _persist_assets(self, assets: dict[str, bytes], media_types: dict[str, str]) -> None:
        for sha256, content in assets.items():
            if hashlib.sha256(content).hexdigest() != sha256:
                raise SandboxServiceError(f"sandbox asset content does not match its declared sha256: {sha256}")
            key = sandbox_asset(sha256)
            if not self._assets.list(key):
                self._assets.write("sandbox", key, content, mime=media_types[sha256])

    def output_texts(self, result: SandboxToolResult, *, media_type: str = "text/plain") -> tuple[str, ...]:
        """Return selected sandbox output bytes without widening the UI payload."""
        texts: list[str] = []
        for output in result.outputs:
            text_ref = output.mime_bundle.get(media_type)
            if text_ref is not None:
                with suppress(StorageNotFoundError, UnicodeDecodeError):
                    texts.append(self._assets.read(text_ref["asset_path"]).content.decode("utf-8"))
                    continue
            if media_type == "text/plain" and output.text:
                texts.append(output.text)
        return tuple(texts)

    async def delete_chat_state(self, *, chat_id: str, scope: str | None = None) -> None:
        chat_id = _validate_chat_id(chat_id)
        scopes = (sandbox_scope(scope),) if scope is not None else KNOWN_SANDBOX_SCOPES
        for state_scope in scopes:
            store = self._store_for(state_scope)
            slot_key = self._slot_key(chat_id, state_scope)
            lock = await self._runtime.lock(slot_key)
            async with lock:
                staged = self._runtime.pop(slot_key)
                shas = store.delete_chat_state(chat_id)
                if staged is not None:
                    shas.update(staged.output_shas)
                if not shas:
                    continue
                still_referenced = store.referenced_shas(exclude_chat_id=chat_id, exclude_scope=state_scope)
                still_referenced.update(self._runtime.referenced_output_shas(exclude_slot=slot_key))
                for sha256 in shas:
                    if sha256 not in still_referenced:
                        with suppress(StorageNotFoundError):
                            self._assets.delete(sandbox_asset(sha256))
