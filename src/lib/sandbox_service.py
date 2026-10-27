"""Coordinate staged workspace snapshots and durable rich sandbox outputs.

`invoke` holds a chat's workspace in this runtime. `checkpoint` promotes it only
after the corresponding chat history has been saved.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
import hashlib
import json
import logging
import os
import tempfile
import time
from typing import TYPE_CHECKING, Any, Literal

from agent_sandbox import (
    AgentSandboxError,
    ErrorEvent,
    OmpTask,
    OutputEvent,
    PromptImage,
    SandboxInvocation,
    SandboxManifest,
    SandboxRun,
    StreamEvent,
    WorkspaceArchive,
)
from agent_sandbox.manifest import Artifact, Outputs
from pydantic import BaseModel, ConfigDict

from lib.agent_sandbox_adapter import SandboxRunner
from lib.asset_store import AssetStore
from lib.data_store import DataStore, StorageNotFoundError
from lib.sandbox_store import RunRecord, SandboxStore
from lib.storage_namespace import DEFAULT_SANDBOX_SCOPE, sandbox_asset, sandbox_scope

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)
PROMPT_MAX_LEN = 12000
# A "running" record older than this is treated as abandoned and may be re-run.
STALE_RUN_TIMEOUT_SECONDS = 300.0
SCRIPT_TIMEOUT_SECONDS = 60.0

MAX_OUTPUTS = 20
MAX_INLINE_TEXT_CHARS = 8000
WORKSPACE_AGENT_SCOPE = "workspace_agent"
SANDBOX_PLOT_SCOPE = "sandbox_plot"
KNOWN_SANDBOX_SCOPES = (DEFAULT_SANDBOX_SCOPE, WORKSPACE_AGENT_SCOPE, SANDBOX_PLOT_SCOPE)


class SandboxServiceError(RuntimeError):
    pass


class SandboxBusyError(SandboxServiceError):
    pass


class SandboxOutputRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    display_id: str | None = None
    title: str | None = None
    mime_bundle: dict[str, dict[str, Any]] = {}
    text: str | None = None


class SandboxToolResult(BaseModel):
    """One workspace-agent run; its JSON dump is the stored run record result."""

    model_config = ConfigDict(frozen=True)

    status: Literal["completed", "failed", "cancelled"]
    summary: str
    manifest_id: str | None = None
    workspace_changed: bool = False
    outputs: tuple[SandboxOutputRecord, ...] = ()
    error: str | None = None

    @classmethod
    def from_record(cls, record: RunRecord) -> SandboxToolResult:
        """Rebuild a recorded result, or a failure placeholder when the record lost it."""
        if record.result is None:
            return cls(
                status="failed" if record.status != "completed" else "completed",
                summary="Sandbox result unavailable.",
                manifest_id=record.manifest_id,
                workspace_changed=False,
                error="missing_result_record",
            )
        return cls.model_validate(record.result)

    def chat_payload(self) -> dict[str, Any]:
        """The `sandbox` field of the browser's tool result."""
        return {
            "status": self.status,
            "manifest_id": f"sha256:{self.manifest_id}" if self.manifest_id else None,
            "workspace_changed": self.workspace_changed,
            "outputs": [output.model_dump(mode="json") for output in self.outputs],
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
        # Serializes checkpoints against asset garbage collection: both hop off the event loop between store reads and writes.
        self.assets_lock = asyncio.Lock()

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


def write_temp_tar(content: bytes, prefix: str) -> str:
    """Write *content* to a fresh temporary `.tar` file and return its path, removing it again on failure."""
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".tar")
    try:
        with os.fdopen(fd, "wb") as archive:
            archive.write(content)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(path)
        raise
    return path


def _discard_restore(restore: WorkspaceArchive | None) -> None:
    if restore is not None:
        with suppress(FileNotFoundError):
            os.unlink(restore.path)


def _failed_result(prior_manifest: SandboxManifest | None, error: str) -> SandboxToolResult:
    """The result of a run that could not complete, still pointing at the manifest it started from."""
    return SandboxToolResult(
        status="failed",
        summary="The workspace agent could not complete this run.",
        manifest_id=prior_manifest.manifest_id if prior_manifest else None,
        workspace_changed=False,
        error=error,
    )


def _scopes(scope: str | None) -> tuple[str, ...]:
    """The one validated scope, or every known scope when none is named."""
    return (sandbox_scope(scope),) if scope is not None else KNOWN_SANDBOX_SCOPES


@dataclass(frozen=True)
class _Slot:
    """One chat's sandbox state in one scope: its runtime key and its durable store."""

    scope: str
    key: str
    store: SandboxStore


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

    @asynccontextmanager
    async def _slot(self, chat_id: str, scope: str) -> AsyncIterator[_Slot]:
        """Hold the slot's lock for the duration of the block."""
        slot = _Slot(scope, self._slot_key(chat_id, scope), SandboxStore(self._data_store, scope=scope))
        async with await self._runtime.lock(slot.key):
            yield slot

    async def run_script(self, script: str, *, interpreter: str = "venv", timeout: float = SCRIPT_TIMEOUT_SECONDS) -> str:
        """Run one throwaway script with no workspace or secrets."""
        return await self._runner.run_script(script, profile=self._profile, interpreter=interpreter, timeout=timeout)

    @staticmethod
    def _replay_or_busy(store: SandboxStore, staged: _StagedState, chat_id: str, tool_call_id: str) -> SandboxToolResult | None:
        """The recorded result of a finished call, None when the call must run. Raises while it is still running."""
        staged_record = staged.runs.get(tool_call_id)
        if staged_record is not None:
            return SandboxToolResult.from_record(staged_record)
        existing = store.read_run(chat_id, tool_call_id)
        if existing is None:
            return None
        if existing.status != "running":
            return SandboxToolResult.from_record(existing)
        # A running record may belong to a finished run whose record write was interrupted.
        if (recovered := store.recover_completed_run(chat_id, tool_call_id)) is not None:
            return SandboxToolResult.from_record(recovered)
        # Fresh running record means a live run. Stale ones fall through and re-run.
        if time.time() - existing.started_at < STALE_RUN_TIMEOUT_SECONDS:
            raise SandboxBusyError(f"sandbox run {tool_call_id} is already in progress")
        return None

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
        """Run one workspace-agent task, idempotent per `tool_call_id`.

        Raises `SandboxBusyError` while the same call is still running elsewhere.
        """
        chat_id = _validate_chat_id(chat_id)
        prompt = _normalize_prompt(prompt)
        scope = sandbox_scope(scope)

        async with self._slot(chat_id, scope) as slot:
            store, slot_key = slot.store, slot.key
            staged = self._runtime.stage(slot_key)
            if (replayed := self._replay_or_busy(store, staged, chat_id, tool_call_id)) is not None:
                return replayed

            prior_manifest = staged.manifest
            started_at = time.time()
            restore: WorkspaceArchive | None = None
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
                restore = self._materialize_workspace_archive(prior_manifest, staged.assets)
                invocation = SandboxInvocation(
                    slot_key=slot_key,
                    profile_name=self._profile,
                    run_id=tool_call_id,
                    task=OmpTask(
                        prompt,
                        prompt_images=prompt_images,
                        model=self._model,
                        thinking=thinking,
                        append_system=append_system,
                        lean=lean,
                    ),
                    active_manifest=prior_manifest,
                    workspace=restore,
                )
                # Shield so a cancelled caller cannot orphan the native run before it is released.
                runner_task = asyncio.create_task(self._runner.run(invocation))
                run = await asyncio.shield(runner_task)
                result = self._accept_run(run, staged, tool_call_id=tool_call_id)
            except asyncio.CancelledError:
                if runner_task is not None and run is None:
                    runner_task.add_done_callback(lambda task, archive=restore: self._release_cancelled_run(task, archive))
                    restore = None
                raise
            except SandboxServiceError:
                logger.exception("sandbox run %s produced an incoherent manifest for slot %s", tool_call_id, slot_key)
                result = _failed_result(prior_manifest, "invalid_manifest")
            except Exception:  # noqa: BLE001 - contain runner failures within this tool call
                logger.exception("sandbox runner failed for slot %s run %s", slot_key, tool_call_id)
                result = _failed_result(prior_manifest, "runner_error")
            finally:
                _discard_restore(restore)
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
                result=result.model_dump(mode="json"),
            )
            return result

    @staticmethod
    def _release_cancelled_run(task: asyncio.Task[SandboxRun], restore: WorkspaceArchive | None) -> None:
        try:
            run = task.result()
        except BaseException:
            logger.exception("cancelled sandbox runner failed")
        else:
            try:
                run.release()
            except Exception:  # noqa: BLE001 - cleanup must not surface after cancellation
                logger.exception("cancelled sandbox run release failed")
        _discard_restore(restore)

    def _materialize_workspace_archive(
        self, prior_manifest: SandboxManifest | None, staged_assets: dict[str, bytes]
    ) -> WorkspaceArchive | None:
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
        return WorkspaceArchive(write_temp_tar(content, "gigachad-sandbox-restore-"), sha256)

    def _accept_run(self, run: SandboxRun, staged: _StagedState, *, tool_call_id: str) -> SandboxToolResult:
        """Persist outputs now and stage workspace bytes until checkpointing."""
        try:
            with open(run.workspace.path, "rb") as capture:
                workspace_bytes = capture.read()
        except OSError as exc:
            raise SandboxServiceError("sandbox workspace capture is unavailable") from exc
        workspace_sha256 = hashlib.sha256(workspace_bytes).hexdigest()
        if run.next_manifest.workspace.sha256 != workspace_sha256:
            raise SandboxServiceError("sandbox capture does not match its native workspace manifest")

        outputs, output_assets = self._project_outputs(run.outputs)
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

        # Only the newest workspace snapshot is kept in memory.
        staged.assets = {workspace_sha256: workspace_bytes}
        staged.output_shas.update(output_assets)
        staged.manifest = manifest
        staged.last_completed_run_id = tool_call_id
        return SandboxToolResult(
            status=run.status,
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
            match event:
                case StreamEvent():
                    title, text = event.name, event.text
                case ErrorEvent():
                    title, text = "Execution error", f"{event.ename}: {event.evalue}"
                case _:
                    raise SandboxServiceError(f"unsupported sandbox output event: {type(event).__name__}")
            output, output_assets = cls._make_output_record(
                display_id=f"output-{index}",
                title=title,
                bundle={"text/plain": text.encode("utf-8")},
                text=text,
            )
            outputs.append(output)
            assets.update(output_assets)
        return tuple(outputs), assets

    @staticmethod
    def _plot_bytes(run: SandboxRun) -> bytes | None:
        """Return `plot.json` bytes from the workspace if it holds a JSON object."""
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
        for scope_name in _scopes(scope):
            async with self._slot(chat_id, scope_name) as slot, self._runtime.assets_lock:
                staged = self._runtime.get(slot.key)
                if staged is None:
                    continue
                if (manifest := staged.manifest) is not None:
                    # Asset writes and the checkpoint block on Postgres or the mount, so they run off the event loop.
                    await asyncio.to_thread(self._persist_checkpoint, chat_id, slot.store, staged, manifest)
                self._runtime.pop(slot.key)

    def _persist_checkpoint(self, chat_id: str, store: SandboxStore, staged: _StagedState, manifest: SandboxManifest) -> None:
        """Write the staged workspace snapshot, then the checkpoint naming *manifest*."""
        self._persist_assets(staged.assets, {manifest.workspace.sha256: "application/x-tar"})
        store.checkpoint(
            chat_id,
            profile=self._profile,
            manifest=manifest,
            runs=staged.runs,
            latest_run_id=staged.last_completed_run_id,
        )

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
        for scope_name in _scopes(scope):
            async with self._slot(chat_id, scope_name) as slot, self._runtime.assets_lock:
                staged = self._runtime.pop(slot.key)
                staged_shas = staged.output_shas if staged is not None else set()
                # Store IO blocks on Postgres, so it runs off the event loop.
                shas, still_referenced = await asyncio.to_thread(self._delete_durable_state, chat_id, slot, staged_shas)
                if not shas:
                    continue
                # Keep assets still referenced by other chats, scopes, or staged runtime state. Runs staged while the
                # assets are deleted are not seen here, so the deletes stay on the event loop to stay atomic with them.
                still_referenced.update(self._runtime.referenced_output_shas(exclude_slot=slot.key))
                for sha256 in shas:
                    if sha256 not in still_referenced:
                        with suppress(StorageNotFoundError):
                            self._assets.delete(sandbox_asset(sha256))

    @staticmethod
    def _delete_durable_state(chat_id: str, slot: _Slot, staged_shas: set[str]) -> tuple[set[str], set[str]]:
        """Delete the chat's durable state, returning its asset shas and those other chats or scopes still reference."""
        shas = slot.store.delete_chat_state(chat_id) | staged_shas
        return shas, slot.store.referenced_shas(exclude_chat_id=chat_id, exclude_scope=slot.scope) if shas else set()
