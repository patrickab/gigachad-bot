"""Persist chat-scoped sandbox metadata.

Store only JSON records here. Keep referenced bytes in `AssetStore`.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import asdict, dataclass, replace
import re
import time
from typing import Any

from agent_sandbox import SandboxManifest, manifest_from_dict, manifest_to_dict

from lib.data_store import DataStore, StorageNotFoundError
from lib.json_docs import JsonDocs
from lib.storage_namespace import (
    DEFAULT_SANDBOX_SCOPE,
    SANDBOX,
    parse_sandbox_key,
    sandbox_active,
    sandbox_manifest,
    sandbox_manifest_prefix,
    sandbox_prefix,
    sandbox_run,
    sandbox_run_prefix,
    sandbox_scope,
)


@dataclass(frozen=True)
class ActivePointer:
    manifest_id: str
    updated_at: float
    last_completed_run_id: str | None = None
    result: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunRecord:
    tool_call_id: str
    status: str  # "completed" | "failed" | "cancelled"
    started_at: float
    finished_at: float | None = None
    manifest_id: str | None = None
    result: dict[str, Any] | None = None


_ASSET_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _asset_sha(asset_id: str | None) -> str | None:
    if asset_id is None:
        return None
    prefix = f"{SANDBOX}/assets/"
    if not asset_id.startswith(prefix) or not _ASSET_SHA256.fullmatch(asset_id.removeprefix(prefix)):
        raise ValueError("sandbox manifest asset IDs must use Gigachad content-addressed storage")
    return asset_id.removeprefix(prefix)


def _validate_workspace_binding(manifest: SandboxManifest) -> None:
    workspace_sha256 = _asset_sha(manifest.workspace.snapshot_asset_id)
    if workspace_sha256 != manifest.workspace.sha256:
        raise ValueError("sandbox manifest workspace must bind its Gigachad content-addressed snapshot")


def manifest_asset_shas(manifest: SandboxManifest) -> set[str]:
    """Return Gigachad-owned asset hashes referenced by a native manifest."""
    _validate_workspace_binding(manifest)
    asset_ids = (
        manifest.workspace.snapshot_asset_id,
        *(session.state_asset_id for session in manifest.pi_sessions.values()),
        manifest.outputs.events_asset_id,
        *(artifact.asset_id for artifact in manifest.outputs.artifacts),
    )
    return {sha256 for asset_id in asset_ids if (sha256 := _asset_sha(asset_id)) is not None}


def _decode_manifest(data: dict[str, Any]) -> SandboxManifest:
    manifest = manifest_from_dict(data)
    manifest_asset_shas(manifest)
    return manifest


class SandboxStore:
    def __init__(self, data_store: DataStore, *, scope: str = DEFAULT_SANDBOX_SCOPE) -> None:
        self._docs = JsonDocs(data_store)
        self._scope = sandbox_scope(scope)

    def read_active(self, chat_id: str) -> ActivePointer | None:
        data = self._docs.read(sandbox_active(chat_id, self._scope))
        return ActivePointer(**data) if isinstance(data, dict) else None

    def checkpoint(
        self,
        chat_id: str,
        *,
        manifest: SandboxManifest,
        runs: dict[str, RunRecord],
        latest_run_id: str | None,
    ) -> None:
        """Write the newest run's manifest and active pointer, and every run record in one transaction."""
        writes: list[tuple[str, dict[str, Any]]] = []
        records = [record for run_id, record in runs.items() if run_id != latest_run_id]
        latest = runs.get(latest_run_id or "")
        if latest is not None:
            manifest_asset_shas(manifest)
            manifest_id = manifest.manifest_id
            manifest_key = sandbox_manifest(chat_id, manifest_id, self._scope)
            if not self._docs.exists(manifest_key):
                writes.append((manifest_key, manifest_to_dict(manifest)))
            result = latest.result or {}
            active = ActivePointer(
                manifest_id=manifest_id, updated_at=time.time(), last_completed_run_id=latest.tool_call_id, result=result
            )
            writes.append((sandbox_active(chat_id, self._scope), asdict(active)))
            records.insert(0, replace(latest, manifest_id=manifest_id, result=result))
        writes.extend((sandbox_run(chat_id, record.tool_call_id, self._scope), asdict(record)) for record in records)
        self._docs.apply(writes)

    def read_run(self, chat_id: str, tool_call_id: str) -> RunRecord | None:
        data = self._docs.read(sandbox_run(chat_id, tool_call_id, self._scope))
        return RunRecord(**data) if isinstance(data, dict) else None

    def read_manifest(self, chat_id: str, manifest_id: str) -> SandboxManifest:
        data = self._docs.read(sandbox_manifest(chat_id, manifest_id, self._scope))
        if not isinstance(data, dict):
            raise StorageNotFoundError(sandbox_manifest(chat_id, manifest_id, self._scope))
        manifest = _decode_manifest(data)
        if manifest.manifest_id != manifest_id:
            raise ValueError("native sandbox manifest ID does not match its storage path")
        return manifest

    def _record_shas(self, kind: str, key: str) -> set[str]:
        """Asset hashes a stored manifest or run record references, none when it is unreadable."""
        data = self._docs.read(key)
        if not isinstance(data, dict):
            return set()
        if kind == "runs":
            outputs = (data.get("result") or {}).get("outputs", [])
            return {ref["sha256"] for output in outputs for ref in output.get("mime_bundle", {}).values()}
        try:
            manifest = _decode_manifest(data)
        except ValueError:
            return set()
        return manifest_asset_shas(manifest)

    def referenced_chat_shas(self, chat_id: str) -> set[str]:
        """Collect asset hashes reachable from this chat in this store's scope."""
        shas: set[str] = set()
        for kind, prefix in (("manifests", sandbox_manifest_prefix), ("runs", sandbox_run_prefix)):
            for entry in self._docs.list(prefix(chat_id, self._scope)):
                if not entry.is_dir:
                    shas |= self._record_shas(kind, entry.key)
        return shas

    def referenced_shas(self, *, exclude_chat_id: str | None = None, exclude_scope: str | None = None) -> set[str]:
        """Collect asset hashes outside an optionally excluded chat scope in one scan."""
        shas: set[str] = set()
        for entry in self._docs.list(SANDBOX, recursive=True):
            key = parse_sandbox_key(entry.key)
            if entry.is_dir or key is None:
                continue
            if key.chat_id == exclude_chat_id and (exclude_scope is None or key.scope == exclude_scope):
                continue
            shas |= self._record_shas(key.kind, entry.key)
        return shas

    def delete_chat_state(self, chat_id: str) -> set[str]:
        """Delete this scope's chat metadata and return hashes for orphan cleanup."""
        shas = self.referenced_chat_shas(chat_id)
        with suppress(StorageNotFoundError):
            self._docs.delete(sandbox_prefix(chat_id, self._scope), recursive=True)
        return shas
