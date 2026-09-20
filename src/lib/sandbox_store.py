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
from agent_sandbox.manifest import from_yaml

from lib.data_store import DataStore, DataStorePath, StorageNotFoundError
from lib.json_io import load_json, safe_write_json
from lib.storage_namespace import (
    SANDBOX,
    sandbox_active,
    sandbox_asset,
    sandbox_manifest,
    sandbox_manifest_prefix,
    sandbox_prefix,
    sandbox_run,
    sandbox_run_prefix,
    sandbox_scope,
    sandbox_slot,
)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AssetRef:
    """Reference Gigachad-owned content-addressed asset bytes."""

    sha256: str
    media_type: str
    size_bytes: int

    def to_json(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "media_type": self.media_type, "size_bytes": self.size_bytes}

    @staticmethod
    def from_json(data: dict[str, Any]) -> "AssetRef":
        return AssetRef(sha256=data["sha256"], media_type=data["media_type"], size_bytes=data["size_bytes"])


@dataclass(frozen=True)
class SlotRecord:
    version: int
    chat_id: str
    profile: str
    created_at: float
    last_activity_at: float


@dataclass(frozen=True)
class ActivePointer:
    manifest_id: str
    updated_at: float
    last_completed_run_id: str | None = None
    result: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunRecord:
    tool_call_id: str
    status: str  # "running" | "completed" | "failed" | "cancelled"
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
        *(session.state_asset_id for session in manifest.omp_sessions.values()),
        manifest.outputs.events_asset_id,
        *(artifact.asset_id for artifact in manifest.outputs.artifacts),
    )
    return {sha256 for asset_id in asset_ids if (sha256 := _asset_sha(asset_id)) is not None}


def _legacy_manifest(data: dict[str, Any]) -> SandboxManifest:
    try:
        serialized = data["extra"]["agent_sandbox_manifest"]
        legacy_workspace = AssetRef.from_json(data["workspace"])
    except (KeyError, TypeError) as exc:
        raise ValueError("legacy sandbox record lacks an embedded AIB manifest") from exc
    if not isinstance(serialized, str):
        raise ValueError("legacy sandbox record lacks an embedded AIB manifest")
    manifest = from_yaml(serialized)
    expected_asset_id = sandbox_asset(legacy_workspace.sha256)
    if manifest.workspace.snapshot_asset_id != expected_asset_id or manifest.workspace.sha256 != legacy_workspace.sha256:
        raise ValueError("legacy sandbox record disagrees with its embedded AIB workspace manifest")
    return manifest


def _decode_manifest(data: dict[str, Any]) -> tuple[SandboxManifest, bool]:
    try:
        manifest = manifest_from_dict(data)
    except ValueError:
        manifest = _legacy_manifest(data)
        legacy = True
    else:
        legacy = False
    manifest_asset_shas(manifest)
    return manifest, legacy


class SandboxStore:
    def __init__(self, data_store: DataStore, *, scope: str = "workspace") -> None:
        self._store = data_store
        self._scope = sandbox_scope(scope)

    def read_slot(self, chat_id: str) -> SlotRecord | None:
        data = load_json(DataStorePath(self._store, sandbox_slot(chat_id, self._scope)))
        return SlotRecord(**data) if isinstance(data, dict) else None

    def ensure_slot(self, chat_id: str, *, profile: str) -> SlotRecord:
        existing = self.read_slot(chat_id)
        now = time.time()
        if existing is not None:
            touched = replace(existing, last_activity_at=now)
            self._write_slot(chat_id, touched)
            return touched
        record = SlotRecord(version=SCHEMA_VERSION, chat_id=chat_id, profile=profile, created_at=now, last_activity_at=now)
        self._write_slot(chat_id, record)
        return record

    def _write_slot(self, chat_id: str, record: SlotRecord) -> None:
        safe_write_json(DataStorePath(self._store, sandbox_slot(chat_id, self._scope)), asdict(record))

    def read_active(self, chat_id: str) -> ActivePointer | None:
        data = load_json(DataStorePath(self._store, sandbox_active(chat_id, self._scope)))
        return ActivePointer(**data) if isinstance(data, dict) else None

    def write_active(self, chat_id: str, pointer: ActivePointer) -> None:
        safe_write_json(DataStorePath(self._store, sandbox_active(chat_id, self._scope)), asdict(pointer))

    def checkpoint(
        self,
        chat_id: str,
        *,
        profile: str,
        manifest: SandboxManifest,
        runs: dict[str, RunRecord],
        latest_run_id: str | None,
    ) -> None:
        latest = runs.get(latest_run_id or "")
        if latest is not None:
            self.ensure_slot(chat_id, profile=profile)
            manifest_id = self.write_manifest(chat_id, manifest)
            result = latest.result or {}
            self.write_active(
                chat_id,
                ActivePointer(
                    manifest_id=manifest_id,
                    updated_at=time.time(),
                    last_completed_run_id=latest.tool_call_id,
                    result=result,
                ),
            )
            self.write_run(chat_id, replace(latest, manifest_id=manifest_id, result=result))
        for tool_call_id, record in runs.items():
            if tool_call_id != latest_run_id:
                self.write_run(chat_id, record)

    def recover_completed_run(self, chat_id: str, tool_call_id: str) -> RunRecord | None:
        active = self.read_active(chat_id)
        if active is None or active.last_completed_run_id != tool_call_id or active.result is None:
            return None
        status = active.result.get("status")
        if status not in {"completed", "failed", "cancelled"}:
            return None
        recovered = RunRecord(
            tool_call_id=tool_call_id,
            status=status,
            started_at=active.updated_at,
            finished_at=active.updated_at,
            manifest_id=active.manifest_id,
            result=active.result,
        )
        self.write_run(chat_id, recovered)
        return recovered

    def read_run(self, chat_id: str, tool_call_id: str) -> RunRecord | None:
        data = load_json(DataStorePath(self._store, sandbox_run(chat_id, tool_call_id, self._scope)))
        return RunRecord(**data) if isinstance(data, dict) else None

    def write_run(self, chat_id: str, record: RunRecord) -> None:
        safe_write_json(DataStorePath(self._store, sandbox_run(chat_id, record.tool_call_id, self._scope)), asdict(record))

    def write_manifest(self, chat_id: str, manifest: SandboxManifest) -> str:
        manifest_asset_shas(manifest)
        manifest_id = manifest.manifest_id
        path = DataStorePath(self._store, sandbox_manifest(chat_id, manifest_id, self._scope))
        if not path.exists():
            safe_write_json(path, manifest_to_dict(manifest))
        return manifest_id

    def read_manifest(self, chat_id: str, manifest_id: str) -> SandboxManifest:
        data = load_json(DataStorePath(self._store, sandbox_manifest(chat_id, manifest_id, self._scope)))
        if not isinstance(data, dict):
            raise StorageNotFoundError(sandbox_manifest(chat_id, manifest_id, self._scope))
        manifest, legacy = _decode_manifest(data)
        if not legacy and manifest.manifest_id != manifest_id:
            raise ValueError("native sandbox manifest ID does not match its storage path")
        return manifest

    def referenced_chat_shas(self, chat_id: str) -> set[str]:
        """Collect asset hashes reachable from this chat in this store's scope."""
        shas: set[str] = set()
        for entry in self._store.list(sandbox_manifest_prefix(chat_id, self._scope), recursive=False):
            if entry.is_dir:
                continue
            data = load_json(DataStorePath(self._store, entry.key))
            if not isinstance(data, dict):
                continue
            try:
                manifest, _ = _decode_manifest(data)
            except ValueError:
                continue
            shas.update(manifest_asset_shas(manifest))
        for entry in self._store.list(sandbox_run_prefix(chat_id, self._scope), recursive=False):
            if entry.is_dir:
                continue
            data = load_json(DataStorePath(self._store, entry.key))
            if not isinstance(data, dict):
                continue
            for output in (data.get("result") or {}).get("outputs", []):
                for ref_json in output.get("mime_bundle", {}).values():
                    shas.add(AssetRef.from_json(ref_json).sha256)
        return shas

    def referenced_shas(self, *, exclude_chat_id: str | None = None, exclude_scope: str | None = None) -> set[str]:
        """Collect asset hashes outside an optionally excluded chat scope in one scan."""
        shas: set[str] = set()
        for entry in self._store.list(SANDBOX, recursive=True):
            if entry.is_dir:
                continue
            parts = entry.key.split("/")
            if len(parts) != 5 or parts[3] not in ("manifests", "runs"):
                continue
            entry_chat_id, entry_scope = parts[1], parts[2]
            if entry_chat_id == exclude_chat_id and (exclude_scope is None or entry_scope == exclude_scope):
                continue
            data = load_json(DataStorePath(self._store, entry.key))
            if not isinstance(data, dict):
                continue
            if parts[3] == "manifests":
                try:
                    manifest, _ = _decode_manifest(data)
                except ValueError:
                    continue
                shas.update(manifest_asset_shas(manifest))
            else:
                for output in (data.get("result") or {}).get("outputs", []):
                    for ref_json in output.get("mime_bundle", {}).values():
                        shas.add(ref_json["sha256"])
        return shas

    def delete_chat_state(self, chat_id: str) -> set[str]:
        """Delete this scope's chat metadata and return hashes for orphan cleanup."""
        shas = self.referenced_chat_shas(chat_id)
        with suppress(StorageNotFoundError):
            self._store.delete(sandbox_prefix(chat_id, self._scope), recursive=True)
        return shas
