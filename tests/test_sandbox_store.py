from agent_sandbox import SandboxManifest, manifest_from_dict, manifest_to_dict
from agent_sandbox.manifest import Artifact, Execution, OmpSessionState, Outputs, WorkspaceRef, to_yaml
import pytest

from lib.data_store import DataStore, StorageNotFoundError
from lib.json_docs import JsonDocs
from lib.sandbox_store import AssetRef, RunRecord, SandboxStore
from lib.storage_namespace import (
    DEFAULT_SANDBOX_SCOPE,
    sandbox_active,
    sandbox_asset,
    sandbox_manifest,
    sandbox_run,
    sandbox_slot,
)


@pytest.fixture
def sandbox_store(data_store: DataStore) -> SandboxStore:
    return SandboxStore(data_store)


def _manifest(
    manifest_id: str,
    workspace_sha256: str,
    *,
    session_sha256: str | None = None,
    events_sha256: str | None = None,
    artifacts: tuple[tuple[str, str], ...] = (),
) -> SandboxManifest:
    return SandboxManifest(
        schema_version=1,
        manifest_id=manifest_id,
        profile="gigachad",
        runtime_fingerprint="test",
        workspace=WorkspaceRef(snapshot_asset_id=sandbox_asset(workspace_sha256), sha256=workspace_sha256),
        outputs=Outputs(
            events_asset_id=sandbox_asset(events_sha256) if events_sha256 else None,
            artifacts=tuple(Artifact(asset_id=sandbox_asset(sha256), mime_type=media_type) for sha256, media_type in artifacts),
        ),
        omp_sessions={"main": OmpSessionState(session_id="main", state_asset_id=sandbox_asset(session_sha256))}
        if session_sha256
        else {},
        execution=Execution(run_id="run-1", status="completed", exit_code=0, created_at=""),
    )


def _put_manifest(data_store: DataStore, chat_id: str, manifest: SandboxManifest, scope: str = DEFAULT_SANDBOX_SCOPE) -> None:
    """Store *manifest* under its own ID, as a checkpoint would."""
    JsonDocs(data_store).write(sandbox_manifest(chat_id, manifest.manifest_id, scope), manifest_to_dict(manifest))


def test_native_manifest_mapping_round_trip() -> None:
    manifest = _manifest("native-round-trip", "a" * 64, session_sha256="b" * 64, events_sha256="c" * 64)

    assert manifest_from_dict(manifest_to_dict(manifest)) == manifest


def test_new_writes_use_native_manifest_mapping(sandbox_store: SandboxStore, data_store: DataStore) -> None:
    chat_id = "chat-native-write"
    manifest = _manifest("native-write", "a" * 64)
    record = RunRecord(tool_call_id="call-1", status="completed", started_at=1.0)

    sandbox_store.checkpoint(chat_id, profile="gigachad", manifest=manifest, runs={"call-1": record}, latest_run_id="call-1")
    persisted = JsonDocs(data_store).read(sandbox_manifest(chat_id, manifest.manifest_id))

    assert sandbox_store.read_active(chat_id).manifest_id == manifest.manifest_id
    assert persisted == manifest_to_dict(manifest)
    assert "extra" not in persisted
    assert "agent_sandbox_manifest" not in persisted


def test_native_manifest_id_must_match_its_storage_path(sandbox_store: SandboxStore, data_store: DataStore) -> None:
    chat_id = "chat-native-path"
    manifest = _manifest("native-embedded-id", "a" * 64)
    JsonDocs(data_store).write(sandbox_manifest(chat_id, "native-path-id"), manifest_to_dict(manifest))

    with pytest.raises(ValueError, match="storage path"):
        sandbox_store.read_manifest(chat_id, "native-path-id")


def test_reads_legacy_wrapper_with_embedded_aib_manifest_despite_path_id_difference(
    sandbox_store: SandboxStore, data_store: DataStore
) -> None:
    chat_id = "chat-legacy-read"
    manifest_id = "legacy-wrapper"
    manifest = _manifest("legacy-native", "a" * 64)
    legacy = {
        "version": 1,
        "workspace": AssetRef("a" * 64, "application/x-tar", 100).to_json(),
        "omp_session": None,
        "extra": {"agent_sandbox_manifest": to_yaml(manifest)},
    }
    key = sandbox_manifest(chat_id, manifest_id)
    JsonDocs(data_store).write(key, legacy)

    assert sandbox_store.read_manifest(chat_id, manifest_id) == manifest
    assert JsonDocs(data_store).read(key) == legacy


def test_invalid_legacy_wrapper_does_not_fabricate_manifest(sandbox_store: SandboxStore, data_store: DataStore) -> None:
    chat_id = "chat-invalid-legacy"
    manifest_id = "invalid-wrapper"
    JsonDocs(data_store).write(
        sandbox_manifest(chat_id, manifest_id),
        {
            "version": 1,
            "workspace": AssetRef("a" * 64, "application/x-tar", 100).to_json(),
            "omp_session": None,
            "extra": {},
        },
    )

    with pytest.raises(ValueError, match="embedded AIB manifest"):
        sandbox_store.read_manifest(chat_id, manifest_id)


def test_recover_completed_run_rebuilds_missing_run_history(sandbox_store: SandboxStore, data_store: DataStore):
    manifest = _manifest("recover-manifest", "a" * 64)
    result = {
        "status": "completed",
        "summary": "done",
        "manifest_id": manifest.manifest_id,
        "workspace_changed": True,
        "outputs": [],
        "error": None,
    }
    record = RunRecord(tool_call_id="call-1", status="completed", started_at=1.0, finished_at=2.0, result=result)
    sandbox_store.checkpoint(
        "chat-recover", profile="gigachad", manifest=manifest, runs={record.tool_call_id: record}, latest_run_id=record.tool_call_id
    )
    data_store.delete(sandbox_run("chat-recover", "call-1"))

    recovered = sandbox_store.recover_completed_run("chat-recover", "call-1")

    assert recovered is not None
    assert recovered.result == result
    assert sandbox_store.read_run("chat-recover", "call-1") == recovered


def test_checkpoint_writes_every_record_in_one_transaction(
    sandbox_store: SandboxStore, data_store: DataStore, monkeypatch: pytest.MonkeyPatch
):
    applied: list[list[str]] = []
    apply = data_store.apply
    monkeypatch.setattr(
        data_store, "apply", lambda writes, deletes: applied.append([key for key, _, _ in writes]) or apply(writes, deletes)
    )
    latest = RunRecord(tool_call_id="latest", status="completed", started_at=1.0, result={"status": "completed"})
    other = RunRecord(tool_call_id="other", status="completed", started_at=2.0)

    sandbox_store.checkpoint(
        "chat-order",
        profile="gigachad",
        manifest=_manifest("checkpoint-order", "a" * 64),
        runs={other.tool_call_id: other, latest.tool_call_id: latest},
        latest_run_id=latest.tool_call_id,
    )

    assert applied == [[
        sandbox_slot("chat-order"),
        sandbox_manifest("chat-order", "checkpoint-order"),
        sandbox_active("chat-order"),
        sandbox_run("chat-order", "latest"),
        sandbox_run("chat-order", "other"),
    ]]
    assert sandbox_store.read_active("chat-order").last_completed_run_id == "latest"
    assert sandbox_store.read_run("chat-order", "latest").manifest_id == "checkpoint-order"


def test_checkpoint_without_latest_run_writes_only_other_runs(sandbox_store: SandboxStore):
    other = RunRecord(tool_call_id="other", status="completed", started_at=2.0)

    sandbox_store.checkpoint(
        "chat-no-latest",
        profile="gigachad",
        manifest=_manifest("no-latest", "a" * 64),
        runs={other.tool_call_id: other},
        latest_run_id="missing",
    )

    assert sandbox_store.read_run("chat-no-latest", "other") == other
    assert sandbox_store.read_slot("chat-no-latest") is None
    assert sandbox_store.read_active("chat-no-latest") is None


def test_referenced_chat_shas_collects_manifest_categories_and_older_run_outputs(
    sandbox_store: SandboxStore, data_store: DataStore
):
    chat_id = "chat-assets-1"
    manifest1 = _manifest(
        "assets-one",
        "a" * 64,
        session_sha256="b" * 64,
        events_sha256="c" * 64,
        artifacts=(("d" * 64, "image/png"),),
    )
    _put_manifest(data_store, chat_id, manifest1)
    _put_manifest(data_store, chat_id, _manifest("assets-two", "e" * 64))

    output_ref = AssetRef(sha256="f" * 64, media_type="image/png", size_bytes=300)
    sandbox_store.write_run(
        chat_id,
        RunRecord(
            tool_call_id="older-run-only-output",
            status="completed",
            started_at=1000.0,
            finished_at=1010.0,
            manifest_id=manifest1.manifest_id,
            result={
                "outputs": [
                    {
                        "display_id": "img-1",
                        "title": "Image",
                        "text": None,
                        "mime_bundle": {"image/png": {**output_ref.to_json(), "asset_path": sandbox_asset(output_ref.sha256)}},
                    }
                ]
            },
        ),
    )

    shas = sandbox_store.referenced_chat_shas(chat_id)

    assert shas == {character * 64 for character in "abcdef"}


def test_referenced_shas_preserves_shared_assets_across_chats(sandbox_store: SandboxStore, data_store: DataStore):
    chat_a = "chat-a"
    chat_b = "chat-b"

    _put_manifest(data_store, chat_a, _manifest("shared-a", "a" * 64))
    _put_manifest(data_store, chat_b, _manifest("shared-b", "a" * 64))
    _put_manifest(data_store, chat_b, _manifest("other", "b" * 64))

    assert sandbox_store.referenced_shas() == {"a" * 64, "b" * 64}
    assert sandbox_store.referenced_shas(exclude_chat_id=chat_a, exclude_scope="workspace") == {"a" * 64, "b" * 64}
    assert sandbox_store.referenced_shas(exclude_chat_id=chat_b, exclude_scope="workspace") == {"a" * 64}


def test_referenced_shas_excludes_only_the_named_scope(sandbox_store: SandboxStore, data_store: DataStore):
    chat_id = "chat-scopes"
    _put_manifest(data_store, chat_id, _manifest("workspace", "a" * 64))
    _put_manifest(data_store, chat_id, _manifest("plot", "b" * 64), scope="sandbox_plot")

    assert sandbox_store.referenced_shas(exclude_chat_id=chat_id, exclude_scope="workspace") == {"b" * 64}
    assert sandbox_store.referenced_shas(exclude_chat_id=chat_id) == set()


def test_delete_chat_state_removes_all_records_and_returns_shas(sandbox_store: SandboxStore):
    chat_id = "chat-delete-1"
    manifest = _manifest("delete-manifest", "a" * 64)
    manifest_id = manifest.manifest_id
    record = RunRecord(tool_call_id="run-1", status="completed", started_at=1000.0, finished_at=1010.0)
    sandbox_store.checkpoint(chat_id, profile="default", manifest=manifest, runs={"run-1": record}, latest_run_id="run-1")

    shas = sandbox_store.delete_chat_state(chat_id)
    assert shas == {"a" * 64}

    assert sandbox_store.read_slot(chat_id) is None
    assert sandbox_store.read_active(chat_id) is None
    with pytest.raises(StorageNotFoundError):
        sandbox_store.read_manifest(chat_id, manifest_id)
    assert sandbox_store.read_run(chat_id, "run-1") is None


def test_delete_chat_state_idempotent_on_missing_chat(sandbox_store: SandboxStore):
    chat_id = "chat-never-existed"

    assert sandbox_store.delete_chat_state(chat_id) == set()
    assert sandbox_store.delete_chat_state(chat_id) == set()
