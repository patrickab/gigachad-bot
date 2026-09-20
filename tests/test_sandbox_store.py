import os
from uuid import uuid4

from agent_sandbox import SandboxManifest, manifest_from_dict, manifest_to_dict
from agent_sandbox.manifest import Artifact, Execution, OmpSessionState, Outputs, WorkspaceRef, to_yaml
from psycopg_pool import ConnectionPool
import pytest

from lib.data_store import DataStore, DataStorePath, StorageNotFoundError
from lib.db_schema import upgrade
from lib.json_io import load_json, safe_write_json
from lib.postgres_data_store import PostgresDataStore
from lib.sandbox_store import ActivePointer, AssetRef, RunRecord, SandboxStore
from lib.storage_namespace import sandbox_asset, sandbox_manifest, sandbox_run


@pytest.fixture(scope="session")
def postgres_pool():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for Postgres DataStore conformance tests")
    upgrade(url)
    pool = ConnectionPool(url, min_size=1, max_size=4)
    yield pool
    pool.close()


@pytest.fixture
def store(postgres_pool) -> DataStore:
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute("TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE")
        user_id = connection.execute(
            "INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", (f"test-{uuid4()}@example.test",)
        ).fetchone()[0]
    return PostgresDataStore(postgres_pool, user_id)


@pytest.fixture
def sandbox_store(store: DataStore) -> SandboxStore:
    return SandboxStore(store)


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


def test_native_manifest_mapping_round_trip() -> None:
    manifest = _manifest("native-round-trip", "a" * 64, session_sha256="b" * 64, events_sha256="c" * 64)

    assert manifest_from_dict(manifest_to_dict(manifest)) == manifest


def test_new_writes_use_native_manifest_mapping(sandbox_store: SandboxStore) -> None:
    chat_id = "chat-native-write"
    manifest = _manifest("native-write", "a" * 64)

    manifest_id = sandbox_store.write_manifest(chat_id, manifest)
    persisted = load_json(DataStorePath(sandbox_store._store, sandbox_manifest(chat_id, manifest_id)))

    assert manifest_id == manifest.manifest_id
    assert persisted == manifest_to_dict(manifest)
    assert "extra" not in persisted
    assert "agent_sandbox_manifest" not in persisted


def test_native_manifest_id_must_match_its_storage_path(sandbox_store: SandboxStore) -> None:
    chat_id = "chat-native-path"
    manifest = _manifest("native-embedded-id", "a" * 64)
    path = DataStorePath(sandbox_store._store, sandbox_manifest(chat_id, "native-path-id"))
    safe_write_json(path, manifest_to_dict(manifest))

    with pytest.raises(ValueError, match="storage path"):
        sandbox_store.read_manifest(chat_id, "native-path-id")


def test_reads_legacy_wrapper_with_embedded_aib_manifest_despite_path_id_difference(sandbox_store: SandboxStore) -> None:
    chat_id = "chat-legacy-read"
    manifest_id = "legacy-wrapper"
    manifest = _manifest("legacy-native", "a" * 64)
    legacy = {
        "version": 1,
        "workspace": AssetRef("a" * 64, "application/x-tar", 100).to_json(),
        "omp_session": None,
        "extra": {"agent_sandbox_manifest": to_yaml(manifest)},
    }
    path = DataStorePath(sandbox_store._store, sandbox_manifest(chat_id, manifest_id))
    safe_write_json(path, legacy)

    assert sandbox_store.read_manifest(chat_id, manifest_id) == manifest
    assert load_json(path) == legacy


def test_invalid_legacy_wrapper_does_not_fabricate_manifest(sandbox_store: SandboxStore) -> None:
    chat_id = "chat-invalid-legacy"
    manifest_id = "invalid-wrapper"
    path = DataStorePath(sandbox_store._store, sandbox_manifest(chat_id, manifest_id))
    safe_write_json(
        path,
        {
            "version": 1,
            "workspace": AssetRef("a" * 64, "application/x-tar", 100).to_json(),
            "omp_session": None,
            "extra": {},
        },
    )

    with pytest.raises(ValueError, match="embedded AIB manifest"):
        sandbox_store.read_manifest(chat_id, manifest_id)


def test_recover_completed_run_rebuilds_missing_run_history(sandbox_store: SandboxStore):
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
    sandbox_store._store.delete(sandbox_run("chat-recover", "call-1"))

    recovered = sandbox_store.recover_completed_run("chat-recover", "call-1")

    assert recovered is not None
    assert recovered.result == result
    assert sandbox_store.read_run("chat-recover", "call-1") == recovered


def test_checkpoint_writes_recoverable_state_before_other_runs(sandbox_store: SandboxStore, monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []
    latest = RunRecord(tool_call_id="latest", status="completed", started_at=1.0, result={"status": "completed"})
    other = RunRecord(tool_call_id="other", status="completed", started_at=2.0)

    monkeypatch.setattr(sandbox_store, "ensure_slot", lambda chat_id, *, profile: calls.append("slot"))
    monkeypatch.setattr(sandbox_store, "write_manifest", lambda chat_id, manifest: calls.append("manifest") or manifest.manifest_id)
    monkeypatch.setattr(sandbox_store, "write_active", lambda chat_id, pointer: calls.append("active"))
    monkeypatch.setattr(sandbox_store, "write_run", lambda chat_id, record: calls.append(record.tool_call_id))

    sandbox_store.checkpoint(
        "chat-order",
        profile="gigachad",
        manifest=_manifest("checkpoint-order", "a" * 64),
        runs={other.tool_call_id: other, latest.tool_call_id: latest},
        latest_run_id=latest.tool_call_id,
    )

    assert calls == ["slot", "manifest", "active", "latest", "other"]


def test_checkpoint_without_latest_run_writes_only_other_runs(sandbox_store: SandboxStore, monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []
    other = RunRecord(tool_call_id="other", status="completed", started_at=2.0)

    monkeypatch.setattr(sandbox_store, "ensure_slot", lambda chat_id, *, profile: calls.append("slot"))
    monkeypatch.setattr(sandbox_store, "write_manifest", lambda chat_id, manifest: calls.append("manifest") or manifest.manifest_id)
    monkeypatch.setattr(sandbox_store, "write_active", lambda chat_id, pointer: calls.append("active"))
    monkeypatch.setattr(sandbox_store, "write_run", lambda chat_id, record: calls.append(record.tool_call_id))

    sandbox_store.checkpoint(
        "chat-no-latest",
        profile="gigachad",
        manifest=_manifest("no-latest", "a" * 64),
        runs={other.tool_call_id: other},
        latest_run_id="missing",
    )

    assert calls == ["other"]


def test_referenced_chat_shas_collects_manifest_categories_and_older_run_outputs(sandbox_store: SandboxStore):
    chat_id = "chat-assets-1"
    manifest1 = _manifest(
        "assets-one",
        "a" * 64,
        session_sha256="b" * 64,
        events_sha256="c" * 64,
        artifacts=(("d" * 64, "image/png"),),
    )
    sandbox_store.write_manifest(chat_id, manifest1)
    manifest2 = _manifest("assets-two", "e" * 64)
    sandbox_store.write_manifest(chat_id, manifest2)

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


def test_referenced_shas_preserves_shared_assets_across_chats(sandbox_store: SandboxStore):
    chat_a = "chat-a"
    chat_b = "chat-b"

    sandbox_store.write_manifest(chat_a, _manifest("shared-a", "a" * 64))
    sandbox_store.write_manifest(chat_b, _manifest("shared-b", "a" * 64))
    sandbox_store.write_manifest(chat_b, _manifest("other", "b" * 64))

    assert sandbox_store.referenced_shas() == {"a" * 64, "b" * 64}
    assert sandbox_store.referenced_shas(exclude_chat_id=chat_a, exclude_scope="workspace") == {"a" * 64, "b" * 64}
    assert sandbox_store.referenced_shas(exclude_chat_id=chat_b, exclude_scope="workspace") == {"a" * 64}


def test_referenced_shas_excludes_only_the_named_scope(sandbox_store: SandboxStore):
    chat_id = "chat-scopes"
    sandbox_store.write_manifest(chat_id, _manifest("workspace", "a" * 64))
    plot_store = SandboxStore(sandbox_store._store, scope="sandbox_plot")
    plot_store.write_manifest(chat_id, _manifest("plot", "b" * 64))

    assert sandbox_store.referenced_shas(exclude_chat_id=chat_id, exclude_scope="workspace") == {"b" * 64}
    assert sandbox_store.referenced_shas(exclude_chat_id=chat_id) == set()


def test_delete_chat_state_removes_all_records_and_returns_shas(sandbox_store: SandboxStore):
    chat_id = "chat-delete-1"

    sandbox_store.ensure_slot(chat_id, profile="default")
    sandbox_store.write_active(chat_id, ActivePointer(manifest_id="m1", updated_at=1000.0))
    manifest = _manifest("delete-manifest", "a" * 64)
    manifest_id = sandbox_store.write_manifest(chat_id, manifest)
    sandbox_store.write_run(
        chat_id,
        RunRecord(
            tool_call_id="run-1",
            status="completed",
            started_at=1000.0,
            finished_at=1010.0,
            manifest_id=manifest_id,
            result=None,
        ),
    )

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
