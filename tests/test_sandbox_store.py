import os
from uuid import uuid4

from psycopg_pool import ConnectionPool
import pytest

from lib.agent_sandbox_adapter import AssetRef, SandboxStateRecord
from lib.data_store import DataStore, StorageNotFoundError
from lib.db_schema import upgrade
from lib.postgres_data_store import PostgresDataStore
from lib.sandbox_store import RunRecord, SandboxStore, manifest_id_for
from lib.storage_namespace import sandbox_run


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


def test_manifest_id_stable_across_independent_builds():
    ref1 = AssetRef(sha256="a" * 64, media_type="application/octet-stream", size_bytes=100)
    ref2 = AssetRef(sha256="b" * 64, media_type="application/octet-stream", size_bytes=200)
    manifest1 = SandboxStateRecord(version=1, workspace=ref1, omp_session=ref2, extra={"foo": "bar"})
    manifest2 = SandboxStateRecord(version=1, workspace=ref1, omp_session=ref2, extra={"foo": "bar"})

    id1 = manifest_id_for(manifest1)
    id2 = manifest_id_for(manifest2)

    assert id1 == id2
    assert len(id1) == 64

def test_recover_completed_run_rebuilds_missing_run_history(sandbox_store: SandboxStore):
    manifest = SandboxStateRecord(version=1, workspace=AssetRef("a" * 64, "application/x-tar", 100), extra={})
    result = {
        "status": "completed",
        "summary": "done",
        "manifest_id": manifest_id_for(manifest),
        "workspace_changed": True,
        "outputs": [],
        "error": None,
    }
    record = RunRecord(tool_call_id="call-1", status="completed", started_at=1.0, finished_at=2.0)
    sandbox_store.complete_run("chat-recover", manifest=manifest, record=record, result=result)
    sandbox_store._store.delete(sandbox_run("chat-recover", "call-1"))

    recovered = sandbox_store.recover_completed_run("chat-recover", "call-1")

    assert recovered is not None
    assert recovered.result == result
    assert sandbox_store.read_run("chat-recover", "call-1") == recovered


def test_referenced_assets_collects_from_manifests_and_run_results(sandbox_store: SandboxStore):
    chat_id = "chat-assets-1"

    ref_ws = AssetRef(sha256="w" * 64, media_type="application/zip", size_bytes=1000)
    ref_omp = AssetRef(sha256="o" * 64, media_type="application/json", size_bytes=500)
    manifest1 = SandboxStateRecord(version=1, workspace=ref_ws, omp_session=ref_omp, extra={})
    sandbox_store.write_manifest(chat_id, manifest1)

    ref_ws2 = AssetRef(sha256="x" * 64, media_type="application/zip", size_bytes=2000)
    manifest2 = SandboxStateRecord(version=1, workspace=ref_ws2, omp_session=None, extra={})
    sandbox_store.write_manifest(chat_id, manifest2)

    from lib.sandbox_store import RunRecord
    output_ref = AssetRef(sha256="y" * 64, media_type="image/png", size_bytes=300)
    run_record = RunRecord(
        tool_call_id="run-1",
        status="completed",
        started_at=1000.0,
        finished_at=1010.0,
        manifest_id=manifest_id_for(manifest1),
        result={
            "outputs": [
                {"display_id": "img-1", "title": "Image", "text": None, "mime_bundle": {"image/png": output_ref.to_json()}}
            ]
        },
    )
    sandbox_store.write_run(chat_id, run_record)

    refs = sandbox_store.referenced_assets(chat_id)
    ref_shas = {r.sha256 for r in refs}

    assert "w" * 64 in ref_shas
    assert "o" * 64 in ref_shas
    assert "x" * 64 in ref_shas
    assert "y" * 64 in ref_shas
    assert len(refs) == 4


def test_asset_is_referenced_across_chats_and_exclude(sandbox_store: SandboxStore):
    chat_a = "chat-a"
    chat_b = "chat-b"

    shared_ref = AssetRef(sha256="s" * 64, media_type="application/octet-stream", size_bytes=100)
    manifest_a = SandboxStateRecord(version=1, workspace=shared_ref, extra={})
    sandbox_store.write_manifest(chat_a, manifest_a)

    other_ref = AssetRef(sha256="o" * 64, media_type="application/octet-stream", size_bytes=200)
    manifest_b = SandboxStateRecord(version=1, workspace=other_ref, extra={})
    sandbox_store.write_manifest(chat_b, manifest_b)

    assert sandbox_store.asset_is_referenced("s" * 64) is True
    assert sandbox_store.asset_is_referenced("s" * 64, exclude_chat_id=chat_a) is False
    assert sandbox_store.asset_is_referenced("o" * 64, exclude_chat_id=chat_b) is False
    assert sandbox_store.asset_is_referenced("u" * 64) is False


def test_delete_chat_state_removes_all_records_and_returns_refs(sandbox_store: SandboxStore):
    chat_id = "chat-delete-1"

    sandbox_store.ensure_slot(chat_id, profile="default")
    from lib.sandbox_store import ActivePointer
    sandbox_store.write_active(chat_id, ActivePointer(manifest_id="m1", updated_at=1000.0))
    ref = AssetRef(sha256="r" * 64, media_type="application/octet-stream", size_bytes=100)
    manifest = SandboxStateRecord(version=1, workspace=ref, extra={})
    manifest_id = sandbox_store.write_manifest(chat_id, manifest)
    from lib.sandbox_store import RunRecord
    sandbox_store.write_run(chat_id, RunRecord(
        tool_call_id="run-1",
        status="completed",
        started_at=1000.0,
        finished_at=1010.0,
        manifest_id=manifest_id,
        result=None,
    ))

    refs = sandbox_store.delete_chat_state(chat_id)
    assert len(refs) == 1
    assert refs[0].sha256 == "r" * 64

    assert sandbox_store.read_slot(chat_id) is None
    assert sandbox_store.read_active(chat_id) is None
    with pytest.raises(StorageNotFoundError):
        sandbox_store.read_manifest(chat_id, manifest_id)
    assert sandbox_store.read_run(chat_id, "run-1") is None


def test_delete_chat_state_idempotent_on_missing_chat(sandbox_store: SandboxStore):
    chat_id = "chat-never-existed"

    refs = sandbox_store.delete_chat_state(chat_id)
    assert refs == []

    refs2 = sandbox_store.delete_chat_state(chat_id)
    assert refs2 == []
