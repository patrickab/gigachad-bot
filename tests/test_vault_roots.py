"""Vault-root registry: ``vault_roots`` rows in Postgres mode, JSON file locally.

Only the *registry* moves into the database. Vault file contents stay on their
external filesystem and must never be copied into ``documents``/``assets``.
"""

import json
import os
from uuid import uuid4

import pytest
from psycopg_pool import ConnectionPool

from lib.db_schema import upgrade
from lib.file_vault import FileVault, PostgresVaultRootRepository


@pytest.fixture(scope="module")
def postgres_pool():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for vault root tests")
    upgrade(url)
    pool = ConnectionPool(url, min_size=1, max_size=4)
    yield pool
    pool.close()


@pytest.fixture(autouse=True)
def clean_database(postgres_pool):
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute("TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE")


def make_vault(pool, login="alice@example.test"):
    """A Postgres-backed FileVault for a freshly created user, plus its device id."""
    device_id = uuid4()
    with pool.connection() as connection, connection.transaction():
        user_id = connection.execute("INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", (login,)).fetchone()[0]
        connection.execute("INSERT INTO devices (id, user_id) VALUES (%s, %s)", (device_id, user_id))
    repository = PostgresVaultRootRepository(pool, user_id, device_id=device_id)
    return FileVault(repository=repository), user_id, device_id


def reopen(pool, user_id, device_id=None):
    """A second FileVault over the same rows — proves persistence, not memory."""
    return FileVault(repository=PostgresVaultRootRepository(pool, user_id, device_id=device_id))


def rows(pool):
    with pool.connection() as connection:
        return connection.execute(
            "SELECT user_id, path, project_slug, mountpoints FROM vault_roots ORDER BY path"
        ).fetchall()


def changes(pool):
    with pool.connection() as connection:
        return connection.execute(
            "SELECT resource_key, operation, device_id FROM changes WHERE resource_kind = 'vault_root' ORDER BY seq"
        ).fetchall()


@pytest.fixture
def vault_dir(tmp_path):
    """An external vault directory with real files in it."""
    directory = tmp_path / "vault"
    (directory / "sub").mkdir(parents=True)
    (directory / "note.md").write_text("# note\n")
    (directory / "sub" / "deep.txt").write_text("deep\n")
    return directory


@pytest.fixture
def documents_root(tmp_path, monkeypatch):
    """Redirect the JSON registry at a throwaway Documents tree so writes are visible."""
    root = tmp_path / "Documents" / "chat_history"
    root.mkdir(parents=True)
    monkeypatch.setattr("lib.file_vault.ROOTS_FILE", root / "file-vault-roots.json")
    monkeypatch.setattr("lib.file_vault._LEGACY_ROOTS_FILE", root / "obsidian-roots.json")
    return root


def test_add_root_writes_a_row_a_change_and_no_json_file(postgres_pool, vault_dir, documents_root):
    vault, user_id, device_id = make_vault(postgres_pool)

    vault.add_root(str(vault_dir), project="alpha")

    assert rows(postgres_pool) == [(user_id, str(vault_dir.resolve()), "alpha", [])]
    assert changes(postgres_pool) == [(str(vault_dir.resolve()), "write", device_id)]
    assert list(documents_root.iterdir()) == []
    assert reopen(postgres_pool, user_id).roots() == [str(vault_dir.resolve())]


def test_mountpoint_add_and_remove_update_the_jsonb(postgres_pool, vault_dir, tmp_path):
    vault, user_id, _ = make_vault(postgres_pool)
    mountpoint = tmp_path / "mnt"
    mountpoint.mkdir()
    vault.add_root(str(vault_dir))

    vault.add_mountpoint(str(vault_dir), str(mountpoint))
    assert rows(postgres_pool)[0][3] == [str(mountpoint.resolve())]
    assert reopen(postgres_pool, user_id).mountpoints(str(vault_dir)) == [str(mountpoint.resolve())]

    vault.remove_mountpoint(str(vault_dir), str(mountpoint))
    assert rows(postgres_pool)[0][3] == []
    assert reopen(postgres_pool, user_id).mountpoints(str(vault_dir)) == []
    assert [(operation) for _, operation, _ in changes(postgres_pool)] == ["write", "write", "write"]


def test_remove_root_deletes_the_row(postgres_pool, vault_dir):
    vault, user_id, device_id = make_vault(postgres_pool)
    vault.add_root(str(vault_dir))

    vault.remove_root(str(vault_dir))

    assert rows(postgres_pool) == []
    assert changes(postgres_pool)[-1] == (str(vault_dir.resolve()), "delete", device_id)
    assert reopen(postgres_pool, user_id).roots() == []


def test_identical_root_paths_stay_isolated_per_user(postgres_pool, vault_dir):
    alice, alice_id, _ = make_vault(postgres_pool, "alice@example.test")
    bob, bob_id, _ = make_vault(postgres_pool, "bob@example.test")
    alice.add_root(str(vault_dir))
    bob.add_root(str(vault_dir))

    assert len(rows(postgres_pool)) == 2
    assert {row[0] for row in rows(postgres_pool)} == {alice_id, bob_id}

    bob.remove_root(str(vault_dir))

    assert rows(postgres_pool) == [(alice_id, str(vault_dir.resolve()), None, [])]
    assert reopen(postgres_pool, alice_id).roots() == [str(vault_dir.resolve())]


def test_one_user_cannot_remove_another_users_root(postgres_pool, vault_dir, tmp_path):
    alice, alice_id, _ = make_vault(postgres_pool, "alice@example.test")
    bob, bob_id, _ = make_vault(postgres_pool, "bob@example.test")
    bob_dir = tmp_path / "bob-vault"
    bob_dir.mkdir()
    alice.add_root(str(vault_dir))
    bob.add_root(str(bob_dir))

    bob.remove_root(str(vault_dir))
    bob.remove_mountpoint(str(vault_dir), str(tmp_path))

    assert rows(postgres_pool) == sorted([
        (alice_id, str(vault_dir.resolve()), None, []),
        (bob_id, str(bob_dir.resolve()), None, []),
    ], key=lambda row: row[1])
    assert reopen(postgres_pool, alice_id).roots() == [str(vault_dir.resolve())]


def test_local_mode_still_round_trips_through_the_json_file(vault_dir, documents_root, tmp_path):
    mountpoint = tmp_path / "mnt"
    mountpoint.mkdir()

    vault = FileVault()
    vault.add_root(str(vault_dir), project="alpha")
    vault.add_mountpoint(str(vault_dir), str(mountpoint))

    roots_file = documents_root / "file-vault-roots.json"
    assert json.loads(roots_file.read_text()) == {
        "roots": [
            {
                "path": str(vault_dir.resolve()),
                "mountpoints": [str(mountpoint.resolve())],
                "project": "alpha",
            }
        ]
    }
    assert FileVault().roots() == [str(vault_dir.resolve())]
    assert FileVault().mountpoints(str(vault_dir)) == [str(mountpoint.resolve())]


def test_vault_file_contents_never_enter_the_database(postgres_pool, vault_dir, documents_root, tmp_path):
    before = {path: path.read_bytes() for path in sorted(vault_dir.rglob("*")) if path.is_file()}
    mountpoint = tmp_path / "mnt"
    mountpoint.mkdir()
    (mountpoint / "mounted.md").write_text("mounted\n")

    vault, _, _ = make_vault(postgres_pool)
    vault.add_root(str(vault_dir))
    vault.add_mountpoint(str(vault_dir), str(mountpoint))

    # Listing still reads the external filesystem, not the database.
    assert [f["name"] for f in vault.list_files()] == ["note.md", "deep.txt"]
    assert vault.read(str(vault_dir / "note.md")) == "# note\n"

    with postgres_pool.connection() as connection:
        assert connection.execute("SELECT count(*) FROM assets").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
    assert {path: path.read_bytes() for path in sorted(vault_dir.rglob("*")) if path.is_file()} == before
    assert list(documents_root.iterdir()) == []
