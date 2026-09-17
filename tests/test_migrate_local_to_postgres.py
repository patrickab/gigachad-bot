import json
import os
from pathlib import Path
import sys
from uuid import UUID

import pytest
from psycopg_pool import ConnectionPool

from lib.db_schema import upgrade

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from migrate_local_to_postgres import main as migrate_main  # noqa: E402
from verify_postgres_migration import main as verify_main  # noqa: E402

LOGIN = "alice@example.test"
VAULT_NOTE = "vault content that must never be copied\n"
PROMPT = "# system prompt\n"

EXPECTED_DOCUMENTS = {
    "Architecture_Graphs/services.yaml",
    "Prompts/system.md",
    "chat_history/chat-1.json",
    "model-providers.yaml",
}
EXPECTED_ASSETS = {
    "Drawings/sketch.excalidraw": "drawing",
    "Mineru/images/paper-fig1.png": "mineru_image",
    "Mineru/paper.md": "mineru_markdown",
    "PDFs/paper.pdf": "pdf",
    "chat_history/_uploads/chat-1/notes.txt": "upload",
}


@pytest.fixture(scope="module")
def postgres_pool():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for migration tests")
    upgrade(url)
    pool = ConnectionPool(url, min_size=1, max_size=4)
    yield pool
    pool.close()


@pytest.fixture(autouse=True)
def clean_database(postgres_pool):
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute("TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE")


@pytest.fixture
def vault(tmp_path):
    """An external file vault plus a mounted drive — referenced, never imported."""
    root = tmp_path / "ExternalVault"
    root.mkdir()
    (root / "note.md").write_text(VAULT_NOTE)
    mount = tmp_path / "MountedDrive"
    mount.mkdir()
    (mount / "mounted.md").write_text(VAULT_NOTE)
    return root, mount


@pytest.fixture
def source(tmp_path, vault):
    root, mount = vault
    roots_config = {"roots": [{"path": str(root), "mountpoints": [str(mount)], "project": "alpha"}]}
    files = {
        "chat_history/chat-1.json": json.dumps({"id": "chat-1", "messages": []}),
        "chat_history/_uploads/chat-1/notes.txt": "uploaded attachment\n",
        "chat_history/file-vault-roots.json": json.dumps(roots_config),
        "Prompts/system.md": PROMPT,
        "Architecture_Graphs/services.yaml": "nodes:\n  - id: api\n",
        "PDFs/paper.pdf": "%PDF-1.7\nfake pdf bytes\n",
        "Mineru/paper.md": "# paper\n\n![fig](images/paper-fig1.png)\n",
        "Mineru/images/paper-fig1.png": "fake png bytes\n",
        "Mineru/paper_content_list.json": "[]",
        "Drawings/sketch.excalidraw": json.dumps({"elements": []}),
        "model-providers.yaml": "Ollama:\n  litellm_id: ollama\n",
    }
    documents = tmp_path / "Documents"
    for key, content in files.items():
        path = documents / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return documents


@pytest.fixture
def run(source, postgres_pool):
    url = os.environ["POSTGRES_TEST_DATABASE_URL"]

    def _run(entrypoint, *extra):
        return entrypoint(["--tailscale-login", LOGIN, "--database-url", url, "--source", str(source), *extra])

    return _run


def _skipped_per_group(summary: str) -> dict[str, int]:
    return {
        line.split(":")[0]: int(line.split("skipped ")[1].split(",")[0])
        for line in summary.splitlines()
        if ": imported " in line or ": would import " in line
    }


def test_migration_imports_each_group_into_its_own_table(run, postgres_pool, vault):
    root, mount = vault

    assert run(migrate_main) == 0

    with postgres_pool.connection() as connection:
        documents = {row[0] for row in connection.execute("SELECT key FROM documents").fetchall()}
        assets = dict(connection.execute("SELECT logical_path, kind FROM assets").fetchall())
        roots = connection.execute("SELECT path, project_slug, mountpoints FROM vault_roots").fetchall()
        kinds = {row[0] for row in connection.execute("SELECT DISTINCT resource_kind FROM changes").fetchall()}

    assert documents == EXPECTED_DOCUMENTS
    assert assets == EXPECTED_ASSETS
    assert roots == [(str(root), "alpha", [str(mount)])]
    assert kinds == {"document", "asset", "vault_root"}


def test_dry_run_writes_nothing(run, postgres_pool, capsys):
    assert run(migrate_main, "--dry-run") == 0

    assert "documents: would import 4" in capsys.readouterr().out
    with postgres_pool.connection() as connection:
        assert connection.execute("SELECT count(*) FROM users").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM documents").fetchone()[0] == 0


def test_verifier_accepts_a_completed_migration(run, capsys):
    assert run(migrate_main) == 0
    capsys.readouterr()

    assert run(verify_main) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["problems"] == []
    assert report["groups"]["documents"] == {"source": 4, "database": 4}
    assert report["groups"]["vault_roots"] == {"source": 1, "database": 1}
    assert report["foreign_rows"] == {"documents": 0, "assets": 0}
    UUID(report["user_id"])


def test_verifier_rejects_a_missing_document(run, postgres_pool, capsys):
    assert run(migrate_main) == 0
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute("DELETE FROM documents WHERE key = 'Prompts/system.md'")
    capsys.readouterr()

    assert run(verify_main) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert "documents: missing Prompts/system.md" in report["problems"]


def test_verifier_rejects_an_asset_without_its_mirror(run, source, capsys):
    assert run(migrate_main) == 0
    (source / "PDFs" / "paper.pdf").unlink()
    capsys.readouterr()

    assert run(verify_main) == 1

    report = json.loads(capsys.readouterr().out)
    assert "assets: no filesystem mirror for PDFs/paper.pdf" in report["problems"]


def test_second_run_imports_nothing_and_still_verifies(run, capsys):
    assert run(migrate_main) == 0
    capsys.readouterr()

    assert run(migrate_main) == 0

    summary = capsys.readouterr().out
    assert _skipped_per_group(summary) == {
        "documents": 4,
        "upload": 1,
        "pdf": 1,
        "mineru_markdown": 1,
        "mineru_image": 1,
        "drawing": 1,
        "vault_roots": 1,
        "empty_directories": 0,
    }
    assert "imported 0" in summary
    assert "imported 1" not in summary
    assert run(verify_main) == 0


def test_changed_source_file_fails_loudly(run, source, postgres_pool, capsys):
    assert run(migrate_main) == 0
    (source / "Prompts" / "system.md").write_text("# tampered\n")
    capsys.readouterr()

    assert run(migrate_main) == 1

    captured = capsys.readouterr()
    assert "Prompts/system.md" in captured.err
    assert "documents: imported 0, skipped 3, failed 1" in captured.out
    with postgres_pool.connection() as connection:
        stored = connection.execute("SELECT content FROM documents WHERE key = 'Prompts/system.md'").fetchone()[0]
    assert bytes(stored) == PROMPT.encode()


def test_changed_vault_config_fails_loudly(run, source, vault, capsys):
    root, _mount = vault
    assert run(migrate_main) == 0
    (source / "chat_history" / "file-vault-roots.json").write_text(json.dumps({"roots": [{"path": str(root)}]}))
    capsys.readouterr()

    assert run(migrate_main) == 1

    captured = capsys.readouterr()
    assert str(root) in captured.err
    assert "vault_roots: imported 0, skipped 0, failed 1" in captured.out


def test_vault_contents_are_never_copied(run, postgres_pool, vault):
    root, mount = vault

    assert run(migrate_main) == 0

    with postgres_pool.connection() as connection:
        keys = [
            row[0]
            for row in connection.execute("SELECT key FROM documents UNION ALL SELECT logical_path FROM assets").fetchall()
        ]
        copies = connection.execute(
            """
            SELECT (SELECT count(*) FROM documents WHERE content = %(note)s)
                 + (SELECT count(*) FROM assets WHERE content = %(note)s)
            """,
            {"note": VAULT_NOTE.encode()},
        ).fetchone()[0]

    assert copies == 0
    assert not [key for key in keys if root.name in key or mount.name in key]


def test_empty_user_folders_survive_the_migration(run, source, postgres_pool):
    # A chat folder the user created but never filled holds no bytes, so it only
    # exists as a directory. Without a marker row it would vanish from the sidebar.
    (source / "chat_history" / "empty-folder" / "nested").mkdir(parents=True)

    assert run(migrate_main) == 0

    with postgres_pool.connection() as connection:
        markers = [
            row[0] for row in connection.execute("SELECT key FROM documents WHERE is_dir ORDER BY key").fetchall()
        ]

    assert markers == ["chat_history/empty-folder", "chat_history/empty-folder/nested"]
    # Markers carry no content, so the verifier must still pass unchanged.
    assert run(verify_main) == 0
