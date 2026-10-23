import json
import os
from uuid import uuid4

from psycopg_pool import ConnectionPool
from psycopg.types.json import Jsonb
import pytest

from lib.architecture_workspace import ArchitectureStore, DirectoryFiles
from lib.asset_store import AssetStore
from lib.chat_store import ChatStore
from lib.data_store import StorageConflictError, StorageNotFoundError
from lib.db_schema import upgrade
from lib.memory_store import MemoryStore
from lib.postgres_data_store import PostgresDataStore
from lib.project_rename import ProjectReferences, renamed_documents
from lib.project_store import ProjectStore


def test_references_move_owned_links_only():
    refs = ProjectReferences("old", "new", "chat")
    assert refs.value({"path": "project/old/document/a.md", "project_slug": "old", "history_file": "old/a.json"}) == {
        "path": "project/new/document/a.md", "project_slug": "new", "history_file": "new/a.json"}
    text = "old oldish project/older/a /home/project/old/a [doc](project/old/document/a.md) memory/project/old/x"
    assert refs.text(text) == "old oldish project/older/a /home/project/old/a [doc](project/new/document/a.md) memory/project/new/x"
    assert refs.value({"path": "/home/project/old/a"}) == {"path": "/home/project/old/a"}
    assert refs.content("model/settings.json", b'{"slug":"old"}') == b'{"slug":"old"}'
    assert refs.content("project/old/document/a.pdf", b"\x00binary old") == b"\x00binary old"
    assert refs.content("note/a.md", b"[x](graph/old/system.c4)") == b"[x](graph/new/system.c4)"


def test_renamed_documents_names_catalog_and_keeps_linked_folder():
    meta = {"projects": [{"name": "Old", "slug": "old", "createdAt": "original"}, {"name": "Older", "slug": "older"}]}
    rows = [("chat/projects-meta.json", json.dumps(meta).encode(), False), ("chat/old/project.json", b'{"name":"Old"}', False),
            ("chat/old", b"", True), ("project/older/document/a.md", b"old prose", False)]
    changed = {target: content for _, target, content in renamed_documents(rows, ProjectReferences("old", "new", "chat"), "New", ["/mnt/old"])}
    entries = json.loads(changed["chat/projects-meta.json"])["projects"]
    assert entries[0] | {"updatedAt": None} == {"name": "New", "slug": "new", "createdAt": "original", "updatedAt": None}
    assert entries[1] == {"name": "Older", "slug": "older"}
    assert json.loads(changed["chat/new/project.json"]) == {"name": "New", "linked_root": "/mnt/old"}
    assert "chat/new" in changed and "project/older/document/a.md" not in changed


@pytest.fixture(scope="module")
def pool():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for transactional rename tests")
    upgrade(url)
    with ConnectionPool(url, min_size=1, max_size=4) as value:
        yield value


@pytest.fixture
def stores(pool):
    with pool.connection() as conn, conn.transaction():
        user = conn.execute("INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", (f"rename-{uuid4()}@example.test",)).fetchone()[0]
    docs = PostgresDataStore(pool, user)
    yield docs, ProjectStore(data_store=docs), AssetStore(pool, user), user
    with pool.connection() as conn, conn.transaction():
        conn.execute("DELETE FROM users WHERE id = %s", (user,))


def put(docs, key, value):
    docs.write_bytes(key, json.dumps(value).encode())


def get(docs, key):
    return json.loads(docs.read_bytes(key)[0])


def test_rename_cascades_consumers_and_preserves_identity(stores, pool, tmp_path):
    docs, projects, assets, user = stores
    projects.create_project("Old")
    projects.create_project("Other")
    projects.update_project("old", archived=True)
    mount = tmp_path / "old"
    source = mount / "docs/architecture/system.c4"
    source.parent.mkdir(parents=True)
    source.write_text("model { system = system 'Original' }")
    with pool.connection() as conn, conn.transaction():
        conn.execute("INSERT INTO vault_roots (user_id, path) VALUES (%s, %s)", (user, str(mount)))
    chats = ChatStore(data_store=docs)
    chats.save("old/main.json", {"chat_id": "parent", "messages": [{"role": "user", "content": "old remains prose"}]})
    branch = chats.branch("old/main.json", 0)["child_file"]
    # Prime the ID index before moving its keys.
    projects._store.find_by_chat_id("parent")
    attachment = {"id": "a", "path": "project/old/document/report.md", "projectSlug": "old"}
    canvas = "project/old/document/board.canvas"
    put(docs, canvas, {"version": 1, "attachments": [attachment]})
    put(docs, "project/other/document/cross.canvas", {"attachments": [attachment]})
    docs.write_bytes("project/old/document/report.md", b"old [self](project/old/document/report.md)")
    docs.write_bytes("graph/old/system.c4", b"model {}")
    docs.write_bytes("memory/project/old/memory/memory.md", b"old facts")
    put(docs, "memory/project/old/memory/memory.json", [])
    put(docs, "chat/other/external.json", {"messages": [{"content": "[report](project/old/document/report.md)", "attachments": [{"vaultPath": attachment["path"]}]}]})
    put(docs, "model/unrelated.json", {"slug": "old"})
    put(docs, "project/older/document/untouched.json", {"text": "old"})
    asset = assets.write("upload", "attachment/project/old/chat/parent/image.png", b"\x00\xffimage", mime="image/png")
    mutation = uuid4()
    secret = b"share-secret"
    with pool.connection() as conn, conn.transaction():
        conn.execute("INSERT INTO canvas_mutations (mutation_id,user_id,canvas_key,revision,position,kind,entity,entity_id,value) VALUES (%s,%s,%s,1,0,'upsert','attachment','a',%s)", (mutation, user, canvas, Jsonb(attachment)))
        conn.execute("INSERT INTO canvas_shares (secret_hash,user_id,canvas_key) VALUES (%s,%s,%s)", (secret, user, canvas))
    before = get(docs, "chat/projects-meta.json")["projects"][0]
    with pool.connection() as conn:
        last_seq = conn.execute("SELECT max(seq) FROM changes WHERE user_id=%s", (user,)).fetchone()[0]
    assert projects.update_project("old", "New Name") == {"name": "New Name", "slug": "new-name", "archived": True}
    after = get(docs, "chat/projects-meta.json")["projects"][0]
    assert after["createdAt"] == before["createdAt"]
    assert projects.get_project("new-name")["linked_root"] == str(mount)
    with pytest.raises(FileNotFoundError):
        projects.get_project("old")
    assert projects._store.find_by_chat_id("parent") == "new-name/main.json"
    assert chats.load(branch.replace("old/", "new-name/", 1))["parent_id"] == "parent"
    assert get(docs, "project/other/document/cross.canvas")["attachments"][0]["path"] == "project/new-name/document/report.md"
    assert "project/new-name/document/report.md" in get(docs, "chat/other/external.json")["messages"][0]["content"]
    assert docs.read_bytes("graph/new-name/system.c4")[0] == b"model {}"
    assert MemoryStore(data_store=docs, model="test")._project_memory_path("new-name").read_bytes() == b"old facts"
    assert isinstance(ArchitectureStore.for_project(docs, "new-name", [str(mount)])._files, DirectoryFiles)
    assert source.read_text() == "model { system = system 'Original' }"
    moved = assets.read_by_id(asset.id)
    assert moved.logical_path == "attachment/project/new-name/chat/parent/image.png"
    assert moved.content == asset.content and moved.sha256 == asset.sha256
    with pytest.raises(StorageNotFoundError):
        assets.read(asset.logical_path)
    assert get(docs, "model/unrelated.json") == {"slug": "old"}
    assert get(docs, "project/older/document/untouched.json") == {"text": "old"}
    with pool.connection() as conn:
        events = conn.execute("SELECT resource_kind, resource_key FROM changes WHERE user_id=%s AND seq>%s", (user, last_seq)).fetchall()
        assert {("document", "chat/old/main.json"), ("document", "chat/new-name/main.json"), ("asset", asset.logical_path)} <= set(events)
        assert conn.execute("SELECT canvas_key,value FROM canvas_mutations WHERE mutation_id=%s", (mutation,)).fetchone() == (
            "project/new-name/document/board.canvas", {**attachment, "path": "project/new-name/document/report.md", "projectSlug": "new-name"})
        assert conn.execute("SELECT canvas_key FROM canvas_shares WHERE secret_hash=%s", (secret,)).fetchone()[0] == "project/new-name/document/board.canvas"
        assert conn.execute("SELECT path FROM vault_roots WHERE user_id=%s", (user,)).fetchone()[0] == str(mount)


def test_same_slug_renames_in_place_and_taken_slug_conflicts(stores):
    docs, projects, _, _ = stores
    projects.create_project("Old")
    assert projects.update_project("old", "OLD!") == {"name": "OLD!", "slug": "old", "archived": False}
    projects.create_project("New")
    with pytest.raises(StorageConflictError):
        projects.update_project("old", "new")
    with pytest.raises(ValueError):
        projects.update_project("old", " ")
    assert projects.get_project("old")["name"] == "OLD!"


def test_asset_conflict_rolls_back_every_document_and_change(stores, pool):
    docs, projects, assets, user = stores
    projects.create_project("Old")
    docs.write_bytes("project/old/document/a.md", b"hello")
    assets.write("upload", "attachment/project/old/chat/id/a.txt", b"source")
    assets.write("upload", "attachment/project/new/chat/id/a.txt", b"destination")
    with pool.connection() as conn:
        count = conn.execute("SELECT count(*) FROM changes WHERE user_id=%s", (user,)).fetchone()[0]
    with pytest.raises(StorageConflictError):
        projects.update_project("old", "New", archived=True)
    assert projects.get_project("old")["name"] == "Old"
    assert docs.read_bytes("project/old/document/a.md")[0] == b"hello"
    assert not docs.exists("project/new")
    assert assets.read("attachment/project/old/chat/id/a.txt").content == b"source"
    assert assets.read("attachment/project/new/chat/id/a.txt").content == b"destination"
    with pool.connection() as conn:
        assert conn.execute("SELECT count(*) FROM changes WHERE user_id=%s", (user,)).fetchone()[0] == count
    assert projects.list_projects()[0]["archived"] is False


def test_rename_never_crosses_user_namespace(stores, pool):
    docs, projects, _, _ = stores
    projects.create_project("Old")
    with pool.connection() as conn, conn.transaction():
        other_user = conn.execute("INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", (f"other-{uuid4()}@example.test",)).fetchone()[0]
    try:
        other_docs = PostgresDataStore(pool, other_user)
        other_projects = ProjectStore(data_store=other_docs)
        other_projects.create_project("Old")
        other_docs.write_bytes("project/old/document/a.md", b"[old](project/old/document/a.md)")
        projects.update_project("old", "New")
        assert other_projects.get_project("old")["name"] == "Old"
        assert other_docs.read_bytes("project/old/document/a.md")[0] == b"[old](project/old/document/a.md)"
        assert not other_docs.exists("project/new")
    finally:
        with pool.connection() as conn, conn.transaction():
            conn.execute("DELETE FROM users WHERE id = %s", (other_user,))
