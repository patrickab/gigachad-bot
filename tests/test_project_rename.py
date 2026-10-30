import json
from uuid import uuid4

from psycopg.types.json import Jsonb
import pytest

from lib.architecture_workspace import ArchitectureStore, DirectoryFiles
from lib.asset_store import AssetStore
from lib.changes import UserScope
from lib.chat_store import ChatStore
from lib.data_store import StorageConflictError, StorageNotFoundError
from lib.memory_store import MemoryScope, MemoryStore
from lib.postgres_data_store import PostgresDataStore
from lib.project_rename import PostgresKeyRelocator, PostgresProjectRenamer, ProjectReferences, renamed_documents
from lib.project_store import ProjectStore
from tests.conftest import create_user


def test_references_move_owned_links_only():
    refs = ProjectReferences("old", "new")
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
    changed = {target: content for _, target, content in renamed_documents(rows, ProjectReferences("old", "new"), "New", ["/mnt/old"])}
    entries = json.loads(changed["chat/projects-meta.json"])["projects"]
    assert entries[0] | {"updatedAt": None} == {"name": "New", "slug": "new", "createdAt": "original", "updatedAt": None}
    assert entries[1] == {"name": "Older", "slug": "older"}
    assert json.loads(changed["chat/new/project.json"]) == {"name": "New", "linked_root": "/mnt/old"}
    assert "chat/new" in changed and "project/older/document/a.md" not in changed


def test_project_rename_moves_named_file_entries_and_keeps_their_names():
    refs = ProjectReferences("old", "new")
    files = ["project/old/document/a.md", {"path": "project/old/document/b.md", "name": "Plan.md"}]
    assert refs.value({"files": files}) == {
        "files": ["project/new/document/a.md", {"path": "project/new/document/b.md", "name": "Plan.md"}],
    }


def test_renamed_project_json_keeps_learn_session_ids():
    rows = [("chat/old/project.json", json.dumps({"name": "Old", "learn_sessions": ["chat_old_1"]}).encode(), False)]
    [(_, target, content)] = renamed_documents(rows, ProjectReferences("old", "new"), "New", [])
    assert target == "chat/new/project.json"
    assert json.loads(content)["learn_sessions"] == ["chat_old_1"]


@pytest.fixture
def stores(postgres_pool, clean_database):
    user = create_user(postgres_pool)
    docs = PostgresDataStore(UserScope(postgres_pool, user))
    return docs, project_store(docs, UserScope(postgres_pool, user)), AssetStore(UserScope(postgres_pool, user)), user


def project_store(docs, scope):
    return ProjectStore(ChatStore(data_store=docs), rename=PostgresProjectRenamer(scope).rename)


def put(docs, key, value):
    docs.write_bytes(key, json.dumps(value).encode())


def get(docs, key):
    return json.loads(docs.read_bytes(key)[0])


def test_rename_cascades_consumers_and_preserves_identity(stores, postgres_pool, tmp_path):
    docs, projects, assets, user = stores
    projects.create_project("Old")
    projects.create_project("Other")
    projects.update_project("old", archived=True)
    mount = tmp_path / "old"
    source = mount / "docs/architecture/system.c4"
    source.parent.mkdir(parents=True)
    source.write_text("model { system = system 'Original' }")
    with postgres_pool.connection() as conn, conn.transaction():
        conn.execute("INSERT INTO vault_roots (user_id, path) VALUES (%s, %s)", (user, str(mount)))
    chats = ChatStore(data_store=docs)
    chats.save("old/main.json", {"chat_id": "parent", "messages": [{"role": "user", "content": "old remains prose"}]})
    branch = chats.branch("old/main.json", 0)["child_file"]
    # Prime the ID index before moving its keys.
    projects._chats._find("parent")
    attachment = {"id": "a", "path": "project/old/document/report.md", "projectSlug": "old"}
    canvas = "project/old/document/board.canvas"
    put(docs, canvas, {"version": 1, "attachments": [attachment]})
    put(docs, "project/other/document/cross.canvas", {"attachments": [attachment]})
    docs.write_bytes("project/old/document/report.md", b"old [self](project/old/document/report.md)")
    docs.write_bytes("graph/old/system.c4", b"model {}")
    put(docs, "memory/project/old/memory/memory.json", [{"id": "f", "text": "old fact", "topic": "note"}])
    put(docs, "chat/other/external.json", {"messages": [{"content": "[report](project/old/document/report.md)", "attachments": [{"vaultPath": attachment["path"]}]}]})
    put(docs, "model/unrelated.json", {"slug": "old"})
    put(docs, "project/older/document/untouched.json", {"text": "old"})
    asset = assets.write("upload", "attachment/project/old/chat/parent/image.png", b"\x00\xffimage", mime="image/png")
    mutation = uuid4()
    secret = b"share-secret"
    with postgres_pool.connection() as conn, conn.transaction():
        conn.execute("INSERT INTO canvas_mutations (mutation_id,user_id,canvas_key,revision,position,kind,entity,entity_id,value) VALUES (%s,%s,%s,1,0,'upsert','attachment','a',%s)", (mutation, user, canvas, Jsonb(attachment)))
        conn.execute("INSERT INTO canvas_shares (secret_hash,user_id,canvas_key) VALUES (%s,%s,%s)", (secret, user, canvas))
    before = get(docs, "chat/projects-meta.json")["projects"][0]
    with postgres_pool.connection() as conn:
        last_seq = conn.execute("SELECT max(seq) FROM changes WHERE user_id=%s", (user,)).fetchone()[0]
    assert projects.update_project("old", "New Name") == {"name": "New Name", "slug": "new-name", "archived": True}
    after = get(docs, "chat/projects-meta.json")["projects"][0]
    assert after["createdAt"] == before["createdAt"]
    assert projects.get_project("new-name")["linked_root"] == str(mount)
    with pytest.raises(FileNotFoundError):
        projects.get_project("old")
    assert projects._chats._find("parent") == "chat/new-name/main.json"
    assert chats.load(branch.replace("old/", "new-name/", 1))["parent_id"] == "parent"
    assert get(docs, "project/other/document/cross.canvas")["attachments"][0]["path"] == "project/new-name/document/report.md"
    assert "project/new-name/document/report.md" in get(docs, "chat/other/external.json")["messages"][0]["content"]
    assert docs.read_bytes("graph/new-name/system.c4")[0] == b"model {}"
    assert [m["text"] for m in MemoryStore(docs).list_memories(MemoryScope("project", "new-name"))] == ["old fact"]
    assert isinstance(ArchitectureStore.for_project(docs, "new-name", [str(mount)])._files, DirectoryFiles)
    assert source.read_text() == "model { system = system 'Original' }"
    moved = assets.read_by_id(asset.id)
    assert moved.logical_path == "attachment/project/new-name/chat/parent/image.png"
    assert moved.content == asset.content and moved.sha256 == asset.sha256
    with pytest.raises(StorageNotFoundError):
        assets.read(asset.logical_path)
    assert get(docs, "model/unrelated.json") == {"slug": "old"}
    assert get(docs, "project/older/document/untouched.json") == {"text": "old"}
    with postgres_pool.connection() as conn:
        events = conn.execute("SELECT resource_kind, resource_key FROM changes WHERE user_id=%s AND seq>%s", (user, last_seq)).fetchall()
        assert {("document", "chat/old/main.json"), ("document", "chat/new-name/main.json"), ("asset", asset.logical_path)} <= set(events)
        assert conn.execute("SELECT canvas_key,value FROM canvas_mutations WHERE mutation_id=%s", (mutation,)).fetchone() == (
            "project/new-name/document/board.canvas", {**attachment, "path": "project/new-name/document/report.md", "projectSlug": "new-name"})
        assert conn.execute("SELECT canvas_key FROM canvas_shares WHERE secret_hash=%s", (secret,)).fetchone()[0] == "project/new-name/document/board.canvas"
        assert conn.execute("SELECT path FROM vault_roots WHERE user_id=%s", (user,)).fetchone()[0] == str(mount)


def test_same_slug_renames_in_place_and_taken_slug_conflicts(stores):
    _, projects, _, _ = stores
    projects.create_project("Old")
    assert projects.update_project("old", "OLD!") == {"name": "OLD!", "slug": "old", "archived": False}
    projects.create_project("New")
    with pytest.raises(StorageConflictError):
        projects.update_project("old", "new")
    with pytest.raises(ValueError):
        projects.update_project("old", " ")
    assert projects.get_project("old")["name"] == "OLD!"


def test_rename_keeps_the_projects_learn_sessions(stores):
    _, projects, _, _ = stores
    projects.create_project("Old")
    projects.assign_learn_session("chat_mvtsqwg2_5e0e6463", "old")

    projects.update_project("old", "New")

    assert projects.learn_sessions("new") == ["chat_mvtsqwg2_5e0e6463"]
    assert projects.learn_session_projects() == {"chat_mvtsqwg2_5e0e6463": "new"}


def test_asset_conflict_rolls_back_every_document_and_change(stores, postgres_pool):
    docs, projects, assets, user = stores
    projects.create_project("Old")
    docs.write_bytes("project/old/document/a.md", b"hello")
    assets.write("upload", "attachment/project/old/chat/id/a.txt", b"source")
    assets.write("upload", "attachment/project/new/chat/id/a.txt", b"destination")
    with postgres_pool.connection() as conn:
        count = conn.execute("SELECT count(*) FROM changes WHERE user_id=%s", (user,)).fetchone()[0]
    with pytest.raises(StorageConflictError):
        projects.update_project("old", "New", archived=True)
    assert projects.get_project("old")["name"] == "Old"
    assert docs.read_bytes("project/old/document/a.md")[0] == b"hello"
    assert not docs.exists("project/new")
    assert assets.read("attachment/project/old/chat/id/a.txt").content == b"source"
    assert assets.read("attachment/project/new/chat/id/a.txt").content == b"destination"
    with postgres_pool.connection() as conn:
        assert conn.execute("SELECT count(*) FROM changes WHERE user_id=%s", (user,)).fetchone()[0] == count
    assert projects.list_projects()[0]["archived"] is False


def test_rename_never_crosses_user_namespace(stores, postgres_pool):
    _, projects, _, _ = stores
    projects.create_project("Old")
    other_user = create_user(postgres_pool)
    other_docs = PostgresDataStore(UserScope(postgres_pool, other_user))
    other_projects = project_store(other_docs, UserScope(postgres_pool, other_user))
    other_projects.create_project("Old")
    other_docs.write_bytes("project/old/document/a.md", b"[old](project/old/document/a.md)")
    projects.update_project("old", "New")
    assert other_projects.get_project("old")["name"] == "Old"
    assert other_docs.read_bytes("project/old/document/a.md")[0] == b"[old](project/old/document/a.md)"
    assert not other_docs.exists("project/new")


@pytest.mark.parametrize("new", ["project/other/document/report.pdf", "/mnt/docs/report.pdf"])
def test_relocating_a_document_moves_every_reference_and_only_exact_ones(stores, postgres_pool, new):
    docs, projects, _, user = stores
    projects.create_project("Old")
    projects.create_project("Other")
    old = "project/old/document/report.pdf"
    docs.write_bytes(old, b"%PDF")
    projects.add_file("old", old)
    projects.add_file("other", old)
    put(docs, "chat/old/main.json", {"messages": [{"attachments": [{"vaultPath": old}, {"vaultPath": old + ".bak"}]}]})
    canvas = "project/old/document/board.canvas"
    put(docs, canvas, {"attachments": [{"id": "a", "path": old}]})
    docs.write_bytes("note/links.md", f"[r]({old}) and {old}x".encode())
    mutation = uuid4()
    with postgres_pool.connection() as conn, conn.transaction():
        conn.execute(
            "INSERT INTO canvas_mutations (mutation_id,user_id,canvas_key,revision,position,kind,entity,entity_id,value)"
            " VALUES (%s,%s,%s,1,0,'upsert','attachment','a',%s)", (mutation, user, canvas, Jsonb({"id": "a", "path": old})))

    PostgresKeyRelocator(UserScope(postgres_pool, user)).relocate(old, new)

    assert projects.list_files("old") == projects.list_files("other") == [new]
    assert get(docs, "chat/old/main.json")["messages"][0]["attachments"] == [{"vaultPath": new}, {"vaultPath": old + ".bak"}]
    assert get(docs, canvas)["attachments"][0]["path"] == new
    assert docs.read_bytes("note/links.md")[0] == f"[r]({new}) and {old}x".encode()
    assert not docs.exists(old)
    assert new.startswith("/") or docs.read_bytes(new)[0] == b"%PDF"
    with postgres_pool.connection() as conn:
        value = conn.execute("SELECT value FROM canvas_mutations WHERE mutation_id=%s", (mutation,)).fetchone()[0]
    assert value == {"id": "a", "path": new}
