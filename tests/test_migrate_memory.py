import importlib.util
import io
from pathlib import Path
import sys

from memory_store import store
import pytest

from lib.changes import UserScope
from lib.chat_store import ChatStore
from lib.json_docs import JsonDocs
from lib.postgres_data_store import PostgresDataStore
from lib.project_store import ProjectStore
from tests.conftest import create_user

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_memory.py"
_spec = importlib.util.spec_from_file_location("migrate_memory", SCRIPT)
assert _spec is not None and _spec.loader is not None
migrate_memory = sys.modules["migrate_memory"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate_memory)

pytestmark = pytest.mark.usefixtures("clean_database")


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """Every login's memory root inside tmp_path: `host` for the host login, else the user id."""

    def root(user_id, host_access):
        return tmp_path / ("host" if host_access else str(user_id))

    monkeypatch.setattr(migrate_memory, "memory_root", root)
    return tmp_path


def seed_old_memories(postgres_pool, login):
    user = create_user(postgres_pool, login)
    docs = PostgresDataStore(UserScope(postgres_pool, user))
    json_docs = JsonDocs(docs)
    json_docs.write(
        "memory/global-topics.json",
        [
            {"name": "profile", "summary": "Who the user is"},
            {"name": "dev_environment", "summary": ""},
        ],
    )
    json_docs.write(
        "memory/global-profile.json",
        [
            {"id": "1", "text": "Studies physics", "topic": "profile"},
            {"id": "2", "text": "Uses Neovim", "topic": "dev_environment"},
            {"id": "3", "text": "Lives in Berlin", "topic": "profile"},
            {"id": "4", "text": "  ", "topic": "profile"},
        ],
    )
    projects = ProjectStore(ChatStore(data_store=docs), rename=lambda *_: None)
    projects.create_project("Thesis")
    json_docs.write("memory/project/thesis/memory/topics.json", [{"name": "deadlines", "summary": "Due dates"}])
    json_docs.write("memory/project/thesis/memory/memory.json", [{"id": "5", "text": "Draft due in May", "topic": "deadlines"}])
    json_docs.write("memory/project/gone/memory/memory.json", [{"id": "6", "text": "Archived fact"}])
    json_docs.write("memory/watermarks.json", {"chat": {"global": 3}})
    return user, docs


def run(postgres_pool, **flags):
    out = io.StringIO()
    migrate_memory.migrate(postgres_pool, "host@example.test", out=out, **{"dry_run": False, "delete_old": False, **flags})
    return out.getvalue()


def test_migration_moves_topics_into_memories_and_keeps_relevance_as_recency(postgres_pool, roots):
    host, _ = seed_old_memories(postgres_pool, "host@example.test")
    other, _ = seed_old_memories(postgres_pool, "kid@example.test")

    run(postgres_pool)

    for root in (roots / "host", roots / str(other)):
        assert store.list_memories(root=root)["memory_stores"] == ["gone", "thesis", "user"]
        profile = store.read_memory("user", "profile", root=root)
        assert profile["description"] == "Who the user is"
        assert [f["text"] for f in profile["facts"]] == ["Studies physics", "Lives in Berlin"]
        assert store.read_memory("user", "dev_environment", root=root)["description"] == "dev_environment"
        assert store.read_memory("thesis", "deadlines", root=root) | {"facts": []} == {"description": "Due dates", "facts": []}
        assert [f["text"] for f in store.read_memory("gone", "note", root=root)["facts"]] == ["Archived fact"]
    assert not (roots / str(host)).exists()


def test_project_memories_go_to_the_linked_folders_name(postgres_pool, roots):
    _, docs = seed_old_memories(postgres_pool, "host@example.test")
    linked = {"name": "Thesis", "kanban": [], "tabs": [], "linked_root": "/home/me/phd-thesis"}
    JsonDocs(docs).write("chat/thesis/project.json", linked)

    run(postgres_pool)

    assert store.list_memories(root=roots / "host")["memory_stores"] == ["gone", "phd-thesis", "user"]


def test_dry_run_writes_nothing_and_a_second_run_adds_nothing(postgres_pool, roots):
    seed_old_memories(postgres_pool, "host@example.test")

    planned = run(postgres_pool, dry_run=True, delete_old=True)
    assert "+ user/profile — Who the user is (2 of 2 facts)" in planned
    assert "- delete memory/global-profile.json" in planned
    assert not (roots / "host").exists()

    run(postgres_pool)
    again = run(postgres_pool)

    assert "= user/profile complete" in again
    assert "+ " not in again
    profile = store.read_memory("user", "profile", root=roots / "host")
    assert [f["text"] for f in profile["facts"]] == ["Studies physics", "Lives in Berlin"]


def test_delete_old_removes_the_old_documents_after_the_migration(postgres_pool, roots):
    _, docs = seed_old_memories(postgres_pool, "host@example.test")

    run(postgres_pool, delete_old=True)

    assert not docs.exists("memory")
    assert docs.exists("chat/thesis/project.json")
    assert store.read_memory("user", "profile", root=roots / "host")["facts"]


def test_delete_old_keeps_the_documents_when_a_project_cannot_be_placed(postgres_pool, roots):
    _, docs = seed_old_memories(postgres_pool, "host@example.test")
    linked = {"name": "Thesis", "kanban": [], "tabs": [], "linked_root": "/home/me/My Thesis"}
    JsonDocs(docs).write("chat/thesis/project.json", linked)

    output = run(postgres_pool, delete_old=True)

    assert "keep memory/project/thesis/memory/memory.json: 'My Thesis' is not a usable memory_store name, 1 memories left" in output
    assert docs.exists("memory/project/thesis/memory/memory.json")
    assert store.list_memories(root=roots / "host")["memory_stores"] == ["gone", "user"]


def test_unknown_host_login_is_refused(postgres_pool, roots):
    seed_old_memories(postgres_pool, "kid@example.test")

    with pytest.raises(SystemExit, match="no login"):
        run(postgres_pool)
    assert not any(roots.iterdir())


def test_an_existing_memory_gets_only_the_missing_facts_and_delete_old_keeps_nothing_unmigrated(postgres_pool, roots):
    _, docs = seed_old_memories(postgres_pool, "host@example.test")
    host = roots / "host"
    store.create_memory_store("user", root=host)
    store.apply_memory_ops("user", [{"op": "add", "path": "profile", "description": "From Pi", "text": "Lives in Berlin"}], root=host)

    output = run(postgres_pool, delete_old=True)

    assert "+ user/profile — Who the user is (1 of 2 facts)" in output
    profile = store.read_memory("user", "profile", root=host)
    assert profile["description"] == "From Pi"
    assert sorted(f["text"] for f in profile["facts"]) == ["Lives in Berlin", "Studies physics"]
    assert not docs.exists("memory")


def test_a_topic_ending_in_md_is_slugged_into_a_memory(postgres_pool, roots):
    _, docs = seed_old_memories(postgres_pool, "host@example.test")
    JsonDocs(docs).write("memory/project/thesis/memory/memory.json", [{"id": "5", "text": "Draft due in May", "topic": "notes.md"}])

    run(postgres_pool, delete_old=True)

    assert [f["text"] for f in store.read_memory("thesis", "notes-md", root=roots / "host")["facts"]] == ["Draft due in May"]
    assert not docs.exists("memory")


def test_delete_old_keeps_a_document_whose_target_cannot_be_read(postgres_pool, roots):
    _, docs = seed_old_memories(postgres_pool, "host@example.test")
    host = roots / "host"
    store.create_memory_store("user", root=host)
    (host / "user" / "profile.json").write_text("{not json", encoding="utf-8")

    output = run(postgres_pool, delete_old=True)
    again = run(postgres_pool, delete_old=True)

    assert "! user/profile: corrupt" in output
    assert "keep memory/global-profile.json: some facts are not in the new root" in again
    assert docs.exists("memory/global-profile.json") and docs.exists("memory/global-topics.json")
    assert not docs.exists("memory/project/thesis/memory/memory.json")
    assert [f["text"] for f in store.read_memory("thesis", "deadlines", root=host)["facts"]] == ["Draft due in May"]


def test_delete_old_keeps_documents_it_cannot_parse_or_does_not_know(postgres_pool, roots):
    _, docs = seed_old_memories(postgres_pool, "host@example.test")
    JsonDocs(docs).write("memory/global-profile.json", {"not": "a list"})
    JsonDocs(docs).write("memory/pending/review.json", {"scopes": []})

    output = run(postgres_pool, delete_old=True)

    assert "keep memory/global-profile.json: could not parse (not a list)" in output
    assert "keep memory/pending/review.json: not a memory document" in output
    assert docs.exists("memory/global-profile.json") and docs.exists("memory/pending/review.json")
    assert not docs.exists("memory/watermarks.json")
    assert not docs.exists("memory/project/thesis/memory/memory.json")


def test_a_project_named_user_migrates_into_project_user(postgres_pool, roots):
    _, docs = seed_old_memories(postgres_pool, "host@example.test")
    JsonDocs(docs).write("memory/project/user/memory/memory.json", [{"id": "7", "text": "User project fact"}])

    run(postgres_pool)

    assert [f["text"] for f in store.read_memory("project-user", "note", root=roots / "host")["facts"]] == ["User project fact"]
    assert "User project fact" not in store.read_memories("user", ["note"], root=roots / "host")
