"""Chat tree metadata stored beside each chat: written with the content, read without it, derived when missing."""

import json

import pytest

from lib.chat_store import ChatStore
from lib.postgres_data_store import PostgresDataStore


@pytest.fixture
def chat_store(data_store: PostgresDataStore) -> ChatStore:
    return ChatStore(data_store=data_store)


def pair(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}, {"role": "assistant", "content": f"re {text}"}]


def stored_meta(data_store: PostgresDataStore, filename: str) -> tuple[dict | None, int]:
    with data_store._scope.connection() as connection:
        return connection.execute(
            "SELECT meta, version FROM documents WHERE user_id = %s AND key = %s", (data_store._scope.user_id, f"chat/{filename}")
        ).fetchone()


def change_count(data_store: PostgresDataStore) -> int:
    with data_store._scope.connection() as connection:
        return connection.execute("SELECT count(*) FROM changes WHERE user_id = %s", (data_store._scope.user_id,)).fetchone()[0]


def write_without_meta(data_store: PostgresDataStore, filename: str, content: bytes) -> None:
    """Overwrite a chat the way code predating the meta column does: new content and version, meta untouched."""
    with data_store._scope.connection() as connection:
        connection.execute(
            "UPDATE documents SET content = %s, version = version + 1 WHERE user_id = %s AND key = %s",
            (content, data_store._scope.user_id, f"chat/{filename}"),
        )


def test_branch_meta_reads_stored_meta_without_chat_bodies(chat_store: ChatStore, data_store: PostgresDataStore, monkeypatch) -> None:
    chat_store.save("root.json", {"messages": pair("a") + pair("b"), "chat_id": "root"})
    branched = chat_store.branch("root.json", 0)
    child, child_id = branched["child_file"], branched["chat_id"]
    meta, version = stored_meta(data_store, "root.json")
    assert meta["version"] == version and meta["qa_count"] == 2

    def no_bodies(*_args, **_kwargs):
        raise AssertionError("branch-meta read chat bodies")

    monkeypatch.setattr(data_store, "read_many", no_bodies)
    monkeypatch.setattr(data_store, "read_bytes", no_bodies)
    assert ChatStore(data_store=data_store).get_branch_meta() == {
        "root.json": {
            "chat_id": "root",
            "parent_id": None,
            "branch_message_idx": None,
            "children": [{"chat_id": child_id, "branch_message_idx": 0}],
            "qa_count": 2,
        },
        child: {"chat_id": child_id, "parent_id": "root", "branch_message_idx": 0, "children": [], "qa_count": 1},
    }


def test_missing_meta_is_derived_and_stored_without_a_new_revision(chat_store: ChatStore, data_store: PostgresDataStore) -> None:
    body = {"messages": pair("a"), "chat_id": "legacy", "children": []}
    data_store.write_bytes("chat/legacy.json", json.dumps(body).encode())
    data_store.write_bytes("chat/list.json", json.dumps(pair("x")).encode())
    assert stored_meta(data_store, "legacy.json")[0] is None
    changes = change_count(data_store)

    meta = chat_store.get_branch_meta()
    assert meta["legacy.json"] == {"chat_id": "legacy", "parent_id": None, "branch_message_idx": None, "children": [], "qa_count": 1}
    assert meta["list.json"] == {"chat_id": None, "parent_id": None, "branch_message_idx": None, "children": [], "qa_count": 1}

    filled, version = stored_meta(data_store, "legacy.json")
    assert (filled["chat_id"], filled["version"], version) == ("legacy", 1, 1)
    assert change_count(data_store) == changes


def test_meta_left_behind_by_a_writer_that_ignores_it_is_not_used(chat_store: ChatStore, data_store: PostgresDataStore) -> None:
    chat_store.save("chat.json", {"messages": pair("a"), "chat_id": "chat"})
    write_without_meta(data_store, "chat.json", json.dumps({"messages": pair("a") + pair("b"), "chat_id": "renamed"}).encode())

    assert chat_store.get_branch_meta()["chat.json"]["chat_id"] == "renamed"
    assert chat_store.get_branch_meta()["chat.json"]["qa_count"] == 2
    meta, version = stored_meta(data_store, "chat.json")
    assert (meta["chat_id"], meta["version"]) == ("renamed", version)


def test_meta_jsonb_cannot_store_exactly_falls_back_to_the_body(chat_store: ChatStore, data_store: PostgresDataStore) -> None:
    # NUL is rejected by jsonb, unsorted child keys and floats would come back changed.
    odd_child = {"chat_id": "gone", "branch_message_idx": 0, "a": 1.5}
    chat_store.save("nul.json", {"messages": pair("a"), "chat_id": "nul", "parent_id": "p\x00"})
    chat_store.save("odd.json", {"messages": pair("a"), "chat_id": "odd", "children": [odd_child], "title": "x"})
    chat_store.save("kept.json", {"messages": pair("a"), "chat_id": "gone"})

    for _ in range(2):
        meta = ChatStore(data_store=data_store).get_branch_meta()
        assert meta["nul.json"]["parent_id"] == "p\x00"
        assert json.dumps(meta["odd.json"]["children"]) == json.dumps([odd_child])
    assert stored_meta(data_store, "nul.json")[0] is None
    assert stored_meta(data_store, "odd.json")[0] is None


def test_unreadable_chats_report_empty_meta(chat_store: ChatStore, data_store: PostgresDataStore) -> None:
    data_store.write_bytes("chat/broken.json", b"{not json")
    data_store.write_bytes("chat/scalar.json", b'"text"')
    data_store.write_bytes("chat/bad-messages.json", json.dumps({"messages": ["x"], "chat_id": "bad"}).encode())

    meta = chat_store.get_branch_meta()
    assert meta["broken.json"] == {} and meta["scalar.json"] == {}
    assert meta["bad-messages.json"]["chat_id"] == "bad" and meta["bad-messages.json"]["qa_count"] is None
    assert stored_meta(data_store, "broken.json")[0] is None


def test_dead_children_are_pruned_from_meta_and_body(chat_store: ChatStore, data_store: PostgresDataStore) -> None:
    chat_store.save("root.json", {"messages": pair("a"), "chat_id": "root"})
    child = chat_store.branch("root.json", 0)["child_file"]
    data_store.delete(f"chat/{child}")

    assert chat_store.get_branch_meta()["root.json"]["children"] == []
    assert chat_store.load("root.json")["children"] == []
    meta, version = stored_meta(data_store, "root.json")
    assert (meta["children"], meta["version"]) == ([], version)


def test_index_and_project_cleanup_use_meta(chat_store: ChatStore, data_store: PostgresDataStore) -> None:
    data_store.write_bytes("chat/proj/project.json", json.dumps({"name": "proj", "kanban": [], "tabs": []}).encode())
    data_store.write_bytes("chat/proj/one.json", json.dumps({"messages": [], "chat_id": "one"}).encode())
    chat_store.save("proj/two.json", {"messages": pair("a"), "chat_id": "two"})

    assert sorted(chat_store.chat_ids_in("proj")) == ["one", "two"]
    assert chat_store._find("one") == "chat/proj/one.json"
    assert stored_meta(data_store, "proj/project.json")[0] is None
