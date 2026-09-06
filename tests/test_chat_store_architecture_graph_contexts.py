from pathlib import Path

from lib.chat_store import ChatStore


CONTEXTS = [{"path": "/documents/Architecture_Graphs/checkout.architecture.yaml"}]


def test_architecture_graph_contexts_round_trip_and_survive_ordinary_save(tmp_path: Path) -> None:
    store = ChatStore(tmp_path)
    store.save("chat.json", {"messages": [], "chat_id": "chat-1", "architecture_graph_contexts": CONTEXTS})

    assert store.load("chat.json")["architecture_graph_contexts"] == CONTEXTS

    # Existing callers that have no graph-context UI yet must not erase it.
    store.save("chat.json", {"messages": [{"role": "user", "content": "Hello"}], "chat_id": "chat-1"})

    assert store.load("chat.json")["architecture_graph_contexts"] == CONTEXTS


def test_architecture_graph_contexts_follow_branch_and_move(tmp_path: Path) -> None:
    store = ChatStore(tmp_path)
    store.save(
        "chat.json",
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_id": "chat-1",
            "architecture_graph_contexts": CONTEXTS,
        },
    )

    branch = store.branch("chat.json", 0)
    assert store.load(branch["child_file"])["architecture_graph_contexts"] == CONTEXTS

    moved = store.move("chat.json", "archive")
    assert store.load(moved["new_path"])["architecture_graph_contexts"] == CONTEXTS
