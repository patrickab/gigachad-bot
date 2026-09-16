from pathlib import Path

from lib.chat_store import ChatStore


def test_save_with_empty_messages_does_not_erase_existing_messages(tmp_path: Path) -> None:
    """Regression test: a save payload missing/empty `messages` (e.g. from a stale
    client-side closure racing a history load) must not wipe stored conversation
    content, matching the fallback-to-existing behavior every other field already has.
    """
    store = ChatStore(tmp_path)
    real_messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]
    store.save("chat.json", {"messages": real_messages, "chat_id": "chat-1", "usage": {"total_tokens": 44800}})

    assert store.load("chat.json")["messages"] == real_messages

    # A save arriving with empty messages but no usage (usage falls back to existing,
    # as observed in the bug report: tokens survive, messages vanish) must preserve messages.
    store.save("chat.json", {"messages": [], "chat_id": "chat-1"})

    loaded = store.load("chat.json")
    assert loaded["messages"] == real_messages
    assert loaded["usage"] == {"total_tokens": 44800}


def test_save_with_no_data_still_preserves_existing_messages(tmp_path: Path) -> None:
    store = ChatStore(tmp_path)
    real_messages = [{"role": "user", "content": "Hello"}]
    store.save("chat.json", {"messages": real_messages, "chat_id": "chat-1"})

    store.save("chat.json", {"chat_id": "chat-1"})

    assert store.load("chat.json")["messages"] == real_messages


def test_save_on_new_file_with_no_messages_is_empty(tmp_path: Path) -> None:
    store = ChatStore(tmp_path)
    store.save("new.json", {"chat_id": "chat-2"})

    assert store.load("new.json")["messages"] == []
