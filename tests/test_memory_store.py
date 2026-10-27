import asyncio
import json
from types import SimpleNamespace

import pytest

from lib import memory_context, memory_store
from lib.memory_extract import MemoryExtractor
from lib.memory_store import MemoryOperation, MemoryScope, MemoryStore

GLOBAL = MemoryScope("global")


@pytest.fixture
def store(data_store) -> MemoryStore:
    return MemoryStore(data_store)


def extract(store: MemoryStore, llm: object, messages: list[dict[str, str]]):
    return asyncio.run(MemoryExtractor(store, llm, "test").extract(messages, chat_id="c1", scope="global"))


class ScriptedLLM:
    """Answer every memory call with the next scripted JSON payload."""

    def __init__(self, *payloads: dict) -> None:
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def api_query(self, **kwargs: object) -> object:
        self.prompts.append(str(kwargs["user_msg"]))
        content = json.dumps(self.payloads.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def seed(store: MemoryStore) -> None:
    store.write_memories(GLOBAL, [
        {"id": "a", "text": "Uses Arch Linux", "topic": "profile"},
        {"id": "b", "text": "Prefers terse answers", "topic": "preferences"},
    ])


def test_extract_keeps_only_valid_changes_and_proposes_new_topics(store):
    seed(store)
    llm = ScriptedLLM({
        "operations": [
            {"op": "update", "id": "m1", "text": "Uses Debian with GNOME"},
            {"op": "delete", "id": "m1"},  # second change to the same fact
            {"op": "delete", "id": "m9"},  # unknown fact
            {"op": "add", "topic": "profile", "text": "prefers terse answers"},  # already on record
            {"op": "update", "id": "m2", "text": "Prefers terse answers"},  # no change
            {"op": "add", "topic": "Dev Environment", "text": "Runs iwd for networking"},
            {"op": "explode", "id": "m2"},
        ],
        "topics": [
            {"name": "preferences", "summary": "Response style and tooling choices."},
            {"name": "unused_topic", "summary": "Nothing filed here."},
        ],
    })

    result = extract(store, llm, [{"role": "user", "content": "hi"}])

    proposal = result.global_proposal
    assert [(o.op, o.target_id, o.topic, o.text) for o in proposal.operations] == [
        ("update", "a", "profile", "Uses Debian with GNOME"),
        ("add", None, "dev_environment", "Runs iwd for networking"),
    ]
    assert proposal.operations[0].target_text == "Uses Arch Linux"
    topics = {t.name: (t.summary, t.previous_summary) for t in proposal.topics}
    assert topics == {
        "preferences": ("Response style and tooling choices.", memory_store.DEFAULT_GLOBAL_TOPICS[1]["summary"]),
        "dev_environment": ("", None),
    }
    assert result.project_proposal is None


def test_commit_applies_operations_and_advances_only_the_committed_scope(store):
    seed(store)
    llm = ScriptedLLM({"operations": [{"op": "delete", "id": "m2"}], "topics": []})
    messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    review = extract(store, llm, messages)

    store.commit(GLOBAL, [
        MemoryOperation("1", "add", "global", "Dev Environment", "Runs iwd"),
        MemoryOperation("2", "update", "global", "profile", "Uses Debian", target_id="a"),
        MemoryOperation("3", "delete", "global", "preferences", target_id="b"),
        MemoryOperation("4", "update", "global", "profile", "Ghost", target_id="gone"),
    ], [{"name": "dev_environment", "summary": "Linux setup"}], review_id=review.review_id)

    assert sorted((m["topic"], m["text"]) for m in store.list_memories(GLOBAL)) == [
        ("dev_environment", "Runs iwd"), ("profile", "Uses Debian"),
    ]
    assert {"name": "dev_environment", "summary": "Linux setup"} in store.get_topics(GLOBAL)
    assert store.watermark("c1", "global") == 2
    assert store.watermark("c1", "project") == 0


def test_cancel_leaves_messages_eligible(store):
    seed(store)
    llm = ScriptedLLM({"operations": [{"op": "delete", "id": "m2"}], "topics": []})
    review = extract(store, llm, [{"role": "user", "content": "x"}])
    store.cancel(review.review_id)
    assert store.watermark("c1", "global") == 0


def test_prompt_keeps_pinned_topics_and_indexes_overflow(store, monkeypatch):
    monkeypatch.setattr(memory_context, "PROMPT_CHAR_BUDGET", 120)
    store.write_memories(GLOBAL, [
        {"id": "a", "text": "Old preference " + "x" * 200, "topic": "preferences", "updated_at": "2020-01-01"},
        {"id": "b", "text": "Recent hobby", "topic": "gardening", "updated_at": "2026-01-02"},
        {"id": "c", "text": "Older hobby " + "y" * 100, "topic": "chess", "updated_at": "2026-01-01"},
    ])
    store.set_topics(GLOBAL, [
        {"name": "preferences", "summary": "How to respond"},
        {"name": "chess", "summary": "Chess openings"},
        {"name": "gardening", "summary": "Pond plants"},
    ])

    prompt = memory_context.augment_system_prompt(store, "Base")

    assert "Old preference" in prompt  # pinned beats the budget
    assert "- Recent hobby" in prompt  # newest unpinned topic fills the remaining budget
    assert "Older hobby" not in prompt
    assert "- Chess: Chess openings" in prompt
