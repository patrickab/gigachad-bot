"""LLM memory extraction and regrouping, proposed for user review (no vector DB).

1. ``extract``: one JSON call per scope proposes operations on the stored facts (add, update,
   delete) plus summaries for new or outgrown topics. The per-chat watermark target is buffered
   in the store's pending buffer.
2. user review: the frontend accepts, edits or denies each operation and topic.
3. ``MemoryStore.commit``: accepted operations apply deterministically, then the scope's watermark advances.

``propose_regroup`` reorganizes a scope's facts into subject topics for user review.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
import logging
from typing import Any, Protocol
import uuid

from lib.llm_json import extract_json_from_llm
from lib.llm_resilience import completion_text
from lib.memory_store import (
    FALLBACK_TOPIC,
    MemoryOperation,
    MemoryScope,
    MemoryStore,
    ScopeKind,
    StoredMemory,
    add_missing_topics,
    topic_name,
)

log = logging.getLogger(__name__)


class MemoryLLM(Protocol):
    def api_query(self, **kwargs: object) -> object: ...


EXTRACT_TIMEOUT = 90

# Per-scope framing fed into the extraction prompt.
SCOPE_GUIDANCE: dict[ScopeKind, str] = {
    "global": "durable facts about the user, valid across all projects",
    "project": "durable facts about this project or subject",
}
# Topic names the regroup prompt must keep verbatim, per scope.
REGROUP_PINNED: dict[ScopeKind, str] = {
    "global": (
        "Two topics have fixed names, use them verbatim: `profile` (who the user is) and `preferences` "
        "(how the user wants the assistant to respond and work). "
    ),
    "project": "",
}

OPERATIONS = ("add", "update", "delete")
EXTRACT_RESPONSE_SHAPE = json.dumps({
    "operations": [{
        "op": "add|update|delete",
        "id": "<recorded id, update and delete only>",
        "topic": "<topic name, add and update only>",
        "text": "<fact, add and update only>",
    }],
    "topics": [{"name": "...", "summary": "..."}],
})


@dataclass
class TopicProposal:
    """New topic (`previous_summary` None) or a revised summary of an existing one."""

    scope: str
    name: str
    summary: str
    previous_summary: str | None = None


@dataclass
class ScopeProposal:
    operations: list[MemoryOperation]
    topics: list[TopicProposal]

    def is_empty(self) -> bool:
        return not self.operations and not self.topics


@dataclass
class ExtractResult:
    review_id: str
    global_proposal: ScopeProposal | None
    project_proposal: ScopeProposal | None


def _dicts(raw: object) -> list[dict[str, Any]]:
    """The dict entries of *raw* when it is a list, else nothing."""
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _short_ids(memories: list[StoredMemory]) -> dict[str, StoredMemory]:
    """Short ids keep prompts small and give the model fewer chances to garble a uuid."""
    return {f"m{i + 1}": m for i, m in enumerate(memories)}


def _facts_block(short_ids: dict[str, StoredMemory]) -> str:
    return "\n".join(f"{sid} [{m.topic}] {m.text}" for sid, m in short_ids.items())


def _transcript(messages: list[dict[str, str]]) -> str:
    blocks = [f"{m.get('role', 'user').upper()}: {m.get('content', '').strip()}" for m in messages]
    return "\n\n".join(blocks) or "No conversation content."


def build_proposal(
    kind: str, data: dict[str, Any], topics: list[dict[str, str]], short_ids: dict[str, StoredMemory]
) -> ScopeProposal:
    """Validate model output: drop unknown ids, duplicates and no-ops, give every new topic a proposal."""
    known_texts = {m.text.strip().lower() for m in short_ids.values()}
    targeted: set[str] = set()
    operations: list[MemoryOperation] = []
    for item in _dicts(data.get("operations")):
        if item.get("op") not in OPERATIONS:
            continue
        op = item["op"]
        text = str(item.get("text") or "").strip()
        topic = topic_name(item.get("topic"))
        if op == "add":
            if not text or text.lower() in known_texts:
                continue
            known_texts.add(text.lower())
            operations.append(MemoryOperation(str(uuid.uuid4()), op, kind, topic or FALLBACK_TOPIC, text))
            continue
        target = short_ids.get(str(item.get("id") or "").strip())
        if not target or target.id in targeted:
            continue
        if op == "update" and (not text or (text == target.text and topic in ("", target.topic))):
            continue
        targeted.add(target.id)
        operations.append(MemoryOperation(
            str(uuid.uuid4()), op, kind, (topic if op == "update" and topic else target.topic),
            text if op == "update" else "", target.id, target.text,
        ))

    summary_by_name = {t["name"]: t["summary"] for t in topics}
    used_topics = {o.topic for o in operations if o.op != "delete"}
    proposals: dict[str, TopicProposal] = {}
    for item in _dicts(data.get("topics")):
        name = topic_name(item.get("name"))
        summary = str(item.get("summary") or "").strip()
        if not name or name in proposals:
            continue
        if name in summary_by_name:
            if summary and summary != summary_by_name[name]:
                proposals[name] = TopicProposal(kind, name, summary, summary_by_name[name])
        elif name in used_topics:
            proposals[name] = TopicProposal(kind, name, summary)
    for name in sorted(used_topics - summary_by_name.keys() - proposals.keys()):
        proposals[name] = TopicProposal(kind, name, "")
    return ScopeProposal(operations, list(proposals.values()))


@dataclass(frozen=True)
class _ScopeInput:
    """What one scope's extraction prompt is built from."""

    scope: MemoryScope
    transcript: str
    topics: list[dict[str, str]]
    short_ids: dict[str, StoredMemory]


@dataclass
class MemoryExtractor:
    """Runs the memory model over a store's facts. *model* is the user's stored default, nothing here names one."""

    store: MemoryStore
    llm: MemoryLLM
    model: str

    async def extract(
        self,
        messages: list[dict[str, str]],
        project_slug: str | None = None,
        chat_id: str | None = None,
    ) -> ExtractResult:
        """Propose memory operations from messages since this conversation's last memorization.

        Extracts the global scope and, with *project_slug*, the project scope in parallel. Watermarks
        are tracked per ``(chat_id, scope)`` and advance only when that scope commits, so cancelling
        leaves the messages eligible next time.
        """
        review_id = str(uuid.uuid4())
        kinds: list[ScopeKind] = ["global", "project"] if project_slug else ["global"]

        def load() -> dict[ScopeKind, _ScopeInput | None]:
            """Each scope's new messages with its topics and facts, None when nothing is new."""
            self.store.ensure_pending_dir()
            loaded: dict[ScopeKind, _ScopeInput | None] = {}
            for kind in kinds:
                tail = messages[self.store.watermark(chat_id, kind):]
                memory_scope = MemoryScope.of(kind, project_slug)
                loaded[kind] = _ScopeInput(
                    memory_scope, _transcript(tail), self.store.get_topics(memory_scope),
                    _short_ids(self.store.read_memories(memory_scope)),
                ) if tail else None
            return loaded

        # Store reads and writes block on Postgres, so each block runs off the event loop.
        inputs = await asyncio.to_thread(load)

        async def run(kind: ScopeKind) -> ScopeProposal | None:
            scope_input = inputs[kind]
            return await self._extract_scope(scope_input) if scope_input else None

        results = dict(zip(kinds, await asyncio.gather(*(run(k) for k in kinds)), strict=True))
        committable = [k for k, proposal in results.items() if proposal and not proposal.is_empty()]
        if committable:
            await asyncio.to_thread(self.store.write_pending, review_id, chat_id, len(messages), committable)
        return ExtractResult(review_id, results.get("global"), results.get("project"))

    async def _extract_scope(self, scope_input: _ScopeInput) -> ScopeProposal:
        scope, transcript, topics, short_ids = (
            scope_input.scope, scope_input.transcript, scope_input.topics, scope_input.short_ids,
        )
        system_prompt = (
            f"You maintain a user's long-term memory of {SCOPE_GUIDANCE[scope.kind]}. Memory is organized in topics: "
            "named subjects with a one-line summary, each holding atomic facts. Compare the conversation "
            "with the memory on record and output only changes, as valid JSON without prose.\n\n"
            "Operations:\n"
            "- add: a durable fact not yet on record.\n"
            "- update: a recorded fact the conversation refines or contradicts. Give its id and the full corrected text.\n"
            "- delete: a recorded fact the conversation shows is no longer true. Give its id.\n\n"
            "Rules:\n"
            "- Never re-output a recorded fact unchanged.\n"
            "- One fact per operation, maximal information in minimal words.\n"
            "- Skip session-transient details such as current errors or one-off requests.\n"
            "- File each fact under the recorded topic whose subject fits. Create a new topic only for a "
            "clearly distinct subject, named in 1-3 words.\n"
            "- In \"topics\", give a one-line summary for every new topic. A summary names what the topic covers, "
            "e.g. \"The user's Linux setup and hardware\", and never repeats a fact. Revise a recorded topic's "
            "summary only when your changes make it wrong or incomplete.\n"
            "- Empty lists are valid."
        )
        topics_block = "\n".join(f"- {t['name']}: {t['summary']}" for t in topics) or "(none yet)"
        facts_block = _facts_block(short_ids) or "(none yet)"
        user_prompt = f"""
Conversation:
{transcript}

Topics on record (name: summary):
{topics_block}

Facts on record (id [topic] text):
{facts_block}

Return JSON with this exact shape:
{EXTRACT_RESPONSE_SHAPE}
""".strip()

        data = await self._generate_json(system_prompt, user_prompt)
        return build_proposal(scope.kind, data, topics, short_ids)

    async def propose_regroup(self, scope: MemoryScope) -> dict[str, Any]:
        """Propose subject topics for every fact in *scope*, returned as `{topics, memories}` without writing."""

        def load() -> tuple[list[StoredMemory], list[dict[str, str]] | None]:
            memories = self.store.read_memories(scope)
            return memories, None if memories else self.store.get_topics(scope)

        # Store reads block on Postgres, so they run off the event loop.
        memories, topics_on_record = await asyncio.to_thread(load)
        if not memories:
            return {"topics": topics_on_record, "memories": []}
        short_ids = _short_ids(memories)
        system_prompt = (
            "You reorganize a user's memory facts into topics: named subjects (1-3 words) with a one-line "
            "summary naming what the topic covers, e.g. \"The user's Linux setup and hardware\", never a fact. "
            f"{REGROUP_PINNED[scope.kind]}Group facts by subject, not by kind of fact. Prefer few, well-filled topics over many "
            "small ones. Assign every fact to exactly one topic. Output valid JSON without prose."
        )
        facts_block = _facts_block(short_ids)
        user_prompt = f"""
Facts (id [current topic] text):
{facts_block}

Return JSON with this exact shape:
{{"topics": [{{"name": "...", "summary": "..."}}], "assignments": [{{"id": "<fact id>", "topic": "<topic name>"}}]}}
""".strip()
        data = await self._generate_json(system_prompt, user_prompt)

        topics: list[dict[str, str]] = []
        for item in _dicts(data.get("topics")):
            name = topic_name(item.get("name"))
            if name and all(t["name"] != name for t in topics):
                topics.append({"name": name, "summary": str(item.get("summary") or "").strip()})
        assignments: dict[str, str] = {}
        for item in _dicts(data.get("assignments")):
            if target := short_ids.get(str(item.get("id") or "").strip()):
                assignments[target.id] = topic_name(item.get("topic")) or target.topic

        # Unassigned facts keep their topic, which keeps its summary.
        regrouped = [{**asdict(m), "topic": assignments.get(m.id, m.topic)} for m in memories]
        previous_summaries = {t["name"]: t["summary"] for t in await asyncio.to_thread(self.store.get_topics, scope)}
        add_missing_topics(topics, (m["topic"] for m in regrouped), previous_summaries)
        used = {m["topic"] for m in regrouped}
        return {"topics": [t for t in topics if t["name"] in used], "memories": regrouped}

    async def _generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Call the memory model and parse JSON, retrying once without response_format.

        Small models occasionally reject ``response_format`` or wrap JSON in prose.
        """

        def call(force_json: bool) -> str:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "user_msg": user_prompt,
                "user_msg_history": [],
                "system_prompt": system_prompt,
                "stream": False,
            }
            if force_json:
                kwargs["response_format"] = {"type": "json_object"}
            return completion_text(self.llm, **kwargs) or "{}"

        last_err: Exception | None = None
        for force_json in (True, False):
            try:
                text = await asyncio.wait_for(asyncio.to_thread(call, force_json), timeout=EXTRACT_TIMEOUT)
                return extract_json_from_llm(text)
            except Exception as e:  # noqa: BLE001 - retry both transport and parse failures
                last_err = e
                log.warning("Memory JSON call failed (response_format=%s): %s", force_json, e)

        raise RuntimeError(f"Memory model did not return valid JSON: {last_err}") from last_err
