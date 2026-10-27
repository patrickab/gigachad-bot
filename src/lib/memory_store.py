"""Document-backed memory: atomic facts grouped into topics, per global profile and project.

Pipeline (no vector DB):

1. ``extract``: one JSON call per scope proposes operations on the stored facts (add, update,
   delete) plus summaries for new or outgrown topics. The per-chat watermark target is buffered
   to ``memory/pending/<review_id>.json``.
2. user review: the frontend accepts, edits or denies each operation and topic.
3. ``commit``: accepted operations apply deterministically, then the scope's watermark advances.

``propose_regroup`` reorganizes a scope's facts into subject topics for user review.
``augment_system_prompt`` injects the stored facts into chats within a character budget.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Protocol
import uuid

from lib.clock import utc_now_iso
from lib.data_store import DataStore, DataStorePath
from lib.json_io import safe_write_json
from lib.llm_json import extract_json_from_llm
from lib.llm_resilience import completion_text
from lib.storage_namespace import MEMORY

log = logging.getLogger(__name__)


class MemoryLLM(Protocol):
    def api_query(self, **kwargs: object) -> object: ...


EXTRACT_TIMEOUT = 90
FALLBACK_TOPIC = "note"
# Facts beyond this many prompt characters are injected as topic summaries only.
PROMPT_CHAR_BUDGET = 16_000

# Per-scope framing fed into the extraction prompt.
GLOBAL_SCOPE_GUIDANCE = "durable facts about the user, valid across all projects"
PROJECT_SCOPE_GUIDANCE = "durable facts about this project or subject"

# Global topics injected in full regardless of the prompt budget.
PINNED_GLOBAL_TOPICS = ("profile", "preferences")

DEFAULT_GLOBAL_TOPICS: list[dict[str, str]] = [
    {"name": "profile", "summary": "Who the user is: background, education, work, and setup."},
    {"name": "preferences", "summary": "How the user wants the assistant to respond and work."},
]
# Project topics emerge from extraction.
DEFAULT_PROJECT_TOPICS: list[dict[str, str]] = []

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
class StoredMemory:
    id: str
    text: str
    topic: str
    scope: str  # "global" | "project"
    created_at: str
    updated_at: str

    @classmethod
    def from_raw(cls, scope: str, raw: dict[str, Any], now: str, *, normalize: bool = False) -> StoredMemory:
        """Memory from a stored or user-supplied dict. *normalize* also strips the text and canonicalizes the topic name."""
        if normalize:
            text, topic = str(raw.get("text") or "").strip(), topic_name(raw.get("topic")) or FALLBACK_TOPIC
        else:
            text, topic = str(raw.get("text", "")), str(raw.get("topic") or FALLBACK_TOPIC)
        return cls(
            id=str(raw.get("id") or uuid.uuid4()),
            text=text,
            topic=topic,
            scope=scope,
            created_at=str(raw.get("created_at") or now),
            updated_at=str(raw.get("updated_at") or now),
        )


@dataclass
class MemoryOperation:
    """Proposed change to one fact, `target_*` name the stored fact an update or delete replaces."""

    id: str
    op: str  # "add" | "update" | "delete"
    scope: str
    topic: str
    text: str = ""
    target_id: str | None = None
    target_text: str | None = None


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

@dataclass
class _TopicBlock:
    """One topic's rendered prompt section, `included` once it fits the budget."""

    scope: str
    name: str
    summary: str
    text: str
    order: int
    pinned: bool
    updated_at: str
    included: bool = False



def topic_name(raw: object) -> str:
    """Stored topic name of a model- or user-written title: `Dev Environment` -> `dev_environment`."""
    return re.sub(r"[^\w]+", "_", str(raw or "").strip().lower()).strip("_")


def topic_title(name: str) -> str:
    return name.replace("_", " ").title()


def _dicts(raw: object) -> list[dict[str, Any]]:
    """The dict entries of *raw* when it is a list, else nothing."""
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _short_ids(memories: list[StoredMemory]) -> dict[str, StoredMemory]:
    """Short ids keep prompts small and give the model fewer chances to garble a uuid."""
    return {f"m{i + 1}": m for i, m in enumerate(memories)}


def _facts_block(short_ids: dict[str, StoredMemory]) -> str:
    return "\n".join(f"{sid} [{m.topic}] {m.text}" for sid, m in short_ids.items())


class MemoryStore:
    """Owns memory extraction, canonical memory documents, and the pending buffer."""

    def __init__(self, prefix: str = MEMORY, *, data_store: DataStore, model: str) -> None:
        # The user's stored default is authoritative; nothing here may name a model.
        self.model = model
        self.base_dir = DataStorePath(data_store, prefix)
        self.pending_dir = self.base_dir / "pending"

    # ------------------------------------------------------------------
    # Chat injection
    # ------------------------------------------------------------------

    def augment_system_prompt(self, system_prompt: str, project_slug: str | None = None) -> str:
        """Append global and project memory to *system_prompt* under a "Persistent Memory Context" heading."""
        memory_context = self._chat_context(project_slug)
        if not memory_context:
            return system_prompt
        if not system_prompt.strip():
            return memory_context
        return f"{system_prompt.rstrip()}\n\n# Persistent Memory Context\n\n{memory_context}"

    def _chat_context(self, project_slug: str | None) -> str:
        """Render memory per topic within `PROMPT_CHAR_BUDGET`.

        - Pinned global topics always render in full, outside the budget
        - The most recently updated other topics fill the budget
        - Topics past the budget appear as a summary index, so the model still knows they exist
        """
        scopes = [("global", None, "About the user")]
        if project_slug:
            scopes.append(("project", project_slug, "Project memory"))

        blocks: list[_TopicBlock] = []
        for scope, slug, _ in scopes:
            topics = self.get_topics(scope, slug)
            order = {t["name"]: i for i, t in enumerate(topics)}
            summary_by_name = {t["name"]: t["summary"] for t in topics}
            by_topic: dict[str, list[StoredMemory]] = {}
            for memory in self._read_memories(scope, slug):
                by_topic.setdefault(memory.topic, []).append(memory)
            for name, memories in by_topic.items():
                summary = summary_by_name.get(name, "")
                lines = [f"### {topic_title(name)}", *([summary] if summary else []), *(f"- {m.text}" for m in memories)]
                blocks.append(_TopicBlock(
                    scope=scope, name=name, summary=summary, text="\n".join(lines),
                    order=order.get(name, len(order)),
                    pinned=scope == "global" and name in PINNED_GLOBAL_TOPICS,
                    updated_at=max(m.updated_at for m in memories),
                ))

        used = 0
        for block in sorted(blocks, key=lambda b: b.updated_at, reverse=True):
            if block.pinned:
                block.included = True
            elif used + len(block.text) <= PROMPT_CHAR_BUDGET:
                block.included = True
                used += len(block.text)

        parts: list[str] = []
        for scope, _, heading in scopes:
            scope_blocks = sorted((b for b in blocks if b.scope == scope), key=lambda b: b.order)
            sections = [b.text for b in scope_blocks if b.included]
            index = [f"- {topic_title(b.name)}: {b.summary or 'no summary'}" for b in scope_blocks if not b.included]
            if index:
                sections.append("### Further topics (summaries only)\n" + "\n".join(index))
            if sections:
                parts.append(f"## {heading}\n\n" + "\n\n".join(sections))
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    async def extract(
        self,
        llm: MemoryLLM,
        messages: list[dict[str, str]],
        project_slug: str | None = None,
        chat_id: str | None = None,
        scope: str | None = None,
    ) -> ExtractResult:
        """Propose memory operations from messages since this conversation's last memorization.

        *scope* is ``"global"``, ``"project"``, or ``None`` for both, extracted in parallel. Watermarks
        are tracked per ``(chat_id, scope)`` and advance only when that scope commits, so cancelling
        leaves the messages eligible next time.
        """
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        review_id = str(uuid.uuid4())
        targets = [s for s in ("global", "project") if scope in (None, s) and (s == "global" or project_slug)]

        async def run(scope_name: str) -> ScopeProposal | None:
            slug = project_slug if scope_name == "project" else None
            start = self._watermark_for(chat_id, scope_name)
            tail = messages[start:]
            if not tail:
                return None
            return await self._extract_scope(llm, scope_name, slug, self._get_transcript(tail))

        results = dict(zip(targets, await asyncio.gather(*(run(s) for s in targets)), strict=True))
        committable = [s for s, proposal in results.items() if proposal and not proposal.is_empty()]
        if committable:
            self._write_pending(review_id, chat_id, len(messages), committable)
        return ExtractResult(review_id, results.get("global"), results.get("project"))

    async def _extract_scope(self, llm: MemoryLLM, scope: str, project_slug: str | None, transcript: str) -> ScopeProposal:
        topics = self.get_topics(scope, project_slug)
        memories = self._read_memories(scope, project_slug)
        # Short ids keep the prompt small and give the model fewer chances to garble a uuid.
        short_ids = _short_ids(memories)
        guidance = GLOBAL_SCOPE_GUIDANCE if scope == "global" else PROJECT_SCOPE_GUIDANCE

        system_prompt = (
            f"You maintain a user's long-term memory of {guidance}. Memory is organized in topics: "
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

        data = await self._generate_json(llm, system_prompt, user_prompt)
        return self._build_proposal(scope, data, topics, short_ids)

    @staticmethod
    def _build_proposal(
        scope: str, data: dict[str, Any], topics: list[dict[str, str]], short_ids: dict[str, StoredMemory]
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
                operations.append(MemoryOperation(str(uuid.uuid4()), op, scope, topic or FALLBACK_TOPIC, text))
                continue
            target = short_ids.get(str(item.get("id") or "").strip())
            if not target or target.id in targeted:
                continue
            if op == "update" and (not text or (text == target.text and topic in ("", target.topic))):
                continue
            targeted.add(target.id)
            operations.append(MemoryOperation(
                str(uuid.uuid4()), op, scope, (topic if op == "update" and topic else target.topic),
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
                    proposals[name] = TopicProposal(scope, name, summary, summary_by_name[name])
            elif name in used_topics:
                proposals[name] = TopicProposal(scope, name, summary)
        for name in sorted(used_topics - summary_by_name.keys() - proposals.keys()):
            proposals[name] = TopicProposal(scope, name, "")
        return ScopeProposal(operations, list(proposals.values()))

    # ------------------------------------------------------------------
    # Commit / cancel
    # ------------------------------------------------------------------

    def commit(
        self,
        scope: str,
        operations: list[MemoryOperation],
        topics: list[dict[str, str]],
        project_slug: str | None = None,
        review_id: str | None = None,
    ) -> None:
        """Apply accepted operations and topic summaries to *scope*, then advance the review's watermark.

        Operations on facts that no longer exist are skipped. Empty *operations* still advance the
        watermark, so denied proposals are not offered again.
        """
        memories = self._read_memories(scope, project_slug)
        by_id = {m.id: m for m in memories}
        now = utc_now_iso()
        for operation in operations:
            text = operation.text.strip()
            topic = topic_name(operation.topic) or FALLBACK_TOPIC
            if operation.op == "add" and text:
                memories.append(StoredMemory(str(uuid.uuid4()), text, topic, scope, now, now))
            elif operation.op == "update" and text and operation.target_id in by_id:
                target = by_id[operation.target_id]
                target.text, target.topic, target.updated_at = text, topic, now
            elif operation.op == "delete" and operation.target_id in by_id:
                memories.remove(by_id.pop(operation.target_id))

        stored_topics = self.get_topics(scope, project_slug)
        for proposal in topics:
            name = topic_name(proposal.get("name"))
            if not name:
                continue
            summary = str(proposal.get("summary") or "").strip()
            existing = next((t for t in stored_topics if t["name"] == name), None)
            if existing:
                existing["summary"] = summary or existing["summary"]
            else:
                stored_topics.append({"name": name, "summary": summary})
        known = {t["name"] for t in stored_topics}
        for memory in memories:
            if memory.topic not in known:
                stored_topics.append({"name": memory.topic, "summary": ""})
                known.add(memory.topic)

        self.set_topics(scope, stored_topics, project_slug)
        self.write_memories(scope, [vars(m) for m in memories], project_slug)
        if review_id:
            self._consume_pending_scope(review_id, scope)

    def cancel(self, review_id: str) -> None:
        """Discard the outstanding review without advancing any watermark."""
        self._delete_pending(review_id)

    # ------------------------------------------------------------------
    # Regroup
    # ------------------------------------------------------------------

    async def propose_regroup(self, llm: MemoryLLM, scope: str, project_slug: str | None = None) -> dict[str, Any]:
        """Propose subject topics for every fact in *scope*, returned as `{topics, memories}` without writing."""
        memories = self._read_memories(scope, project_slug)
        if not memories:
            return {"topics": self.get_topics(scope, project_slug), "memories": []}
        short_ids = _short_ids(memories)
        pinned = (
            "Two topics have fixed names, use them verbatim: `profile` (who the user is) and `preferences` "
            "(how the user wants the assistant to respond and work). "
            if scope == "global"
            else ""
        )
        system_prompt = (
            "You reorganize a user's memory facts into topics: named subjects (1-3 words) with a one-line "
            "summary naming what the topic covers, e.g. \"The user's Linux setup and hardware\", never a fact. "
            f"{pinned}Group facts by subject, not by kind of fact. Prefer few, well-filled topics over many "
            "small ones. Assign every fact to exactly one topic. Output valid JSON without prose."
        )
        facts_block = _facts_block(short_ids)
        user_prompt = f"""
Facts (id [current topic] text):
{facts_block}

Return JSON with this exact shape:
{{"topics": [{{"name": "...", "summary": "..."}}], "assignments": [{{"id": "<fact id>", "topic": "<topic name>"}}]}}
""".strip()
        data = await self._generate_json(llm, system_prompt, user_prompt)

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
        previous_summaries = {t["name"]: t["summary"] for t in self.get_topics(scope, project_slug)}
        regrouped = []
        for memory in memories:
            topic = assignments.get(memory.id, memory.topic)
            if all(t["name"] != topic for t in topics):
                topics.append({"name": topic, "summary": previous_summaries.get(topic, "")})
            regrouped.append({**vars(memory), "topic": topic})
        used = {m["topic"] for m in regrouped}
        return {"topics": [t for t in topics if t["name"] in used], "memories": regrouped}

    # ------------------------------------------------------------------
    # Pending buffer
    # ------------------------------------------------------------------

    def _pending_path(self, review_id: str) -> DataStorePath:
        # Strip everything but filename-safe chars so review_id cannot traverse paths.
        safe = re.sub(r"[^A-Za-z0-9_-]", "", review_id)
        return self.pending_dir / f"{safe}.json"

    def _write_pending(self, review_id: str, chat_id: str | None, target_index: int, scopes: list[str]) -> None:
        """Buffer the watermark target each proposing scope advances to when it commits."""
        safe_write_json(self._pending_path(review_id), {"chat_id": chat_id, "target_index": target_index, "scopes": scopes})

    def _delete_pending(self, review_id: str) -> None:
        try:
            self._pending_path(review_id).unlink(missing_ok=True)
        except OSError as e:
            log.warning("Failed to delete pending buffer %s: %s", review_id, e)

    def _consume_pending_scope(self, review_id: str, scope: str) -> None:
        """Advance ``(chat_id, scope)``'s watermark and drop *scope* from the buffer, deleting it once empty."""
        path = self._pending_path(review_id)
        data = self._read_json(path, "Failed to read pending buffer %s: %s", review_id, level=logging.WARNING)
        if not isinstance(data, dict):
            if path.exists():
                self._delete_pending(review_id)
            return

        chat_id = data.get("chat_id")
        target = data.get("target_index")
        pending_scopes = data.get("scopes") or []
        if scope in pending_scopes and chat_id and isinstance(target, int):
            self._advance_watermark(str(chat_id), scope, target)

        remaining = [s for s in pending_scopes if s != scope]
        if remaining:
            data["scopes"] = remaining
            safe_write_json(path, data)
        else:
            self._delete_pending(review_id)

    # ------------------------------------------------------------------
    # Conversation watermarks (messages already memorized, by chat id + scope)
    # ------------------------------------------------------------------

    def _watermarks_path(self) -> DataStorePath:
        return self.base_dir / "watermarks.json"

    def _read_watermarks(self) -> dict[str, dict[str, int]]:
        raw = self._read_json(self._watermarks_path(), "Failed to read watermarks: %s")
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict[str, int]] = {}
        for chat_id, scopes in raw.items():
            if isinstance(scopes, dict):
                out[str(chat_id)] = {str(s): int(i) for s, i in scopes.items() if isinstance(i, int)}
        return out

    def _watermark_for(self, chat_id: str | None, scope: str) -> int:
        """Return the index into a chat's messages before which everything is already memorized."""
        if not chat_id:
            return 0
        return self._read_watermarks().get(chat_id, {}).get(scope, 0)

    def _advance_watermark(self, chat_id: str, scope: str, index: int) -> None:
        """Move ``(chat_id, scope)``'s watermark forward to *index* (monotonic, never backwards)."""
        marks = self._read_watermarks()
        scopes = marks.setdefault(chat_id, {})
        if index <= scopes.get(scope, 0):
            return
        scopes[scope] = index
        safe_write_json(self._watermarks_path(), marks)

    # ------------------------------------------------------------------
    # Topics and memories
    # ------------------------------------------------------------------

    def _topics_path(self, scope: str, project_slug: str | None) -> DataStorePath:
        if scope == "global":
            return self.base_dir / "global-topics.json"
        return self._project_dir(project_slug) / "memory/topics.json"

    def _memories_path(self, scope: str, project_slug: str | None) -> DataStorePath:
        if scope == "global":
            return self.base_dir / "global-profile.json"
        return self._project_dir(project_slug) / "memory/memory.json"

    def get_topics(self, scope: str, project_slug: str | None = None) -> list[dict[str, str]]:
        """Return the stored topics of *scope*, or the defaults before the first write."""
        fallback = [dict(t) for t in (DEFAULT_GLOBAL_TOPICS if scope == "global" else DEFAULT_PROJECT_TOPICS)]
        raw = self._read_json(self._topics_path(scope, project_slug), "Failed to read topics for %r: %s", scope)
        if not isinstance(raw, list):
            return fallback
        return [
            {"name": str(t["name"]), "summary": str(t["summary"])}
            for t in raw if isinstance(t, dict) and "name" in t and "summary" in t
        ]

    def set_topics(self, scope: str, topics: list[dict[str, str]], project_slug: str | None = None) -> None:
        safe_write_json(self._topics_path(scope, project_slug), topics)

    def list_memories(self, scope: str, project_slug: str | None = None) -> list[dict[str, Any]]:
        """Return the stored memories of *scope* as plain dicts."""
        return [vars(m) for m in self._read_memories(scope, project_slug)]

    def write_memories(self, scope: str, memories: list[dict[str, Any]], project_slug: str | None = None) -> None:
        """Replace the stored memories of *scope*, normalizing every entry."""
        now = utc_now_iso()
        normalized = [
            m
            for m in (StoredMemory.from_raw(scope, raw, now, normalize=True) for raw in memories if isinstance(raw, dict))
            if m.text
        ]
        safe_write_json(self._memories_path(scope, project_slug), [vars(m) for m in normalized])

    def _read_memories(self, scope: str, project_slug: str | None) -> list[StoredMemory]:
        path = self._memories_path(scope, project_slug)
        raw = self._read_json(path, "Failed to read stored memories from %s: %s", path)
        now = utc_now_iso()
        return [StoredMemory.from_raw(scope, m, now) for m in _dicts(raw)]

    @staticmethod
    def _read_json(path: DataStorePath, failure: str, *args: object, level: int = logging.ERROR) -> object:
        """Parsed JSON at *path*, or None when missing or unreadable (the failure is logged with *args* and the error)."""
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.log(level, failure, *args, e)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project_dir(self, project_slug: str | None) -> DataStorePath:
        # Reject separators and dot segments so a slug cannot escape the project directory.
        if not project_slug:
            raise ValueError("Project slug is required")
        if "/" in project_slug or "\\" in project_slug or project_slug in {".", ".."}:
            raise ValueError("Invalid project slug")
        return self.base_dir / "project" / project_slug

    @staticmethod
    def _get_transcript(messages: list[dict[str, str]]) -> str:
        blocks = [f"{m.get('role', 'user').upper()}: {m.get('content', '').strip()}" for m in messages]
        return "\n\n".join(blocks) or "No conversation content."

    async def _generate_json(self, llm: MemoryLLM, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Call the memory model and parse JSON, retrying once without response_format.

        Small models occasionally reject ``response_format`` or wrap JSON in prose.
        """

        def call(force_json: bool) -> str:
            kwargs: dict[str, Any] = dict(
                model=self.model,
                user_msg=user_prompt,
                user_msg_history=[],
                system_prompt=system_prompt,
                stream=False,
            )
            if force_json:
                kwargs["response_format"] = {"type": "json_object"}
            return completion_text(llm, **kwargs) or "{}"

        last_err: Exception | None = None
        for force_json in (True, False):
            try:
                text = await asyncio.wait_for(asyncio.to_thread(call, force_json), timeout=EXTRACT_TIMEOUT)
                return extract_json_from_llm(text)
            except Exception as e:  # noqa: BLE001 - retry both transport and parse failures
                last_err = e
                log.warning("Memory JSON call failed (response_format=%s): %s", force_json, e)

        raise RuntimeError(f"Memory model did not return valid JSON: {last_err}") from last_err
