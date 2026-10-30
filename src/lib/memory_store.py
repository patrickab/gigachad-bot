"""Document-backed memory: atomic facts grouped into topics, per global profile and project.

``MemoryStore`` persists each ``MemoryScope``'s topics and facts, the pending review buffer and the per-chat
watermarks. Extraction lives in ``lib.memory_extract``, prompt injection in ``lib.memory_context``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import re
from typing import TYPE_CHECKING, Any, Literal
import uuid

from lib.clock import utc_now_iso
from lib.data_store import DataStore
from lib.json_docs import JsonDocs, JsonValue
from lib.storage_namespace import MEMORY, memory_memories_key, memory_topics_key

if TYPE_CHECKING:
    from collections.abc import Iterable

log = logging.getLogger(__name__)

ScopeKind = Literal["global", "project"]
FALLBACK_TOPIC = "note"
PENDING_PREFIX = f"{MEMORY}/pending"
WATERMARKS_KEY = f"{MEMORY}/watermarks.json"

DEFAULT_GLOBAL_TOPICS: list[dict[str, str]] = [
    {"name": "profile", "summary": "Who the user is: background, education, work, and setup."},
    {"name": "preferences", "summary": "How the user wants the assistant to respond and work."},
]

Write = tuple[str, JsonValue]


def topic_name(raw: object) -> str:
    """Stored topic name of a model- or user-written title: `Dev Environment` -> `dev_environment`."""
    return re.sub(r"[^\w]+", "_", str(raw or "").strip().lower()).strip("_")


def add_missing_topics(topics: list[dict[str, str]], names: Iterable[str], summaries: dict[str, str] | None = None) -> None:
    """Append a topic for every name in *names* that *topics* lacks, with its summary from *summaries*."""
    known = {t["name"] for t in topics}
    for name in names:
        if name not in known:
            topics.append({"name": name, "summary": (summaries or {}).get(name, "")})
            known.add(name)


@dataclass(frozen=True)
class MemoryScope:
    """The global profile (`slug` ignored) or one project's memory. A project slug is validated when a key is built."""

    kind: ScopeKind
    slug: str | None = None

    @classmethod
    def of(cls, kind: ScopeKind, project_slug: str | None) -> MemoryScope:
        return cls("global") if kind == "global" else cls("project", project_slug)

    @property
    def _project_slug(self) -> str | None:
        # An empty slug makes the key builders raise "Project slug is required".
        return None if self.kind == "global" else self.slug or ""

    @property
    def topics_key(self) -> str:
        return memory_topics_key(self._project_slug)

    @property
    def memories_key(self) -> str:
        return memory_memories_key(self._project_slug)

    def default_topics(self) -> list[dict[str, str]]:
        """Topics before the first write: the global profile starts with two, project topics emerge from extraction."""
        return [dict(t) for t in DEFAULT_GLOBAL_TOPICS] if self.kind == "global" else []


@dataclass
class MemoryOperation:
    """Proposed change to one fact, `target_*` name the stored fact an update or delete replaces.

    `index` is the 0-based place the fact takes in the scope's relevance order (0 = most relevant).
    None appends an add and keeps an update where it is.
    """

    id: str
    op: str  # "add" | "update" | "delete"
    scope: str
    topic: str
    text: str = ""
    target_id: str | None = None
    target_text: str | None = None
    index: int | None = None



@dataclass
class StoredMemory:
    id: str
    text: str
    topic: str
    scope: str  # "global" | "project"
    created_at: str
    updated_at: str

    @classmethod
    def from_stored(cls, scope: str, raw: dict[str, Any], now: str) -> StoredMemory:
        return cls._of(scope, raw, now, str(raw.get("text", "")), str(raw.get("topic") or FALLBACK_TOPIC))

    @classmethod
    def from_input(cls, scope: str, raw: dict[str, Any], now: str) -> StoredMemory:
        """Memory from a user-supplied dict, with stripped text and a canonical topic name."""
        return cls._of(scope, raw, now, str(raw.get("text") or "").strip(), topic_name(raw.get("topic")) or FALLBACK_TOPIC)

    @classmethod
    def _of(cls, scope: str, raw: dict[str, Any], now: str, text: str, topic: str) -> StoredMemory:
        created_at, updated_at = str(raw.get("created_at") or now), str(raw.get("updated_at") or now)
        return cls(str(raw.get("id") or uuid.uuid4()), text, topic, scope, created_at, updated_at)


class MemoryStore:
    """Topics and facts per scope, the pending review buffer, and per-chat watermarks."""

    def __init__(self, data_store: DataStore) -> None:
        self._docs = JsonDocs(data_store)

    def get_topics(self, scope: MemoryScope) -> list[dict[str, str]]:
        """Return the stored topics of *scope*, or the defaults before the first write."""
        raw = self._docs.read(scope.topics_key, log="memory topics")
        if not isinstance(raw, list):
            return scope.default_topics()
        valid = (t for t in raw if isinstance(t, dict) and "name" in t and "summary" in t)
        return [{"name": str(t["name"]), "summary": str(t["summary"])} for t in valid]

    def set_topics(self, scope: MemoryScope, topics: list[dict[str, str]]) -> None:
        self._docs.write(scope.topics_key, topics)

    def read_memories(self, scope: MemoryScope) -> list[StoredMemory]:
        raw = self._docs.read(scope.memories_key, log="stored memories")
        now = utc_now_iso()
        items = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
        return [StoredMemory.from_stored(scope.kind, item, now) for item in items]

    def list_memories(self, scope: MemoryScope) -> list[dict[str, Any]]:
        """Return the stored memories of *scope* as plain dicts."""
        return [asdict(m) for m in self.read_memories(scope)]

    def write_memories(self, scope: MemoryScope, memories: list[dict[str, Any]]) -> None:
        """Replace the stored memories of *scope*, normalizing every entry."""
        self._docs.write(scope.memories_key, _normalized(scope, memories))

    def commit(
        self, scope: MemoryScope, operations: list[MemoryOperation], topics: list[dict[str, str]], review_id: str | None = None
    ) -> None:
        """Apply accepted operations and topic summaries to *scope* and advance the review's watermark, atomically.

        Operations on facts that no longer exist are skipped. Empty *operations* still advance the
        watermark, so denied proposals are not offered again.
        """
        memories = self.read_memories(scope)
        by_id = {m.id: m for m in memories}
        now = utc_now_iso()
        for operation in operations:
            text = operation.text.strip()
            topic = topic_name(operation.topic) or FALLBACK_TOPIC
            if operation.op == "add" and text:
                _place(memories, StoredMemory(str(uuid.uuid4()), text, topic, scope.kind, now, now), operation.index)
            elif operation.op == "update" and text and operation.target_id in by_id:
                target = by_id[operation.target_id]
                target.text, target.topic, target.updated_at = text, topic, now
                if operation.index is not None:
                    memories.remove(target)
                    _place(memories, target, operation.index)
            elif operation.op == "delete" and operation.target_id in by_id:
                memories.remove(by_id.pop(operation.target_id))

        stored_topics = self.get_topics(scope)
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
        add_missing_topics(stored_topics, (m.topic for m in memories))

        writes: list[Write] = [
            (scope.topics_key, stored_topics),
            (scope.memories_key, _normalized(scope, [asdict(m) for m in memories])),
        ]
        deletes: list[str] = []
        if review_id:
            self._consume_pending_scope(review_id, scope.kind, writes, deletes)
        self._docs.apply(writes, deletes)

    def cancel(self, review_id: str) -> None:
        """Discard the outstanding review without advancing any watermark."""
        try:
            self._docs.delete(_pending_key(review_id))
        except OSError as e:
            log.warning("Failed to delete pending buffer %s: %s", review_id, e)

    def ensure_pending_dir(self) -> None:
        """Create the pending buffer's directory, as extraction always has."""
        if not self._docs.exists(MEMORY):
            self._docs.mkdir(MEMORY)
        self._docs.mkdir(PENDING_PREFIX)

    def write_pending(self, review_id: str, chat_id: str | None, target_index: int, scopes: list[str]) -> None:
        """Buffer the watermark target each proposing scope advances to when it commits."""
        self._docs.write(_pending_key(review_id), {"chat_id": chat_id, "target_index": target_index, "scopes": scopes})

    def _consume_pending_scope(self, review_id: str, kind: str, writes: list[Write], deletes: list[str]) -> None:
        """Queue advancing ``(chat_id, kind)``'s watermark and dropping *kind* from the buffer, deleting it once empty."""
        key = _pending_key(review_id)
        data = self._docs.read(key, log="pending memory buffer")
        if not isinstance(data, dict):
            if self._docs.exists(key):
                deletes.append(key)
            return

        chat_id = data.get("chat_id")
        target = data.get("target_index")
        pending_scopes = data.get("scopes") or []
        if kind in pending_scopes and chat_id and isinstance(target, int):
            marks = self._read_watermarks()
            per_chat = marks.setdefault(str(chat_id), {})
            # Watermarks are monotonic, never moving backwards.
            if target > per_chat.get(kind, 0):
                per_chat[kind] = target
                writes.append((WATERMARKS_KEY, marks))

        remaining = [s for s in pending_scopes if s != kind]
        if remaining:
            writes.append((key, {**data, "scopes": remaining}))
        else:
            deletes.append(key)

    def _read_watermarks(self) -> dict[str, dict[str, int]]:
        raw = self._docs.read(WATERMARKS_KEY, log="memory watermarks")
        if not isinstance(raw, dict):
            return {}
        chats = ((str(chat_id), scopes) for chat_id, scopes in raw.items() if isinstance(scopes, dict))
        return {chat_id: {str(s): int(i) for s, i in scopes.items() if isinstance(i, int)} for chat_id, scopes in chats}

    def watermark(self, chat_id: str | None, kind: str) -> int:
        """Return the index into a chat's messages before which everything is already memorized."""
        if not chat_id:
            return 0
        return self._read_watermarks().get(chat_id, {}).get(kind, 0)


def _place(memories: list[StoredMemory], memory: StoredMemory, index: int | None) -> None:
    """Insert *memory* so it ends up at *index* of *memories*, appended when None or past the end."""
    memories.insert(len(memories) if index is None else max(0, min(index, len(memories))), memory)


def _normalized(scope: MemoryScope, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now_iso()
    parsed = (StoredMemory.from_input(scope.kind, raw, now) for raw in memories if isinstance(raw, dict))
    return [asdict(m) for m in parsed if m.text]


def _pending_key(review_id: str) -> str:
    # Strip everything but filename-safe chars so review_id cannot traverse paths.
    safe = re.sub(r"[^A-Za-z0-9_-]", "", review_id)
    return f"{PENDING_PREFIX}/{safe}.json"
