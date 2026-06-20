"""Document-backed memory store for user profile and project memory.

Pipeline (all local-friendly, no vector DB):

1. ``extract``  – one JSON call to a small model produces atomic candidate
   memories, each tagged with a category from ``DEFAULT_GLOBAL_CATEGORIES`` or
   ``DEFAULT_PROJECT_CATEGORIES``. Candidates are buffered to
   ``memory/pending/<review_id>.json``.
2. user review  – the frontend gates which candidates are accepted.
3. ``reconcile`` – accepted candidates are merged into the canonical store
   *per category* (1-to-n within a single category bucket), so only memories of
   the same category are ever compared. Unaffected categories are left untouched.
4. ``commit``   – the canonical JSON + rendered markdown are written.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Protocol
import uuid

from config import DIRECTORY_CHAT_HISTORIES, MEMORY_MODEL
from lib.json_io import safe_write_json
from lib.safe_path import safe_resolve

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path


class MemoryLLM(Protocol):
    def api_query(self, **kwargs: object) -> object: ...


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


PROJECT_MEMORY = "memory/memory.md"
EXTRACT_TIMEOUT = 60
FALLBACK_CATEGORY = "note"

# Per-scope framing fed into the extraction prompt (fills "You extract durable ___").
GLOBAL_SCOPE_GUIDANCE = "facts about the user, valid across all projects (identity, preferences, working style, goals)"
PROJECT_SCOPE_GUIDANCE = (
    "facts about the active project or subject (purpose, concepts, decisions, constraints, resources, open questions)"
)


# Categories define which *types* of memory are extracted. Tuned to be useful
# for both studying lectures and hobby programming. Add entries here to teach the
# extractor new buckets — extraction and per-category dedup pick them up
# automatically.
DEFAULT_GLOBAL_CATEGORIES: list[dict[str, str]] = [
    {
        "name": "identity",
        "description": "Stable facts about who the user is: name, role, field of study, spoken/programming languages, background.",
    },
    {
        "name": "communication_preference",
        "description": "How the user wants the assistant to respond: tone, verbosity, language, formatting.",
    },
    {
        "name": "learning_preference",
        "description": "How the user likes to study and learn: preferred explanations, examples, pace, analogies, formats.",
    },
    {
        "name": "engineering_preference",
        "description": "General coding and tooling preferences across projects: languages, frameworks, libraries, code style.",
    },
    {"name": "goals", "description": "Longer-term personal, academic, or career goals the user is working toward."},
]

DEFAULT_PROJECT_CATEGORIES: list[dict[str, str]] = [
    {"name": "purpose", "description": "What this project or course/subject is about and what it aims to achieve."},
    {"name": "current_focus", "description": "What is being worked on or studied right now."},
    {
        "name": "key_concept",
        "description": "Important concepts, definitions, formulas, or facts worth retaining (e.g. from lectures).",
    },
    {"name": "decisions", "description": "Design, architecture, or technical decisions that were made, with their rationale."},
    {"name": "constraint", "description": "Requirements, limitations, deadlines, or rules that apply to the project."},
    {"name": "resource", "description": "Useful references, files, links, datasets, tools, or commands tied to this project."},
    {"name": "open_questions", "description": "Unresolved questions, blockers, or pending tasks to follow up on."},
]


@dataclass
class ProposedMemory:
    id: str
    memory: str
    scope: str
    category: str = FALLBACK_CATEGORY


@dataclass
class ExtractResult:
    review_id: str
    global_memories: list[ProposedMemory] = field(default_factory=list)
    project_memories: list[ProposedMemory] = field(default_factory=list)


@dataclass
class StoredMemory:
    id: str
    text: str
    category: str
    scope: str  # "global" | "project"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)


def _json_from_response(text: str) -> dict[str, Any]:
    from lib.llm_json import extract_json_from_llm

    return extract_json_from_llm(text)


class MemoryStore:
    """Owns memory extraction, canonical document updates, and the pending buffer."""

    def __init__(self, base_dir: Path = DIRECTORY_CHAT_HISTORIES) -> None:
        self.base_dir = base_dir.resolve()
        self.memory_root = self.base_dir / "memory"
        self.pending_dir = self.memory_root / "pending"

    # ------------------------------------------------------------------
    # Public document helpers
    # ------------------------------------------------------------------

    def chat_context(self, project_slug: str | None = None) -> str:
        """Return canonical memory context for runtime chat injection."""
        global_path = self._global_profile_path()
        project_path = self._project_memory_path(project_slug) if project_slug else None

        global_profile = self._read_existing_doc(global_path).strip()

        project_memory = ""
        if project_path:
            project_memory = self._read_existing_doc(project_path).strip()

        parts = []
        if global_profile:
            parts.append(f"## Global User Profile\n\n{global_profile}")
        if project_memory:
            parts.append(f"## Active Project Memory\n\n{project_memory}")
        if not parts:
            return ""
        return "\n\n".join(parts)

    def augment_system_prompt(self, system_prompt: str, project_slug: str | None = None) -> str:
        memory_context = self.chat_context(project_slug)
        if not memory_context:
            return system_prompt
        if not system_prompt.strip():
            return memory_context
        return f"{system_prompt.rstrip()}\n\n# Persistent Memory Context\n\n{memory_context}"

    # ------------------------------------------------------------------
    # Extract candidate memories (stateless; buffered to pending/<review_id>.json)
    # ------------------------------------------------------------------

    async def extract(
        self,
        llm: MemoryLLM,
        messages: list[dict[str, str]],
        project_slug: str | None = None,
        chat_id: str | None = None,
        scope: str | None = None,
    ) -> ExtractResult:
        """Extract candidate memories from messages since this conversation's last memorization.

        *scope* targets which scope to extract: ``"global"``, ``"project"``, or ``None`` for both.
        Requested scopes are extracted in parallel; each prompt carries only its own categories
        plus the memories already on record, so the model emits *deltas* (new or corrected facts)
        instead of re-proposing known ones. Watermarks are tracked **per (chat_id, scope)** and
        advance only when that scope's review is committed — so a scope-targeted ``/memorize-global``
        does not starve a later ``/memorize-project`` on the same conversation, and cancelling
        leaves the messages eligible next time.
        """
        self._ensure_dirs()
        review_id = str(uuid.uuid4())
        target = len(messages)

        do_global = scope in (None, "global")
        do_project = scope in (None, "project") and bool(project_slug)

        async def run_scope(scope_name: str, guidance: str, categories: list[dict[str, str]], json_path: Path) -> list[ProposedMemory]:
            start = self._watermark_for(chat_id, scope_name) if chat_id else 0
            tail = messages[start:]
            if not tail:
                return []
            return await self._extract_scope(
                llm, scope_name, guidance, self._get_transcript(tail), categories, self._read_stored_memories(json_path)
            )

        tasks: list[Any] = []
        if do_global:
            global_json, _, _ = self._scope_paths("global", None)
            tasks.append(("global", run_scope("global", GLOBAL_SCOPE_GUIDANCE, self.get_categories("global"), global_json)))
        if do_project:
            project_json, _, _ = self._scope_paths("project", project_slug)
            tasks.append(
                ("project", run_scope("project", PROJECT_SCOPE_GUIDANCE, self.get_categories("project", project_slug), project_json))
            )

        results = await asyncio.gather(*(coro for _, coro in tasks))
        by_scope = dict(zip((name for name, _ in tasks), results))
        global_memories = by_scope.get("global", [])
        project_memories = by_scope.get("project", [])

        committable = [name for name, mems in (("global", global_memories), ("project", project_memories)) if mems]
        if committable:
            self._write_pending(review_id, chat_id, target, committable, global_memories + project_memories)
        return ExtractResult(review_id=review_id, global_memories=global_memories, project_memories=project_memories)

    async def _extract_scope(
        self,
        llm: MemoryLLM,
        scope: str,
        scope_guidance: str,
        transcript: str,
        categories: list[dict[str, str]],
        existing: list[StoredMemory],
    ) -> list[ProposedMemory]:
        """One scope's extraction call: emit only new/corrected memories given what's on record."""
        cats_block = self._format_categories(categories)
        existing_block = self._format_existing(existing) or "(nothing on record yet)"

        system_prompt = (
            f"You extract durable {scope_guidance} from a conversation for a user-reviewed "
            "learning system. Output valid JSON only, no prose.\n\n"
            "You are given the allowed categories and the memories already on record. Output only "
            "NEW information as a delta:\n"
            "- SKIP anything the conversation merely repeats — never re-output a fact already on record.\n"
            "- When the conversation refines or CONTRADICTS a recorded memory, output the corrected "
            "statement as a new memory; a later step replaces the stale one.\n"
            "- Output genuinely new durable facts not yet on record.\n\n"
            "Rules:\n"
            "- Tag each memory with exactly one allowed category.\n"
            "- Prefer atomic memories: maximal information in minimal words; a longer form is "
            "acceptable only when compressing would lose meaning.\n"
            "- Omit session-transient details.\n"
            "- An empty list is valid when there is nothing new to record."
        )
        user_prompt = f"""
Conversation:
{transcript}

Allowed categories:
{cats_block}

Already on record (current knowledge, may be outdated):
{existing_block}

Return JSON with this exact shape:
{{"memories": [{{"memory": "...", "category": "<one allowed category name>"}}]}}
""".strip()

        data = await self._generate_json(llm, system_prompt, user_prompt, timeout=EXTRACT_TIMEOUT)
        return self._build_candidates(data.get("memories", []), scope, categories)

    # ------------------------------------------------------------------
    # Commit / Cancel (stateless & idempotent)
    # ------------------------------------------------------------------

    async def commit_async(
        self,
        llm: MemoryLLM,
        scope: str,  # "global" | "project"
        accepted_memories: list[ProposedMemory],
        project_slug: str | None = None,
        review_id: str | None = None,
        rejected_memories: list[dict[str, Any]] | None = None,
        revised_memories: list[dict[str, Any]] | None = None,
    ) -> None:
        json_path, md_path, title = self._scope_paths(scope, project_slug)

        if revised_memories is not None:
            # Revised memories come from a preview and may carry an extra "status"
            # field for the UI — keep only the canonical StoredMemory fields.
            now = _now_iso()
            updated = [
                StoredMemory(
                    id=m["id"],
                    text=m["text"],
                    category=m["category"],
                    scope=m["scope"],
                    created_at=str(m.get("created_at") or now),
                    updated_at=str(m.get("updated_at") or now),
                )
                for m in revised_memories
            ]
        else:
            existing = self._read_stored_memories(json_path)
            updated = await self.reconcile(llm, accepted_memories, existing, scope=scope)

        self._write_stored_memories(json_path, updated)
        md_content = self.render_memories_as_markdown(updated, title, scope=scope, project_slug=project_slug)
        self._write_doc(md_path, md_content)

        if review_id:
            self._consume_pending_scope(review_id, scope)

    async def cancel_async(self, review_id: str) -> None:
        """Discard the outstanding candidates buffer for *review_id*."""
        self._delete_pending(review_id)

    # ------------------------------------------------------------------
    # Preview: reconcile without writing (for diff display)
    # ------------------------------------------------------------------

    async def preview(
        self,
        llm: MemoryLLM,
        scope: str,
        accepted_memories: list[ProposedMemory],
        project_slug: str | None = None,
    ) -> dict[str, object]:
        """Run reconcile + render and return existing vs proposed memory lists and markdown."""
        json_path, md_path, title = self._scope_paths(scope, project_slug)

        existing_stored = self._read_stored_memories(json_path)
        existing_md = self._read_existing_doc(md_path)

        updated_pairs = await self._reconcile_with_status(llm, accepted_memories, existing_stored, scope=scope)
        updated = [m for m, _ in updated_pairs]
        proposed_md = self.render_memories_as_markdown(updated, title, scope=scope, project_slug=project_slug)

        return {
            "existing_markdown": existing_md,
            "revised_markdown": proposed_md.rstrip() + "\n",
            "existing_memories": [self._memory_to_dict(m) for m in existing_stored],
            "revised_memories": [{**self._memory_to_dict(m), "status": status} for m, status in updated_pairs],
        }

    # ------------------------------------------------------------------
    # Per-category reconciliation (1-to-n within a single category bucket)
    # ------------------------------------------------------------------

    async def reconcile(
        self,
        llm: MemoryLLM,
        candidates: list[ProposedMemory],
        existing: list[StoredMemory],
        scope: str = "global",
    ) -> list[StoredMemory]:
        """Merge accepted *candidates* into *existing*, comparing only within a category."""
        pairs = await self._reconcile_with_status(llm, candidates, existing, scope)
        return [m for m, _ in pairs]

    async def _reconcile_with_status(
        self,
        llm: MemoryLLM,
        candidates: list[ProposedMemory],
        existing: list[StoredMemory],
        scope: str = "global",
    ) -> list[tuple[StoredMemory, str]]:
        """Reconcile and tag each resulting memory with its origin.

        Status is one of ``pre-existing`` (unchanged), ``new`` (a candidate kept
        verbatim), or ``combined`` (a model-synthesized merge of old and/or new).
        Categories with no new candidates are passed through untouched. For each
        touched category the model receives just that category's existing + new
        memories and returns the merged canonical list for it.
        """
        if not candidates:
            return [(m, "pre-existing") for m in existing]

        existing_by_cat: dict[str, list[StoredMemory]] = {}
        for m in existing:
            existing_by_cat.setdefault(m.category, []).append(m)

        candidates_by_cat: dict[str, list[ProposedMemory]] = {}
        for c in candidates:
            candidates_by_cat.setdefault(c.category, []).append(c)

        result: list[tuple[StoredMemory, str]] = []

        # Untouched categories pass through verbatim.
        for cat, mems in existing_by_cat.items():
            if cat not in candidates_by_cat:
                result.extend((m, "pre-existing") for m in mems)

        # Touched categories get reconciled — independent per category, so run concurrently.
        touched = list(candidates_by_cat.items())
        reconciled = await asyncio.gather(
            *(self._reconcile_category(llm, scope, cat, existing_by_cat.get(cat, []), cands) for cat, cands in touched)
        )
        for cat_result in reconciled:
            result.extend(cat_result)

        return result

    async def _reconcile_category(
        self,
        llm: MemoryLLM,
        scope: str,
        category: str,
        existing: list[StoredMemory],
        candidates: list[ProposedMemory],
    ) -> list[tuple[StoredMemory, str]]:
        existing_texts = [m.text.strip() for m in existing if m.text.strip()]
        candidate_texts = [c.memory.strip() for c in candidates if c.memory.strip()]

        # Nothing to merge against — accept candidates as-is (deduped against existing).
        if not existing_texts:
            final_texts = self._dedup_keep_order(candidate_texts)
        else:
            final_texts = await self._merge_category_with_llm(llm, category, existing_texts, candidate_texts)

        return self._classify_texts(final_texts, category, scope, existing, existing_texts, candidate_texts)

    async def _merge_category_with_llm(
        self,
        llm: MemoryLLM,
        category: str,
        existing_texts: list[str],
        candidate_texts: list[str],
    ) -> list[str]:
        system_prompt = (
            "You maintain a deduplicated list of memories within a single category. "
            "You are given the current memories and some new candidate memories. "
            "Return the updated canonical list for this category as JSON only. "
            "Merge overlapping old and new memories into a single clear statement. "
            "When a new memory updates or contradicts an old one, keep the new information "
            "and drop the stale one. Keep genuinely distinct memories separate. "
            "Do not invent facts that are not present in the inputs. "
            "Keep each memory atomic and concise."
        )
        user_prompt = f"""
Category: {category}

Current memories:
{json.dumps(existing_texts, indent=2, ensure_ascii=False)}

New candidate memories:
{json.dumps(candidate_texts, indent=2, ensure_ascii=False)}

Return JSON with this exact shape:
{{"memories": ["...", "..."]}}
""".strip()

        fallback = self._dedup_keep_order(existing_texts + candidate_texts)
        try:
            data = await self._generate_json(llm, system_prompt, user_prompt, timeout=EXTRACT_TIMEOUT)
        except Exception as e:
            log.error("Category reconcile failed for %r — keeping existing + new: %s", category, e)
            return fallback

        raw = data.get("memories")
        if not isinstance(raw, list):
            log.error("Category reconcile for %r returned non-list — keeping existing + new", category)
            return fallback

        final = self._dedup_keep_order(str(t).strip() for t in raw if str(t).strip())
        # Guard against a model that wipes a populated category.
        if not final and (existing_texts or candidate_texts):
            log.error("Category reconcile for %r returned empty — keeping existing + new", category)
            return fallback
        return final

    def _classify_texts(
        self,
        texts: list[str],
        category: str,
        scope: str,
        existing: list[StoredMemory],
        existing_texts: list[str],
        candidate_texts: list[str],
    ) -> list[tuple[StoredMemory, str]]:
        """Build (StoredMemory, status) pairs, reusing existing ids and timestamps for unchanged text."""
        by_text = {m.text.strip(): m for m in existing}
        existing_set = {t.lower() for t in existing_texts}
        candidate_set = {t.lower() for t in candidate_texts}
        now = _now_iso()
        out: list[tuple[StoredMemory, str]] = []
        for text in texts:
            prev = by_text.get(text)
            key = text.lower()
            if key in existing_set and prev:
                # Unchanged — preserve id and both timestamps.
                status = "pre-existing"
                mem = StoredMemory(
                    id=prev.id,
                    text=text,
                    category=category,
                    scope=scope,
                    created_at=prev.created_at,
                    updated_at=prev.updated_at,
                )
            elif key in candidate_set:
                # Brand-new memory from this extraction.
                status = "new"
                mem = StoredMemory(
                    id=str(uuid.uuid4()),
                    text=text,
                    category=category,
                    scope=scope,
                    created_at=now,
                    updated_at=now,
                )
            else:
                # LLM synthesised a combined/updated form — preserve created_at if we can find it.
                status = "combined"
                created_at = prev.created_at if prev else now
                mem = StoredMemory(
                    id=prev.id if prev else str(uuid.uuid4()),
                    text=text,
                    category=category,
                    scope=scope,
                    created_at=created_at,
                    updated_at=now,
                )
            out.append((mem, status))
        return out

    @staticmethod
    def _dedup_keep_order(items: Any) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            text = item.strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                out.append(text)
        return out

    # ------------------------------------------------------------------
    # Pending buffer (replaces the old vector-store staging)
    # ------------------------------------------------------------------

    def _pending_path(self, review_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_-]", "", review_id)
        return self.pending_dir / f"{safe}.json"

    def _write_pending(
        self, review_id: str, chat_id: str | None, target_index: int, scopes: list[str], candidates: list[ProposedMemory]
    ) -> None:
        """Buffer the candidates plus the per-scope watermark target the commit step advances to.

        *scopes* are the scopes that produced candidates (and so will be committed); each is
        removed as its scope commits, and the buffer is deleted once all have been consumed.
        """
        self._ensure_dirs()
        safe_write_json(
            self._pending_path(review_id),
            {"chat_id": chat_id, "target_index": target_index, "scopes": scopes, "candidates": [vars(c) for c in candidates]},
        )

    def _delete_pending(self, review_id: str) -> None:
        try:
            self._pending_path(review_id).unlink(missing_ok=True)
        except OSError as e:
            log.warning("Failed to delete pending buffer %s: %s", review_id, e)

    def _consume_pending_scope(self, review_id: str, scope: str) -> None:
        """Advance ``(chat_id, scope)``'s watermark from the pending buffer for the committed scope.

        Removes *scope* from the buffer's pending scopes; deletes the buffer once all extracted
        scopes have committed. Idempotent and order-independent across per-scope commits; cancelling
        deletes without advancing.
        """
        path = self._pending_path(review_id)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("Failed to read pending buffer %s: %s", review_id, e)
            self._delete_pending(review_id)
            return
        if not isinstance(data, dict):
            self._delete_pending(review_id)
            return

        chat_id = data.get("chat_id")
        target = data.get("target_index")
        scopes = [s for s in (data.get("scopes") or []) if s != scope]
        if scope in (data.get("scopes") or []) and chat_id and isinstance(target, int):
            self._advance_watermark(str(chat_id), scope, target)

        if scopes:
            data["scopes"] = scopes
            safe_write_json(path, data)
        else:
            self._delete_pending(review_id)

    # ------------------------------------------------------------------
    # Conversation watermarks (messages already memorized, by chat id + scope)
    # ------------------------------------------------------------------

    def _watermarks_path(self) -> Path:
        return self.memory_root / "watermarks.json"

    def _read_watermarks(self) -> dict[str, dict[str, int]]:
        path = self._watermarks_path()
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.error("Failed to read watermarks: %s", e)
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict[str, int]] = {}
        for chat_id, scopes in raw.items():
            if isinstance(scopes, dict):
                out[str(chat_id)] = {str(s): int(i) for s, i in scopes.items() if isinstance(i, int)}
        return out

    def _watermark_for(self, chat_id: str | None, scope: str) -> int:
        if not chat_id:
            return 0
        return self._read_watermarks().get(chat_id, {}).get(scope, 0)

    def _advance_watermark(self, chat_id: str, scope: str, index: int) -> None:
        """Move ``(chat_id, scope)``'s watermark forward to *index* (monotonic — never backwards)."""
        marks = self._read_watermarks()
        scopes = marks.setdefault(chat_id, {})
        if index <= scopes.get(scope, 0):
            return
        scopes[scope] = index
        safe_write_json(self._watermarks_path(), marks)

    # ------------------------------------------------------------------
    # Category management
    # ------------------------------------------------------------------

    def _categories_path(self, scope: str, project_slug: str | None) -> "Path":
        if scope == "global":
            return self.memory_root / "global-categories.json"
        return self._project_dir(project_slug) / "memory/categories.json"

    def get_categories(self, scope: str, project_slug: str | None = None) -> list[dict[str, str]]:
        """Return the active category list for *scope*, falling back to defaults."""
        path = self._categories_path(scope, project_slug)
        if not path.exists():
            return DEFAULT_GLOBAL_CATEGORIES if scope == "global" else DEFAULT_PROJECT_CATEGORIES
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            cats = [c for c in raw if isinstance(c, dict) and "name" in c and "description" in c]
            return cats or (DEFAULT_GLOBAL_CATEGORIES if scope == "global" else DEFAULT_PROJECT_CATEGORIES)
        except Exception as e:
            log.error("Failed to read categories for %r: %s", scope, e)
            return DEFAULT_GLOBAL_CATEGORIES if scope == "global" else DEFAULT_PROJECT_CATEGORIES

    def set_categories(self, scope: str, categories: list[dict[str, str]], project_slug: str | None = None) -> None:
        """Persist *categories* for *scope*."""
        safe_write_json(self._categories_path(scope, project_slug), categories)

    async def remap_orphaned(
        self,
        llm: MemoryLLM,
        orphaned_memories: list[dict[str, Any]],
        remaining_categories: list[dict[str, str]],
        scope: str,
    ) -> list[dict[str, Any]]:
        """Reassign *orphaned_memories* to *remaining_categories* via LLM.

        Falls back to the first remaining category on any failure so that
        memories are never silently lost.
        """
        if not orphaned_memories:
            return []
        if not remaining_categories:
            return []

        fallback_cat = remaining_categories[0]["name"]

        categories_block = self._format_categories(remaining_categories)
        system_prompt = "You reassign memory items to the best-fitting category from a given list. Output valid JSON only, no prose."
        user_prompt = f"""
Memories to reassign:
{json.dumps([m["text"] for m in orphaned_memories], indent=2, ensure_ascii=False)}

Available categories:
{categories_block}

Return JSON with this exact shape:
{{"assignments": [{{"memory": "...", "category": "<one category name>"}}]}}

Rules:
- "category" MUST be one of the listed category names.
- Keep the memory text unchanged.
- One entry per input memory, in the same order.
""".strip()

        try:
            data = await self._generate_json(llm, system_prompt, user_prompt, timeout=EXTRACT_TIMEOUT)
        except Exception as e:
            log.error("remap_orphaned LLM call failed: %s — using fallback category %r", e, fallback_cat)
            return [{**m, "category": fallback_cat} for m in orphaned_memories]

        assignments = data.get("assignments", [])
        allowed = {c["name"] for c in remaining_categories}
        result: list[dict[str, Any]] = []
        for idx, m in enumerate(orphaned_memories):
            if idx < len(assignments):
                cat = str(assignments[idx].get("category", "")).strip()
                if cat not in allowed:
                    cat = fallback_cat
            else:
                cat = fallback_cat
            result.append({**m, "category": cat})
        return result

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def list_memories(self, scope: str, project_slug: str | None = None) -> list[dict[str, Any]]:
        """Return the canonical stored memories for *scope* as plain dicts."""
        json_path, _, _ = self._scope_paths(scope, project_slug)
        return [self._memory_to_dict(m) for m in self._read_stored_memories(json_path)]

    def move_memory(
        self,
        memory_id: str,
        from_scope: str,
        to_scope: str,
        from_project_slug: str | None = None,
        to_project_slug: str | None = None,
    ) -> None:
        """Move a single memory from one scope to another, committing both sides immediately.

        If the memory's category does not exist in the destination scope's category list,
        it is reassigned to ``FALLBACK_CATEGORY`` so it remains visible.
        """
        from_json, from_md, from_title = self._scope_paths(from_scope, from_project_slug)
        to_json, to_md, to_title = self._scope_paths(to_scope, to_project_slug)

        from_mems = self._read_stored_memories(from_json)
        target = next((m for m in from_mems if m.id == memory_id), None)
        if target is None:
            raise ValueError(f"Memory {memory_id!r} not found in {from_scope}")

        dest_cat_names = {c["name"] for c in self.get_categories(to_scope, to_project_slug)}
        dest_category = target.category if target.category in dest_cat_names else FALLBACK_CATEGORY

        now = _now_iso()
        moved = StoredMemory(
            id=target.id,
            text=target.text,
            category=dest_category,
            scope=to_scope,
            created_at=target.created_at,
            updated_at=now,
        )

        to_mems = self._read_stored_memories(to_json)
        # Idempotent: remove any existing entry with same id from destination.
        to_mems_updated = [m for m in to_mems if m.id != memory_id] + [moved]
        from_mems_updated = [m for m in from_mems if m.id != memory_id]

        self._write_stored_memories(from_json, from_mems_updated)
        self._write_doc(
            from_md, self.render_memories_as_markdown(from_mems_updated, from_title, scope=from_scope, project_slug=from_project_slug)
        )

        self._write_stored_memories(to_json, to_mems_updated)
        self._write_doc(
            to_md, self.render_memories_as_markdown(to_mems_updated, to_title, scope=to_scope, project_slug=to_project_slug)
        )

    # ------------------------------------------------------------------
    # Canonical storage I/O & rendering
    # ------------------------------------------------------------------

    def _read_stored_memories(self, path: "Path") -> list[StoredMemory]:
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.error("Failed to read stored memories from %s: %s", path, e)
            return []
        if not isinstance(raw, list):
            return []
        now = _now_iso()
        out: list[StoredMemory] = []
        for m in raw:
            try:
                out.append(
                    StoredMemory(
                        id=str(m.get("id") or uuid.uuid4()),
                        text=str(m.get("text", "")),
                        category=str(m.get("category") or FALLBACK_CATEGORY),
                        scope=str(m.get("scope") or "global"),
                        created_at=str(m.get("created_at") or now),
                        updated_at=str(m.get("updated_at") or now),
                    )
                )
            except AttributeError:
                continue
        return out

    @staticmethod
    def _memory_to_dict(m: StoredMemory) -> dict[str, Any]:
        return dict(vars(m))

    def _write_stored_memories(self, path: Path, memories: list[StoredMemory]) -> None:
        safe_write_json(path, [vars(m) for m in memories])

    def render_memories_as_markdown(
        self,
        memories: list[StoredMemory],
        title: str,
        *,
        scope: str | None = None,
        project_slug: str | None = None,
    ) -> str:
        from collections import defaultdict

        groups: dict[str, list[str]] = defaultdict(list)
        for m in memories:
            groups[m.category].append(m.text)

        if scope is not None:
            category_names = [c["name"] for c in self.get_categories(scope, project_slug)]
            ordered = [name for name in category_names if name in groups]
            ordered.extend(name for name in groups if name not in category_names)
        else:
            ordered = list(groups.keys())

        parts = [f"# {title}"]
        for category in ordered:
            items = groups[category]
            parts.append(f"\n## {category.replace('_', ' ').title()}")
            parts.extend(f"- {item}" for item in items)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scope_paths(self, scope: str, project_slug: str | None) -> tuple[Path, Path, str]:
        if scope == "global":
            return self.memory_root / "global-profile.json", self._global_profile_path(), "Global Profile"
        return (
            self._project_dir(project_slug) / "memory/memory.json",
            self._project_memory_path(project_slug),
            "Project Memory",
        )

    @staticmethod
    def _format_categories(categories: list[dict[str, str]]) -> str:
        return "\n".join(f'  - "{c["name"]}": {c["description"]}' for c in categories)

    @staticmethod
    def _format_existing(memories: list[StoredMemory]) -> str:
        """Render stored memories grouped by category, for injection into the extraction prompt."""
        groups: dict[str, list[str]] = {}
        for m in memories:
            text = m.text.strip()
            if text:
                groups.setdefault(m.category, []).append(text)
        lines: list[str] = []
        for cat, texts in groups.items():
            lines.append(f"{cat}:")
            lines.extend(f"  - {t}" for t in texts)
        return "\n".join(lines)

    def _ensure_dirs(self) -> None:
        self.memory_root.mkdir(parents=True, exist_ok=True)
        self.pending_dir.mkdir(parents=True, exist_ok=True)

    def _global_profile_path(self) -> Path:
        return self.memory_root / "global-profile.md"

    def _project_dir(self, project_slug: str | None) -> Path:
        if not project_slug:
            raise ValueError("Project slug is required")
        return safe_resolve(self.base_dir, project_slug)

    def _project_memory_path(self, project_slug: str | None) -> Path:
        return self._project_dir(project_slug) / PROJECT_MEMORY

    def _read_existing_doc(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _write_doc(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")

    def _build_candidates(
        self,
        raw_items: object,
        scope: str,
        categories: list[dict[str, str]],
    ) -> list[ProposedMemory]:
        if not isinstance(raw_items, list):
            return []
        allowed = {c["name"] for c in categories}
        candidates: list[ProposedMemory] = []
        seen: set[str] = set()
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            memory = str(item.get("memory", "")).strip()
            if not memory:
                continue
            key = memory.lower()
            if key in seen:
                continue
            seen.add(key)
            category = str(item.get("category") or "").strip()
            if category not in allowed:
                category = FALLBACK_CATEGORY
            candidates.append(ProposedMemory(id=str(uuid.uuid4()), memory=memory, scope=scope, category=category))
        return candidates

    def _get_transcript(self, messages: list[dict[str, str]]) -> str:
        blocks: list[str] = []
        for message in messages:
            role = message.get("role", "user").upper()
            content = message.get("content", "").strip()
            blocks.append(f"{role}: {content}")
        transcript = "\n\n".join(blocks)
        return transcript or "No conversation content."

    async def _generate_json(
        self,
        llm: MemoryLLM,
        system_prompt: str,
        user_prompt: str,
        timeout: int = EXTRACT_TIMEOUT,
    ) -> dict[str, Any]:
        """Call the small model and parse JSON, retrying once without response_format.

        Small models occasionally reject ``response_format`` or wrap JSON in prose;
        the two-pass strategy plus ``_json_from_response`` salvaging keeps extraction
        from hard-failing.
        """

        def call(force_json: bool) -> str:
            kwargs: dict[str, Any] = dict(
                model=MEMORY_MODEL,
                user_msg=user_prompt,
                user_msg_history=[],
                system_prompt=system_prompt,
                stream=False,
            )
            if force_json:
                kwargs["response_format"] = {"type": "json_object"}
            response = llm.api_query(**kwargs)
            if isinstance(response, Exception):
                raise response
            return response.choices[0].message.content or "{}"

        last_err: Exception | None = None
        for force_json in (True, False):
            try:
                text = await asyncio.wait_for(asyncio.to_thread(call, force_json), timeout=timeout)
                return _json_from_response(text)
            except Exception as e:  # noqa: BLE001 - retry both transport and parse failures
                last_err = e
                log.warning("Memory JSON call failed (response_format=%s): %s", force_json, e)

        raise RuntimeError(f"Memory model did not return valid JSON: {last_err}") from last_err
