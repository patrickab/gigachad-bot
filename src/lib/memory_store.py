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
import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Protocol
import uuid

from config import DIRECTORY_CHAT_HISTORIES, SMALL_MODEL

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path


class MemoryLLM(Protocol):
    def api_query(self, **kwargs: object) -> object: ...


MEMORY_ROOT = DIRECTORY_CHAT_HISTORIES / "memory"
PENDING_DIR = MEMORY_ROOT / "pending"
GLOBAL_PROFILE = MEMORY_ROOT / "global-profile.md"
PROJECT_MEMORY = "memory/memory.md"
EXTRACT_TIMEOUT = 60
FALLBACK_CATEGORY = "note"


# Categories define which *types* of memory are extracted. Tuned to be useful
# for both studying lectures and hobby programming. Add entries here to teach the
# extractor new buckets — extraction and per-category dedup pick them up
# automatically.
DEFAULT_GLOBAL_CATEGORIES: list[dict[str, str]] = [
    {"name": "identity",
     "description": "Stable facts about who the user is: name, role, field of study, spoken/programming languages, background."},
    {"name": "communication_preference",
     "description": "How the user wants the assistant to respond: tone, verbosity, language, formatting."},
    {"name": "learning_preference",
     "description": "How the user likes to study and learn: preferred explanations, examples, pace, analogies, formats."},
    {"name": "engineering_preference",
     "description": "General coding and tooling preferences across projects: languages, frameworks, libraries, code style."},
    {"name": "environment",
     "description": "The user's tools and setup: OS, editor, hardware, shell, recurring environment details."},
    {"name": "interest",
     "description": "Recurring topics, subjects, or hobbies the user cares about beyond a single project."},
    {"name": "goal",
     "description": "Longer-term personal, academic, or career goals the user is working toward."},
]

DEFAULT_PROJECT_CATEGORIES: list[dict[str, str]] = [
    {"name": "purpose",
     "description": "What this project or course/subject is about and what it aims to achieve."},
    {"name": "current_focus",
     "description": "What is being worked on or studied right now."},
    {"name": "key_concept",
     "description": "Important concepts, definitions, formulas, or facts worth retaining (e.g. from lectures)."},
    {"name": "decision",
     "description": "Design, architecture, or technical decisions that were made, with their rationale."},
    {"name": "constraint",
     "description": "Requirements, limitations, deadlines, or rules that apply to the project."},
    {"name": "resource",
     "description": "Useful references, files, links, datasets, tools, or commands tied to this project."},
    {"name": "open_question",
     "description": "Unresolved questions, blockers, or pending tasks to follow up on."},
]


GLOBAL_TEMPLATE = """# User Profile

## Communication Preferences

## Engineering Preferences

## UI Preferences

## Stable Personal Context

## Open Questions
"""

PROJECT_TEMPLATE = """# Project Memory

## Purpose

## Current Focus

## Architecture Decisions

## Design Decisions

## Constraints

## Important Discoveries

## Open Questions
"""


@dataclass
class ProposedMemory:
    id: str
    memory: str
    scope: str
    kind: str = FALLBACK_CATEGORY  # holds the category name


@dataclass
class ExtractResult:
    review_id: str
    global_memories: list[ProposedMemory] = field(default_factory=list)
    project_memories: list[ProposedMemory] = field(default_factory=list)


@dataclass
class StoredMemory:
    id: str
    text: str
    kind: str  # category name
    scope: str  # "global" | "project"


def _memory_extraction_model() -> str:
    model = os.environ.get("MEMORY_EXTRACTION_MODEL") or SMALL_MODEL
    if "/" in model:
        return model
    return f"ollama/{model}"


def _json_from_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped)


class MemoryStore:
    """Owns memory extraction, canonical document updates, and the pending buffer."""

    def __init__(self, base_dir: Path = DIRECTORY_CHAT_HISTORIES) -> None:
        self.base_dir = base_dir.resolve()
        self.memory_root = self.base_dir / "memory"
        self.pending_dir = self.memory_root / "pending"

    # ------------------------------------------------------------------
    # Public document helpers
    # ------------------------------------------------------------------

    def chat_context(
        self,
        project_slug: str | None = None,
        global_filepath: str | None = None,
        project_filepath: str | None = None,
    ) -> str:
        """Return canonical memory context for runtime chat injection."""
        global_path = self.resolve_path(global_filepath) if global_filepath else self._global_profile_path()

        if project_filepath:
            project_path = self.resolve_path(project_filepath)
        elif project_slug:
            project_path = self._project_memory_path(project_slug)
        else:
            project_path = None

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
        _llm: MemoryLLM,
        messages: list[dict[str, str]],
        project_slug: str | None = None,
    ) -> ExtractResult:
        """Extract categorized candidate memories with the configured small model."""
        self._ensure_dirs()
        review_id = str(uuid.uuid4())

        conversation = self._get_transcript(messages)
        global_block = self._format_categories(DEFAULT_GLOBAL_CATEGORIES)
        project_block = self._format_categories(DEFAULT_PROJECT_CATEGORIES)

        system_prompt = (
            "You extract atomic, durable candidate memories from a conversation for a "
            "user-reviewed learning system. Output valid JSON only, no prose. "
            "Each memory must be a single, self-contained fact or preference, tagged with "
            "exactly one of the allowed categories. Do not emit two memories that express "
            "the same idea; keep the more specific one. Empty lists are valid when there is "
            "nothing durable to remember."
        )

        project_section = ""
        if project_slug:
            project_section = (
                "\nProject categories (facts about the active project/subject):\n"
                f"{project_block}\n"
            )
        no_project_rule = (
            "" if project_slug else '\n- No project is active: return an empty "project" list.'
        )

        user_prompt = f"""
Conversation:
{conversation}

Extract durable memories and assign each to one allowed category.

Global categories (facts about the user, valid across all projects):
{global_block}
{project_section}
Return JSON with this exact shape:
{{
  "global": [{{"memory": "...", "category": "<one global category name>"}}],
  "project": [{{"memory": "...", "category": "<one project category name>"}}]
}}

Rules:
- "category" MUST be one of the allowed names listed above for that list.
- Global = stable user identity, preferences, and cross-project working style.
- Project = this project's/subject's purpose, concepts, decisions, constraints, resources, and open questions.{no_project_rule}
- Prefer a few high-value memories over many trivial ones.
- Never output two memories that say the same thing with different wording.
""".strip()

        data = await self._generate_json(_llm, system_prompt, user_prompt, timeout=EXTRACT_TIMEOUT)

        global_memories = self._build_candidates(data.get("global", []), "global", DEFAULT_GLOBAL_CATEGORIES)
        project_memories = (
            self._build_candidates(data.get("project", []), "project", DEFAULT_PROJECT_CATEGORIES)
            if project_slug else []
        )

        self._write_pending(review_id, global_memories + project_memories)
        return ExtractResult(review_id=review_id, global_memories=global_memories, project_memories=project_memories)

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

        self._backup_existing_profile(json_path)

        if revised_memories is not None:
            # Revised memories come from a preview and may carry an extra "status"
            # field for the UI — keep only the canonical StoredMemory fields.
            updated = [
                StoredMemory(id=m["id"], text=m["text"], kind=m["kind"], scope=m["scope"])
                for m in revised_memories
            ]
        else:
            existing = self._read_stored_memories(json_path)
            updated = await self.reconcile(llm, accepted_memories, existing, scope=scope)

        self._write_stored_memories(json_path, updated)
        md_content = self.render_memories_as_markdown(updated, title)
        self._write_doc(md_path, md_content)

        if review_id:
            self._delete_pending(review_id)

    async def cancel_async(self, review_id: str, project_slug: str | None = None) -> None:
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
        proposed_md = self.render_memories_as_markdown(updated, title)

        return {
            "existing_markdown": existing_md,
            "revised_markdown": proposed_md.rstrip() + "\n",
            "existing_memories": [vars(m) for m in existing_stored],
            "revised_memories": [{**vars(m), "status": status} for m, status in updated_pairs],
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
            existing_by_cat.setdefault(m.kind, []).append(m)

        candidates_by_cat: dict[str, list[ProposedMemory]] = {}
        for c in candidates:
            candidates_by_cat.setdefault(c.kind, []).append(c)

        result: list[tuple[StoredMemory, str]] = []

        # Untouched categories pass through verbatim.
        for cat, mems in existing_by_cat.items():
            if cat not in candidates_by_cat:
                result.extend((m, "pre-existing") for m in mems)

        # Touched categories get reconciled.
        for cat, cands in candidates_by_cat.items():
            cat_existing = existing_by_cat.get(cat, [])
            result.extend(await self._reconcile_category(llm, scope, cat, cat_existing, cands))

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
            final_texts = await self._merge_category_with_llm(
                llm, category, existing_texts, candidate_texts
            )

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
        """Build (StoredMemory, status) pairs, reusing existing ids for unchanged text."""
        by_text = {m.text.strip(): m for m in existing}
        existing_set = {t.lower() for t in existing_texts}
        candidate_set = {t.lower() for t in candidate_texts}
        out: list[tuple[StoredMemory, str]] = []
        for text in texts:
            prev = by_text.get(text)
            mem_id = prev.id if prev else str(uuid.uuid4())
            key = text.lower()
            if key in existing_set:
                status = "pre-existing"
            elif key in candidate_set:
                status = "new"
            else:
                status = "combined"
            out.append((StoredMemory(id=mem_id, text=text, kind=category, scope=scope), status))
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

    def _write_pending(self, review_id: str, candidates: list[ProposedMemory]) -> None:
        if not candidates:
            return
        self._ensure_dirs()
        path = self._pending_path(review_id)
        path.write_text(
            json.dumps([vars(c) for c in candidates], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _delete_pending(self, review_id: str) -> None:
        try:
            self._pending_path(review_id).unlink(missing_ok=True)
        except OSError as e:
            log.warning("Failed to delete pending buffer %s: %s", review_id, e)

    # ------------------------------------------------------------------
    # Profile management & versioning
    # ------------------------------------------------------------------

    def list_profiles(self, project_slug: str | None = None) -> list[dict[str, Any]]:
        """List all memory profile files (current and backups) under memory/ or <project_slug>/memory/."""
        dir_path = self._project_dir(project_slug) / "memory" if project_slug else self.memory_root

        if not dir_path.exists():
            return []

        profiles = []
        stem_name = "memory" if project_slug else "global-profile"

        active_file = dir_path / f"{stem_name}.md"
        if active_file.exists():
            profiles.append({
                "filepath": self._relative_base_path(active_file),
                "title": "Current",
            })

        backups = sorted(dir_path.glob(f"{stem_name}_*.md"))
        for file in backups:
            try:
                match = re.search(r"_(\d+)\.md$", file.name)
                idx_str = match.group(1) if match else "00"
                profiles.append({
                    "filepath": self._relative_base_path(file),
                    "title": f"Version {idx_str}",
                })
            except Exception as e:
                log.error("Failed to parse backup version %s: %s", file, e)

        return profiles

    def get_profile_content(self, filepath: str) -> str:
        """Retrieve complete content of the profile at filepath, creating it from template if missing."""
        path = self.resolve_path(filepath)
        if not path.exists():
            if filepath == "memory/global-profile.md" or filepath.endswith("global-profile.md"):
                return GLOBAL_TEMPLATE
            return PROJECT_TEMPLATE
        return path.read_text(encoding="utf-8")

    def _backup_existing_profile(self, path: Path) -> Path | None:
        """Back up current file to next available sequential backup name (e.g. memory_00.md) before overwrite."""
        if not path.exists():
            return None
        parent_dir = path.parent
        stem = path.stem
        suffix = path.suffix

        pattern = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(suffix)}$")
        max_idx = -1
        if parent_dir.exists():
            for f in parent_dir.glob(f"{stem}_*{suffix}"):
                match = pattern.match(f.name)
                if match:
                    max_idx = max(max_idx, int(match.group(1)))

        next_idx = max_idx + 1
        backup_name = f"{stem}_{next_idx:02d}{suffix}"
        backup_path = parent_dir / backup_name

        backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        return backup_path

    # ------------------------------------------------------------------
    # Canonical storage I/O & rendering
    # ------------------------------------------------------------------

    def _read_stored_memories(self, path: Path) -> list[StoredMemory]:
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        out: list[StoredMemory] = []
        for m in raw:
            try:
                out.append(StoredMemory(
                    id=str(m.get("id") or uuid.uuid4()),
                    text=str(m.get("text", "")),
                    kind=str(m.get("kind") or FALLBACK_CATEGORY),
                    scope=str(m.get("scope") or "global"),
                ))
            except AttributeError:
                continue
        return out

    def _write_stored_memories(self, path: Path, memories: list[StoredMemory]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([vars(m) for m in memories], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def render_memories_as_markdown(self, memories: list[StoredMemory], title: str) -> str:
        from collections import defaultdict
        groups: dict[str, list[str]] = defaultdict(list)
        for m in memories:
            groups[m.kind].append(m.text)
        parts = [f"# {title}"]
        for kind, items in groups.items():
            parts.append(f"\n## {kind.replace('_', ' ').title()}")
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

    def resolve_path(self, filepath: str) -> Path:
        """Resolve *filepath* under the base directory, rejecting traversal."""
        resolved = (self.base_dir / filepath).resolve()
        if not str(resolved).startswith(str(self.base_dir)):
            raise ValueError(f"Invalid path: {filepath}")
        return resolved

    def _ensure_dirs(self) -> None:
        self.memory_root.mkdir(parents=True, exist_ok=True)
        self.pending_dir.mkdir(parents=True, exist_ok=True)

    def _global_profile_path(self) -> Path:
        return self.memory_root / "global-profile.md"

    def _project_dir(self, project_slug: str | None) -> Path:
        if not project_slug:
            raise ValueError("Project slug is required")
        path = (self.base_dir / project_slug).resolve()
        try:
            path.relative_to(self.base_dir)
        except ValueError as e:
            raise ValueError("Invalid project slug") from e
        return path

    def _project_memory_path(self, project_slug: str | None) -> Path:
        return self._project_dir(project_slug) / PROJECT_MEMORY

    def _relative_base_path(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.base_dir))

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
            category = str(item.get("category") or item.get("kind") or "").strip()
            if category not in allowed:
                category = FALLBACK_CATEGORY
            candidates.append(ProposedMemory(id=str(uuid.uuid4()), memory=memory, scope=scope, kind=category))
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
                model=_memory_extraction_model(),
                user_msg=user_prompt,
                user_msg_history=[],
                system_prompt=system_prompt,
                stream=False,
            )
            if force_json:
                kwargs["response_format"] = {"type": "json_object"}
            response = llm.api_query(**kwargs)
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
