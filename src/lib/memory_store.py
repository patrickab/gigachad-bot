"""Document-backed memory store for user profile and project memory."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import difflib
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Protocol
import uuid

from config import DIRECTORY_CHAT_HISTORIES, SMALL_MODEL
from lib.json_io import safe_read_json, safe_write_json
from lib.legacy.mem0_memory_store import LegacyMem0MemoryStore

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path


class MemoryLLM(Protocol):
    def api_query(self, **kwargs: object) -> object: ...

MEMORY_ROOT = DIRECTORY_CHAT_HISTORIES / "memory"
PENDING_DIR = MEMORY_ROOT / "pending"
GLOBAL_PROFILE = MEMORY_ROOT / "global-profile.md"
GLOBAL_LOG = MEMORY_ROOT / "global-log.jsonl"
PROJECT_MEMORY = "memory.md"
PROJECT_LOG = "memory-log.jsonl"
COMPOSE_NUM_PREDICT = 4096
MEMORY_LLM_TIMEOUT = 120
EXTRACT_NUM_PREDICT: int | None = None
EXTRACT_TIMEOUT = 60
EXTRACT_TRANSCRIPT_CHARS = 3000
EXTRACT_MESSAGE_SLICE_CHARS = 240

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
    kind: str = "note"
    reason: str = ""
    categories: list[str] = field(default_factory=list)


@dataclass
class ExtractResult:
    review_id: str
    global_memories: list[ProposedMemory] = field(default_factory=list)
    project_memories: list[ProposedMemory] = field(default_factory=list)


@dataclass
class ComposeResult:
    review_id: str
    global_document: str | None = None
    project_document: str | None = None
    global_diff: str | None = None
    project_diff: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_model() -> str:
    if "/" in SMALL_MODEL:
        return SMALL_MODEL
    return f"ollama/{SMALL_MODEL}"


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


def _unified_diff(old: str, new: str, fromfile: str, tofile: str) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    return "".join(difflib.unified_diff(old_lines, new_lines, fromfile=fromfile, tofile=tofile, lineterm=""))


class MemoryStore:
    """Owns memory extraction, canonical document updates, and memory logs."""

    def __init__(self, base_dir: Path = DIRECTORY_CHAT_HISTORIES) -> None:
        self.base_dir = base_dir.resolve()
        self.memory_root = self.base_dir / "memory"
        self.pending_dir = self.memory_root / "pending"
        self._mem0: LegacyMem0MemoryStore | None = None

    @property
    def mem0(self) -> LegacyMem0MemoryStore:
        if self._mem0 is None:
            self._mem0 = LegacyMem0MemoryStore()
        return self._mem0

    # ------------------------------------------------------------------
    # Public document helpers
    # ------------------------------------------------------------------

    def chat_context(self, project_slug: str | None = None) -> str:
        """Return canonical memory context for runtime chat injection."""
        global_profile = self._read_existing_doc(self._global_profile_path()).strip()
        project_memory = ""
        if project_slug:
            project_memory = self._read_existing_doc(self._project_memory_path(project_slug)).strip()

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
    # Extract candidate memories
    # ------------------------------------------------------------------

    async def extract(self, _llm: MemoryLLM, messages: list[dict[str, str]], project_slug: str | None = None) -> ExtractResult:
        """Extract candidates with the configured small model and persist pending facts in Mem0."""
        self._ensure_dirs()
        review_id = str(uuid.uuid4())
        raw_global: list[object] = []
        raw_project: list[object] = []

        system_prompt = (
            "You extract atomic candidate memories for a user-reviewed learning system. "
            "Only propose concise durable memory points. Output valid JSON only. "
            "When a project is active, technical/product/design decisions belong to project, not global."
        )
        conversation = self._compact_transcript(messages)
        if not conversation:
            conversation = "No conversation content."
        for idx, conversation_chunk in enumerate([conversation], start=1):
            user_prompt = f"""
Conversation chunk {idx}:
{conversation_chunk}

Return JSON with this exact shape:
{{
  "global": [
    {{"memory": "...", "kind": "preference|identity|working_style|ui_preference|engineering_preference|note"}}
  ],
  "project": [
    {{"memory": "...", "kind": "purpose|current_focus|design_decision|architecture_decision|constraint|discovery|open_question|note"}}
  ]
}}

Rules:
- Global memories are only stable user preferences, identity, and cross-project working style.
- Project memories are active project purpose, technical decisions, design decisions, constraints, discoveries, and current focus.
- If a memory is about this app, this repo, the memory layer, architecture, implementation, UI, routes,
  APIs, storage, or prompts, classify it as project.
- Do not put project-specific architecture or product strategy into global.
- Prefer high-value candidates only. Empty lists are valid.
""".strip()
            data = await self._generate_json(
                _llm,
                system_prompt,
                user_prompt,
                num_predict=EXTRACT_NUM_PREDICT,
                timeout=EXTRACT_TIMEOUT,
                model=_memory_extraction_model(),
            )
            raw_global.extend(data.get("global", []))
            if project_slug:
                raw_project.extend(data.get("project", []))

        pre_global = self._build_candidates(raw_global, "global")
        pre_project = self._build_candidates(raw_project, "project") if project_slug else []
        if project_slug:
            pre_global, moved_project = self._rebalance_project_scoped_globals(pre_global)
            pre_project = [*pre_project, *moved_project]
        legacy_result = await self.mem0.store_pending(
            [m.memory for m in pre_global],
            [m.memory for m in pre_project],
            review_id,
            project_slug=project_slug,
        )
        global_memories = [self._from_mem0_memory(m, "global") for m in legacy_result.global_memories]
        project_memories = [self._from_mem0_memory(m, "project") for m in legacy_result.project_memories]
        self._write_pending(
            review_id,
            {
                "review_id": review_id,
                "project_slug": project_slug,
                "created_at": _now_iso(),
                "candidates": [m.__dict__ for m in [*global_memories, *project_memories]],
                "accepted_ids": [],
                "composed_at": None,
                "status": "pending",
            },
        )
        return ExtractResult(review_id=review_id, global_memories=global_memories, project_memories=project_memories)

    # ------------------------------------------------------------------
    # Compose canonical document proposals
    # ------------------------------------------------------------------

    async def compose(
        self,
        llm: MemoryLLM,
        review_id: str,
        accepted_ids: list[str],
        project_slug: str | None = None,
        manual_memories: list[dict[str, Any]] | None = None,
    ) -> ComposeResult:
        """Turn accepted atomic memories into revised markdown documents."""
        pending = self._read_pending(review_id)
        status = pending.get("status", "pending")
        if status in ("committed", "cancelled"):
            raise ValueError(f"Cannot compose: Memory review has already been {status}")
        if project_slug != pending.get("project_slug"):
            project_slug = pending.get("project_slug")
        existing_candidates = list(pending.get("candidates", []))
        existing_ids = {str(c.get("id", "")) for c in existing_candidates}
        manual_candidates = self._manual_candidates(manual_memories or [], existing_ids, project_slug)
        if manual_candidates:
            pending["candidates"] = [*existing_candidates, *manual_candidates]
        candidates = [c for c in [*existing_candidates, *manual_candidates] if c.get("id") in set(accepted_ids)]
        pending["accepted_ids"] = [c.get("id") for c in candidates]
        pending["composed_at"] = _now_iso()

        global_candidates = [c for c in candidates if c.get("scope") == "global"]
        project_candidates = [c for c in candidates if c.get("scope") == "project"]

        old_global = self._read_doc(self._global_profile_path(), GLOBAL_TEMPLATE)
        old_project = self._read_doc(self._project_memory_path(project_slug), PROJECT_TEMPLATE) if project_slug else None
        new_global = old_global
        new_project = old_project

        if global_candidates:
            new_global = await self._compose_document(llm, "global user profile", old_global, global_candidates, GLOBAL_TEMPLATE)
            pending["global_document"] = new_global
        if project_slug and project_candidates:
            assert old_project is not None
            new_project = await self._compose_document(llm, "project memory", old_project, project_candidates, PROJECT_TEMPLATE)
            pending["project_document"] = new_project

        self._write_pending(review_id, pending)
        return ComposeResult(
            review_id=review_id,
            global_document=new_global,
            project_document=new_project,
            global_diff=(
                _unified_diff(old_global, new_global, "global-profile.md", "global-profile.md")
                if global_candidates
                else None
            ),
            project_diff=(
                _unified_diff(old_project or "", new_project or "", "memory.md", "memory.md")
                if project_candidates
                else None
            ),
        )

    def _manual_candidates(
        self,
        manual_memories: list[dict[str, Any]],
        existing_ids: set[str],
        project_slug: str | None,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for item in manual_memories:
            memory_id = str(item.get("id", "")).strip()
            memory = str(item.get("memory", "")).strip()
            scope = str(item.get("scope", "global")).strip()
            if scope not in {"global", "project"}:
                scope = "global"
            if scope == "project" and not project_slug:
                continue
            if not memory_id or memory_id in existing_ids or not memory:
                continue
            kind = str(item.get("kind") or "manual").strip() or "manual"
            reason = str(item.get("reason") or "Added manually").strip()
            categories = item.get("categories")
            candidates.append(
                {
                    "id": memory_id,
                    "memory": memory,
                    "scope": scope,
                    "kind": kind,
                    "reason": reason,
                    "categories": categories if isinstance(categories, list) else [kind],
                }
            )
            existing_ids.add(memory_id)
        return candidates

    async def _compose_document(
        self,
        llm: MemoryLLM,
        doc_name: str,
        current_doc: str,
        candidates: list[dict[str, Any]],
        template: str,
    ) -> str:
        system_prompt = (
            "You maintain concise canonical markdown memory documents. "
            "Incorporate accepted atomic memories into the current document. "
            "Replace outdated or conflicting statements instead of appending duplicates. "
            "If a conflict is ambiguous, preserve the current statement and add a short item under Open Questions. "
            "Output valid JSON only."
        )
        user_prompt = f"""
Document type: {doc_name}

Current markdown document:
{current_doc or template}

Accepted atomic memories:
{json.dumps(candidates, indent=2)}

Return JSON with exactly this shape:
{{"document": "# ... revised full markdown document ..."}}

Rules:
- Preserve useful existing context.
- Keep the document sectioned and easy to scan.
- Do not include provenance chatter or mention this prompt.
- Do not add a memory unless it is supported by the accepted atomic memories or already present.
- The returned document must be complete markdown, not a patch.
""".strip()
        data = await self._generate_json(llm, system_prompt, user_prompt, num_predict=COMPOSE_NUM_PREDICT)
        document = str(data.get("document", "")).strip()
        return document + "\n" if document else current_doc

    # ------------------------------------------------------------------
    # Commit / cancel
    # ------------------------------------------------------------------

    def commit(
        self,
        review_id: str,
        global_document: str | None = None,
        project_document: str | None = None,
        project_slug: str | None = None,
    ) -> None:
        raise NotImplementedError("Use commit_async")

    async def commit_async(
        self,
        review_id: str,
        global_document: str | None = None,
        project_document: str | None = None,
        project_slug: str | None = None,
    ) -> None:
        pending = self._read_pending(review_id)
        status = pending.get("status", "pending")
        if status == "committed":
            return
        if status == "cancelled":
            raise ValueError("Memory review was already cancelled")

        if project_slug != pending.get("project_slug"):
            project_slug = pending.get("project_slug")
        accepted_ids = set(pending.get("accepted_ids", []))
        candidates = [c for c in pending.get("candidates", []) if c.get("id") in accepted_ids]

        if global_document is not None:
            self._write_doc(self._global_profile_path(), global_document)
        if project_slug and project_document is not None:
            self._write_doc(self._project_memory_path(project_slug), project_document)

        for candidate in candidates:
            if candidate.get("scope") == "project" and project_slug:
                log_path = self._project_log_path(project_slug)
                doc_path = self._relative_base_path(self._project_memory_path(project_slug))
            else:
                log_path = self._global_log_path()
                doc_path = self._relative_base_path(self._global_profile_path())
            self._append_memory_log(log_path, review_id, candidate, doc_path)

        for candidate in pending.get("candidates", []):
            memory_id = str(candidate.get("id", ""))
            if not memory_id:
                continue
            if memory_id in accepted_ids:
                await self.mem0.accept_one(review_id, memory_id)
            else:
                await self.mem0.cancel_one(review_id, memory_id)

        pending["status"] = "committed"
        self._write_pending(review_id, pending)

    def cancel(self, review_id: str) -> None:
        raise NotImplementedError("Use cancel_async")

    async def cancel_async(self, review_id: str, project_slug: str | None = None) -> None:
        pending = self._read_pending(review_id)
        status = pending.get("status", "pending")
        if status == "cancelled":
            return
        if status == "committed":
            raise ValueError("Memory review was already committed")

        await self.mem0.cancel(review_id, project_slug=project_slug)
        pending["status"] = "cancelled"
        self._write_pending(review_id, pending)

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        self.memory_root.mkdir(parents=True, exist_ok=True)
        self.pending_dir.mkdir(parents=True, exist_ok=True)

    def _global_profile_path(self) -> Path:
        return self.memory_root / "global-profile.md"

    def _global_log_path(self) -> Path:
        return self.memory_root / "global-log.jsonl"

    def _project_dir(self, project_slug: str) -> Path:
        path = (self.base_dir / project_slug).resolve()
        try:
            path.relative_to(self.base_dir)
        except ValueError as e:
            raise ValueError("Invalid project slug") from e
        return path

    def _project_memory_path(self, project_slug: str | None) -> Path:
        if not project_slug:
            raise ValueError("Project slug is required")
        return self._project_dir(project_slug) / PROJECT_MEMORY

    def _project_log_path(self, project_slug: str) -> Path:
        return self._project_dir(project_slug) / PROJECT_LOG

    def _relative_base_path(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.base_dir))

    def _pending_path(self, review_id: str) -> Path:
        if "/" in review_id or ".." in review_id:
            raise ValueError("Invalid review id")
        return self.pending_dir / f"{review_id}.json"

    def _read_doc(self, path: Path, template: str) -> str:
        if not path.exists():
            return template
        return path.read_text(encoding="utf-8")

    def _read_existing_doc(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _write_doc(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")

    def _write_pending(self, review_id: str, data: dict[str, Any]) -> None:
        self._ensure_dirs()
        safe_write_json(self._pending_path(review_id), data)

    def _read_pending(self, review_id: str) -> dict[str, Any]:
        data = safe_read_json(self._pending_path(review_id), {})
        if not data:
            raise ValueError("Memory review not found")
        return data

    def _append_memory_log(self, path: Path, review_id: str, candidate: dict[str, Any], doc_path: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate_id = str(candidate.get("id", ""))
        entry_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{review_id}:{candidate_id}"))

        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                if f'"id": "{entry_id}"' in content:
                    # Already logged, skip appending to prevent duplication on retry
                    return
            except Exception:
                pass

        parent_id = self._last_log_id(path)
        entry = {
            "id": entry_id,
            "parent_id": parent_id,
            "review_id": review_id,
            "candidate_id": candidate_id,
            "memory": candidate.get("memory", ""),
            "kind": candidate.get("kind", "note"),
            "reason": candidate.get("reason", ""),
            "scope": candidate.get("scope", "global"),
            "doc_path": doc_path,
            "created_at": _now_iso(),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")

    def _last_log_id(self, path: Path) -> str | None:
        if not path.exists():
            return None
        last_id = None
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            last_id = entry.get("id") or last_id
        return last_id

    def _build_candidates(self, raw_items: object, scope: str) -> list[ProposedMemory]:
        if not isinstance(raw_items, list):
            return []
        candidates: list[ProposedMemory] = []
        seen: set[str] = set()
        for item in raw_items:
            if isinstance(item, str):
                memory = item.strip()
                kind = "note"
                reason = ""
            elif isinstance(item, dict):
                memory = str(item.get("memory", "")).strip()
                kind = str(item.get("kind", "note")).strip() or "note"
                reason = str(item.get("reason", "")).strip()
            else:
                continue
            key = memory.lower()
            if not memory or key in seen:
                continue
            seen.add(key)
            candidates.append(
                ProposedMemory(
                    id=str(uuid.uuid4()),
                    memory=memory,
                    scope=scope,
                    kind=kind,
                    reason=reason,
                    categories=[kind] if kind else [],
                )
            )
        return candidates

    def _rebalance_project_scoped_globals(self, memories: list[ProposedMemory]) -> tuple[list[ProposedMemory], list[ProposedMemory]]:
        global_memories: list[ProposedMemory] = []
        project_memories: list[ProposedMemory] = []
        for memory in memories:
            if self._looks_project_scoped(memory.memory):
                project_memories.append(
                    ProposedMemory(
                        id=memory.id,
                        memory=memory.memory,
                        scope="project",
                        kind="note" if memory.kind in {"preference", "identity", "working_style"} else memory.kind,
                        reason=memory.reason,
                        categories=memory.categories,
                    )
                )
            else:
                global_memories.append(memory)
        return global_memories, project_memories

    def _looks_project_scoped(self, text: str) -> bool:
        lower = text.lower()
        user_markers = (
            "user prefers",
            "user likes",
            "user wants",
            "user values",
            "user's preferred",
            "communication preference",
            "working style",
        )
        if any(marker in lower for marker in user_markers):
            return False
        project_markers = (
            "project",
            "repo",
            "codebase",
            "app",
            "architecture",
            "implementation",
            "design decision",
            "technical decision",
            "memory system",
            "memory layer",
            "backend",
            "frontend",
            "route",
            "api",
            "storage",
            "qdrant",
            "mem0",
            "prompt",
            "ui",
            "document",
        )
        return any(marker in lower for marker in project_markers)

    def _from_mem0_memory(self, memory: object, scope: str) -> ProposedMemory:
        memory_id = str(getattr(memory, "id", ""))
        text = str(getattr(memory, "memory", ""))
        categories = list(getattr(memory, "categories", []) or [])
        kind = categories[0] if categories else "note"
        return ProposedMemory(id=memory_id, memory=text, scope=scope, kind=kind, categories=categories)

    def _compact_transcript(self, messages: list[dict[str, str]]) -> str:
        blocks: list[str] = []
        for message in messages:
            role = message.get("role", "user").upper()
            content = message.get("content", "").strip()
            if len(content) > EXTRACT_MESSAGE_SLICE_CHARS * 2:
                content = f"{content[:EXTRACT_MESSAGE_SLICE_CHARS]}\n...[snip]...\n{content[-EXTRACT_MESSAGE_SLICE_CHARS:]}"
            blocks.append(f"{role}: {content}")

        transcript = "\n\n".join(blocks)
        if len(transcript) <= EXTRACT_TRANSCRIPT_CHARS:
            return transcript
        half = EXTRACT_TRANSCRIPT_CHARS // 2
        return transcript[:half] + "\n\n...[middle omitted]...\n\n" + transcript[-half:]

    async def _generate_json(
        self,
        llm: MemoryLLM,
        system_prompt: str,
        user_prompt: str,
        num_predict: int | None,
        timeout: int = MEMORY_LLM_TIMEOUT,
        model: str | None = None,
    ) -> dict[str, Any]:
        model = model or _memory_model()
        def call(use_response_format: bool) -> str:
            kwargs: dict[str, object] = {
                "model": model,
                "user_msg": user_prompt,
                "user_msg_history": [],
                "system_prompt": system_prompt,
                "stream": False,
            }
            if num_predict is not None:
                kwargs["max_tokens"] = num_predict
            if use_response_format:
                kwargs["response_format"] = {"type": "json_object"}
            response = llm.api_query(**kwargs)
            return response.choices[0].message.content or "{}"

        try:
            return _json_from_response(await asyncio.wait_for(asyncio.to_thread(call, True), timeout=timeout))
        except Exception as e:
            log.warning("Memory LLM JSON response_format failed, retrying plain JSON prompt: %s", e)
        try:
            return _json_from_response(await asyncio.wait_for(asyncio.to_thread(call, False), timeout=timeout))
        except Exception as e:
            log.error("Memory LLM JSON call failed: %s", e)
            raise RuntimeError(f"Memory model did not return valid JSON: {e}") from e
