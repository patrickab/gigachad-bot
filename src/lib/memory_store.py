"""Document-backed memory store for user profile and project memory."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Protocol
import uuid

from config import DIRECTORY_CHAT_HISTORIES, SMALL_MODEL
from lib.legacy.mem0_memory_store import LegacyMem0MemoryStore

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path


class MemoryLLM(Protocol):
    def api_query(self, **kwargs: object) -> object: ...


MEMORY_ROOT = DIRECTORY_CHAT_HISTORIES / "memory"
PENDING_DIR = MEMORY_ROOT / "pending"
GLOBAL_PROFILE = MEMORY_ROOT / "global-profile.md"
PROJECT_MEMORY = "memory/memory.md"
MEMORY_LLM_TIMEOUT = 120
EXTRACT_TIMEOUT = 60

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


@dataclass
class ExtractResult:
    review_id: str
    global_memories: list[ProposedMemory] = field(default_factory=list)
    project_memories: list[ProposedMemory] = field(default_factory=list)


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
    # Extract candidate memories (Stateless)
    # ------------------------------------------------------------------

    async def extract(
        self,
        _llm: MemoryLLM,
        messages: list[dict[str, str]],
        project_slug: str | None = None,
    ) -> ExtractResult:
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
        conversation = self._get_transcript(messages)
        user_prompt = f"""
Conversation:
{conversation}

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
            timeout=EXTRACT_TIMEOUT,
        )
        raw_global.extend(data.get("global", []))
        if project_slug:
            raw_project.extend(data.get("project", []))

        pre_global = self._build_candidates(raw_global, "global")
        pre_project = self._build_candidates(raw_project, "project") if project_slug else []
        legacy_result = await self.mem0.store_pending(
            [m.memory for m in pre_global],
            [m.memory for m in pre_project],
            review_id,
            project_slug=project_slug,
        )
        global_memories = [self._from_mem0_memory(m, "global") for m in legacy_result.global_memories]
        project_memories = [self._from_mem0_memory(m, "project") for m in legacy_result.project_memories]

        return ExtractResult(review_id=review_id, global_memories=global_memories, project_memories=project_memories)

    # ------------------------------------------------------------------
    # Compose canonical document proposals (Stateless)
    # ------------------------------------------------------------------

    async def compose_stateless(
        self,
        llm: MemoryLLM,
        base_content: str,
        accepted_memories: list[dict[str, Any]],
        template: str,
        doc_name: str,
    ) -> str:
        """Compose revised markdown document body."""
        return await self._compose_document(llm, doc_name, base_content, accepted_memories, template)

    # ------------------------------------------------------------------
    # Commit / Cancel (Stateless & Idempotent)
    # ------------------------------------------------------------------

    async def commit_async(
        self,
        filepath: str,
        content: str,
        review_id: str | None = None,
        accepted_memories: list[dict[str, Any]] | None = None,
        rejected_memories: list[dict[str, Any]] | None = None,
    ) -> None:
        """Write final memory document, back up previous version, and sync with Mem0."""
        path = self.resolve_path(filepath)

        # Back up existing version if it exists
        self._backup_existing_profile(path)

        # Write clean markdown content directly
        self._write_doc(path, content)

        if review_id:
            if accepted_memories:
                for candidate in accepted_memories:
                    memory_id = str(candidate.get("id", ""))
                    if memory_id:
                        await self.mem0.accept_one(review_id, memory_id)
            if rejected_memories:
                for candidate in rejected_memories:
                    memory_id = str(candidate.get("id", ""))
                    if memory_id:
                        await self.mem0.cancel_one(review_id, memory_id)

    async def cancel_async(self, review_id: str, project_slug: str | None = None) -> None:
        """Cancel the outstanding candidates session in Mem0."""
        await self.mem0.cancel(review_id, project_slug=project_slug)

    # ------------------------------------------------------------------
    # Tree branching & profile management
    # ------------------------------------------------------------------

    def list_profiles(self, project_slug: str | None = None) -> list[dict[str, Any]]:
        """List all memory profile files (current and backups) under memory/ or <project_slug>/memory/."""
        dir_path = self._project_dir(project_slug) / "memory" if project_slug else self.memory_root

        if not dir_path.exists():
            return []

        profiles = []
        stem_name = "memory" if project_slug else "global-profile"

        # 1. Add current active version
        active_file = dir_path / f"{stem_name}.md"
        if active_file.exists():
            rel_path = self._relative_base_path(active_file)
            profiles.append(
                {
                    "filepath": rel_path,
                    "id": "current",
                    "title": "Current Active",
                    "updated_at": datetime.fromtimestamp(active_file.stat().st_mtime, tz=timezone.utc).isoformat(),
                }
            )

        # 2. Add sequential backup files
        backups = sorted(dir_path.glob(f"{stem_name}_*.md"))
        for file in backups:
            try:
                rel_path = self._relative_base_path(file)
                match = re.search(r"_(\d+)\.md$", file.name)
                idx_str = match.group(1) if match else "00"
                profiles.append(
                    {
                        "filepath": rel_path,
                        "id": f"v{idx_str}",
                        "title": f"Previous Version {idx_str}",
                        "updated_at": datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc).isoformat(),
                    }
                )
            except Exception as e:
                log.error("Failed to parse backup version %s: %s", file, e)

        return profiles

    def get_profile_content(self, filepath: str) -> str:
        """Retrieve complete content of the profile at filepath, creating it from template if missing."""
        path = self.resolve_path(filepath)
        if not path.exists():
            if filepath == "memory/global-profile.md" or filepath.endswith("global-profile.md"):
                return GLOBAL_TEMPLATE
            else:
                return PROJECT_TEMPLATE
        return path.read_text(encoding="utf-8")

    def _backup_existing_profile(self, path: Path) -> Path | None:
        """Back up current file to next available sequential backup name (e.g. memory_00.md) before overwrite."""
        if not path.exists():
            return None
        parent_dir = path.parent
        stem = path.stem  # e.g., "memory" or "global-profile"
        suffix = path.suffix  # e.g., ".md"

        # Find next available idx
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
    # Internal I/O
    # ------------------------------------------------------------------

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

    def _relative_base_path(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.base_dir))

    def _read_existing_doc(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _write_doc(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")

    def _build_candidates(self, raw_items: object, scope: str) -> list[ProposedMemory]:
        if not isinstance(raw_items, list):
            return []
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
            kind = str(item.get("kind", "note")).strip() or "note"
            candidates.append(ProposedMemory(id=str(uuid.uuid4()), memory=memory, scope=scope, kind=kind))
        return candidates

    def _from_mem0_memory(self, memory: object, scope: str) -> ProposedMemory:
        memory_id = str(getattr(memory, "id", ""))
        text = str(getattr(memory, "memory", ""))
        categories = list(getattr(memory, "categories", []) or [])
        kind = categories[0] if categories else "note"
        return ProposedMemory(id=memory_id, memory=text, scope=scope, kind=kind)

    def _get_transcript(self, messages: list[dict[str, str]]) -> str:
        blocks: list[str] = []
        for message in messages:
            role = message.get("role", "user").upper()
            content = message.get("content", "").strip()
            blocks.append(f"{role}: {content}")
        transcript = "\n\n".join(blocks)
        return transcript or "No conversation content."

    async def _compose_document(
        self,
        llm: MemoryLLM,
        doc_name: str,
        current_doc: str,
        candidates: list[dict[str, Any]],
        template: str,
    ) -> str:
        system_prompt = (
            "You maintain concise canonical markdown memory documents."
            "Your main duty is to incorporate new accepted atomic memories into the current document "
            "Replace outdated or conflicting statements. For example <...>\n"
            "Rules for merging:\n"
            "1. Preserve existing items, sections, and statements that are not directly contradicted or made obsolete by new accepted memories.\n"
            "2. DO NOT delete, summarize away, omit, or clean up existing unrelated information.\n"
            "3. If a new memory conflicts with or updates an existing item, replace or edit that specific item.\n"
            "4. If a conflict is ambiguous, preserve the current statement and add a short item under Open Questions.\n"
            "5. Output valid JSON only."
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
            - Preserve existing statements & sections in the 'Current markdown document'. Do NOT delete any unless directly contradicted by 'Accepted atomic memories'.
            - Keep the document sectioned and easy to scan.
            - Do not include provenance chatter or mention this prompt.
            - The returned document must be the complete, updated markdown document containing both the preserved existing information and the newly added memories. It must not be a patch or a partial document.
            """.strip()
        data = await self._generate_json(llm, system_prompt, user_prompt)
        return str(data.get("document", ""))

    async def _generate_json(
        self,
        llm: MemoryLLM,
        system_prompt: str,
        user_prompt: str,
        timeout: int = MEMORY_LLM_TIMEOUT,
    ) -> dict[str, Any]:
        def call() -> str:
            response = llm.api_query(
                model=_memory_extraction_model(),
                user_msg=user_prompt,
                user_msg_history=[],
                system_prompt=system_prompt,
                stream=False,
            )
            return response.choices[0].message.content or "{}"

        try:
            return _json_from_response(await asyncio.wait_for(asyncio.to_thread(call), timeout=timeout))
        except Exception as e:
            log.error("Memory LLM JSON call failed: %s", e)
            raise RuntimeError(f"Memory model did not return valid JSON: {e}") from e
