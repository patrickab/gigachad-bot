"""Single seam for all memory I/O, extraction, and review operations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import logging
import os
from typing import Any
import uuid

from mem0 import AsyncMemory

from config import MEM0_DATA_DIR, MEM0_EMBEDDING_MODEL, MEM0_USER_ID, OLLAMA_BASE_URL, SMALL_MODEL
from lib.prompts.memory import MEM0_GLOBAL_PROMPT, MEM0_PROJECT_PROMPT

log = logging.getLogger(__name__)


def _parse_model_string(model_str: str) -> tuple[str, str]:
    if "/" in model_str:
        return model_str.split("/", 1)
    return "ollama", model_str


def _get_mem0_provider(p_env: str, m_env: str, fallback_str: str) -> dict:
    p = os.environ.get(p_env)
    m = os.environ.get(m_env)
    if not p or not m:
        def_p, def_m = _parse_model_string(fallback_str)
        p = p or def_p
        m = m or def_m

    cfg = {"provider": p, "config": {"model": m}}
    if p == "ollama":
        cfg["config"]["ollama_base_url"] = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    elif p == "openai" and os.environ.get("OPENAI_API_BASE"):
        cfg["config"]["openai_api_base"] = os.environ.get("OPENAI_API_BASE")
    elif p == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if key:
            cfg["config"]["api_key"] = key
    return cfg


_llm = _get_mem0_provider("MEM0_PROVIDER", "MEM0_MODEL", SMALL_MODEL)
_llm["config"].setdefault("temperature", 0.1)

_emb = _get_mem0_provider("MEM0_EMBEDDER_PROVIDER", "MEM0_EMBEDDER_MODEL", MEM0_EMBEDDING_MODEL)

_EMBED_DIMS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 768,
    "all-MiniLM-L6-v2": 384,
    "bge-m3": 1024,
}
_emb_dims = _EMBED_DIMS.get(_emb["config"]["model"], 768 if _emb["provider"] in ("gemini", "ollama") else 1536)

MEM0_CONFIG = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "path": str(MEM0_DATA_DIR / "memory_db"),
            "embedding_model_dims": _emb_dims,
        },
    },
    "llm": _llm,
    "embedder": _emb,
    "history_db_path": str(MEM0_DATA_DIR / "chat_history_db"),
}


@dataclass
class ProposedMemory:
    id: str
    memory: str
    categories: list[str] = field(default_factory=list)


@dataclass
class ExtractResult:
    review_id: str
    global_memories: list[ProposedMemory] = field(default_factory=list)
    project_memories: list[ProposedMemory] = field(default_factory=list)


class MemoryStore:
    """Owns all memory extraction, review, and persistence operations."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or MEM0_CONFIG
        self._memory = AsyncMemory.from_config(self._config)

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------

    async def _extract_memories_for_prompt(
        self,
        messages: list[dict[str, str]],
        instruction_prompt: str,
    ) -> list[str]:
        """Extract a list of facts/memories from the conversation using the LLM."""
        formatted_convo = "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
            for m in messages
        )

        system_prompt = (
            "You are a precise facts-extraction assistant.\n"
            "Given a conversation history, extract durable facts according to the instructions.\n"
            "You MUST output a valid JSON object with a single key \"memories\" containing a list of strings.\n"
            "Do not include any explanation or other fields.\n"
            "Example output:\n"
            "{\n"
            "  \"memories\": [\"User's name is Bob\", \"User is a software engineer\"]\n"
            "}"
        )
        user_prompt = (
            f"Instructions: {instruction_prompt}\n\n"
            f"Conversation:\n{formatted_convo}\n\n"
            "Extract durable facts and output them as a JSON object with a \"memories\" key."
        )

        try:
            response_text = await asyncio.to_thread(
                self._memory.llm.generate_response,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            data = json.loads(response_text)
            return data.get("memories", [])
        except Exception as e:
            log.error(f"Failed to extract memories: {e}")
            return []

    async def extract(
        self,
        messages: list[dict[str, str]],
        project_slug: str | None = None,
    ) -> ExtractResult:
        """Add conversation to Mem0 with pending status and return proposed memories."""
        review_id = str(uuid.uuid4())
        metadata = {"status": "pending", "review_id": review_id}

        # Global extraction
        global_facts = await self._extract_memories_for_prompt(messages, MEM0_GLOBAL_PROMPT)
        for fact in global_facts:
            await self._memory.add(
                fact,
                user_id=MEM0_USER_ID,
                metadata=metadata,
                infer=False,
            )
        global_memories = await self._fetch_pending(MEM0_USER_ID, None, review_id)

        # Project-scoped extraction (if a project is active)
        project_memories: list[ProposedMemory] = []
        if project_slug:
            project_facts = await self._extract_memories_for_prompt(messages, MEM0_PROJECT_PROMPT)
            for fact in project_facts:
                await self._memory.add(
                    fact,
                    user_id=MEM0_USER_ID,
                    agent_id=project_slug,
                    metadata=metadata,
                    infer=False,
                )
            project_memories = await self._fetch_pending(MEM0_USER_ID, project_slug, review_id)

        return ExtractResult(
            review_id=review_id,
            global_memories=global_memories,
            project_memories=project_memories,
        )

    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    async def accept(self, review_id: str, project_slug: str | None = None) -> None:
        """Promote all pending memories to active status."""
        await self._update_pending(MEM0_USER_ID, None, review_id, "active")
        if project_slug:
            await self._update_pending(MEM0_USER_ID, project_slug, review_id, "active")

    async def accept_one(self, review_id: str, memory_id: str) -> None:
        """Promote one pending memory to active status."""
        await self._update_memory_if_pending(MEM0_USER_ID, review_id, memory_id, "active")

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    async def cancel(self, review_id: str, project_slug: str | None = None) -> None:
        """Delete all pending memories."""
        await self._delete_pending(MEM0_USER_ID, None, review_id)
        if project_slug:
            await self._delete_pending(MEM0_USER_ID, project_slug, review_id)

    async def cancel_one(self, review_id: str, memory_id: str) -> None:
        """Delete one pending memory."""
        await self._delete_memory_if_pending(MEM0_USER_ID, review_id, memory_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_pending(
        self,
        user_id: str,
        agent_id: str | None,
        review_id: str,
    ) -> list[ProposedMemory]:
        """Fetch all memories with status='pending' for the given namespace."""
        filters: dict[str, Any] = {"user_id": user_id}
        if agent_id:
            filters["agent_id"] = agent_id

        all_memories = await self._memory.get_all(filters=filters)
        results = all_memories.get("results", [])

        pending: list[ProposedMemory] = []
        for entry in results:
            meta = entry.get("metadata", {}) or {}
            if meta.get("status") == "pending" and meta.get("review_id") == review_id:
                # Match scope: global should have agent_id=None, project should match agent_id
                entry_agent_id = entry.get("agent_id")
                if agent_id is None and entry_agent_id is not None:
                    continue
                if agent_id is not None and entry_agent_id != agent_id:
                    continue

                pending.append(
                    ProposedMemory(
                        id=entry.get("id", ""),
                        memory=entry.get("memory", ""),
                        categories=entry.get("categories", []) or [],
                    )
                )
        return pending

    async def _update_pending(
        self,
        user_id: str,
        agent_id: str | None,
        review_id: str,
        new_status: str,
    ) -> None:
        """Update all pending memories to a new status."""
        pending = await self._fetch_pending(user_id, agent_id, review_id)
        for mem in pending:
            existing_memory = await asyncio.to_thread(self._memory.vector_store.get, vector_id=mem.id)
            if existing_memory:
                new_payload = dict(existing_memory.payload)
                new_payload["status"] = new_status
                await asyncio.to_thread(
                    self._memory.vector_store.update,
                    vector_id=mem.id,
                    payload=new_payload
                )

    async def _update_memory_if_pending(
        self,
        user_id: str,
        review_id: str,
        memory_id: str,
        new_status: str,
    ) -> None:
        """Update one memory only if it belongs to the pending review."""
        existing_memory = await asyncio.to_thread(self._memory.vector_store.get, vector_id=memory_id)
        if not existing_memory:
            return
        payload = dict(existing_memory.payload)
        if payload.get("user_id") != user_id or payload.get("status") != "pending" or payload.get("review_id") != review_id:
            return
        payload["status"] = new_status
        await asyncio.to_thread(
            self._memory.vector_store.update,
            vector_id=memory_id,
            payload=payload,
        )

    async def _delete_pending(
        self,
        user_id: str,
        agent_id: str | None,
        review_id: str,
    ) -> None:
        """Delete all pending memories."""
        pending = await self._fetch_pending(user_id, agent_id, review_id)
        for mem in pending:
            await asyncio.to_thread(self._memory.vector_store.delete, vector_id=mem.id)

    async def _delete_memory_if_pending(
        self,
        user_id: str,
        review_id: str,
        memory_id: str,
    ) -> None:
        """Delete one memory only if it belongs to the pending review."""
        existing_memory = await asyncio.to_thread(self._memory.vector_store.get, vector_id=memory_id)
        if not existing_memory:
            return
        payload = existing_memory.payload
        if payload.get("user_id") != user_id or payload.get("status") != "pending" or payload.get("review_id") != review_id:
            return
        await asyncio.to_thread(self._memory.vector_store.delete, vector_id=memory_id)
