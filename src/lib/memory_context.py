"""Inject every stored fact into chat system prompts, most relevant first."""

from __future__ import annotations

from lib.memory_store import MemoryScope, MemoryStore, ScopeKind

HEADINGS: dict[ScopeKind, str] = {"global": "About the user", "project": "Project memory"}


def topic_title(name: str) -> str:
    return name.replace("_", " ").title()


def augment_system_prompt(store: MemoryStore, system_prompt: str, project_slug: str | None = None) -> str:
    """Append global and project memory to *system_prompt* under a "Persistent Memory Context" heading."""
    memory_context = render_memory_context(store, project_slug)
    if not memory_context:
        return system_prompt
    if not system_prompt.strip():
        return memory_context
    return f"{system_prompt.rstrip()}\n\n# Persistent Memory Context\n\n{memory_context}"


def render_memory_context(store: MemoryStore, project_slug: str | None) -> str:
    """Render each scope's facts in stored order, which is their relevance order (most relevant first)."""
    scopes = [MemoryScope("global")]
    if project_slug:
        scopes.append(MemoryScope("project", project_slug))

    parts: list[str] = []
    for scope in scopes:
        facts = [f"- [{topic_title(m.topic)}] {m.text}" for m in store.read_memories(scope)]
        if facts:
            parts.append(f"## {HEADINGS[scope.kind]} (most relevant first)\n\n" + "\n".join(facts))
    return "\n\n".join(parts)
