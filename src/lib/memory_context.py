"""Inject stored memory into chat system prompts within a character budget."""

from __future__ import annotations

from dataclasses import dataclass

from lib.memory_store import MemoryScope, MemoryStore, ScopeKind, StoredMemory

# Facts beyond this many prompt characters are injected as topic summaries only.
PROMPT_CHAR_BUDGET = 16_000
# Topics injected in full regardless of the prompt budget, per scope.
PINNED_TOPICS: dict[ScopeKind, tuple[str, ...]] = {"global": ("profile", "preferences"), "project": ()}
HEADINGS: dict[ScopeKind, str] = {"global": "About the user", "project": "Project memory"}


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
    """Render memory per topic within `PROMPT_CHAR_BUDGET`.

    - Pinned global topics always render in full, outside the budget
    - The most recently updated other topics fill the budget
    - Topics past the budget appear as a summary index, so the model still knows they exist
    """
    scopes = [MemoryScope("global")]
    if project_slug:
        scopes.append(MemoryScope("project", project_slug))

    blocks: list[_TopicBlock] = []
    for scope in scopes:
        topics = store.get_topics(scope)
        order = {t["name"]: i for i, t in enumerate(topics)}
        summary_by_name = {t["name"]: t["summary"] for t in topics}
        by_topic: dict[str, list[StoredMemory]] = {}
        for memory in store.read_memories(scope):
            by_topic.setdefault(memory.topic, []).append(memory)
        for name, memories in by_topic.items():
            summary = summary_by_name.get(name, "")
            lines = [f"### {topic_title(name)}", *([summary] if summary else []), *(f"- {m.text}" for m in memories)]
            blocks.append(_TopicBlock(
                scope=scope.kind, name=name, summary=summary, text="\n".join(lines),
                order=order.get(name, len(order)),
                pinned=name in PINNED_TOPICS[scope.kind],
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
    for scope in scopes:
        scope_blocks = sorted((b for b in blocks if b.scope == scope.kind), key=lambda b: b.order)
        sections = [b.text for b in scope_blocks if b.included]
        index = [f"- {topic_title(b.name)}: {b.summary or 'no summary'}" for b in scope_blocks if not b.included]
        if index:
            sections.append("### Further topics (summaries only)\n" + "\n".join(index))
        if sections:
            parts.append(f"## {HEADINGS[scope.kind]}\n\n" + "\n\n".join(sections))
    return "\n\n".join(parts)
