"""Which memory-store root a login uses, and the memory block a chat's system prompt gets while the Memory toggle is on."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from memory_store import store

if TYPE_CHECKING:
    from uuid import UUID

USER_MEMORY_STORE = "user"
# Read in full into every prompt; the rest of `user` and the project only as an index.
PINNED_USER_MEMORIES = ("profile", "preferences")
# Per-turn prompt bounds: each pinned memory's characters and each index's memories (newest first).
PINNED_MAX_CHARS = 4000
INDEX_MAX_MEMORIES = 60
CUT_NOTICE = "Some memory was cut here; use `list_memories` / `read_memories` for the rest."
REPO_ROOT = Path(__file__).resolve().parents[2]

MEMORY_RULES = (
    "Memory rules: save only durable facts worth knowing in a later chat (who the user is, preferences, decisions, "
    "project facts), one short self-contained fact per add. Change a fact with update (by its id) instead of adding a "
    "near-duplicate, retire facts that turned false. A new memory needs a short description. Never store secrets."
)


def memory_root(user_id: UUID | str, host_access: bool) -> Path:
    """The host login shares the machine's memory-store root with Pi; every other login gets its own folder in the checkout."""
    if host_access:
        return store.default_root()
    family_root = os.environ.get("GIGACHAD_MEMORY_DIR")
    base = Path(family_root).expanduser() if family_root else REPO_ROOT / ".memory_store"
    return base / str(user_id)


def unreserved_memory_store(name: str) -> str:
    """A project's memory_store name, `project-user` for a project that would take the login's `user`."""
    return f"project-{name}" if name == USER_MEMORY_STORE else name


def project_memory_store(project: dict | None, slug: str | None) -> str | None:
    """A project's memory_store: its linked folder's name when it has one, else its slug, never `user`."""
    if project is None:
        return None
    linked_root = project.get("linked_root")
    name = Path(linked_root).name if linked_root else slug
    return unreserved_memory_store(name) if name else name


def _pinned(root: Path, path: str) -> tuple[str, bool]:
    """The pinned memory *path* of `user` without ids, and whether it was cut to `PINNED_MAX_CHARS`."""
    text = store.read_memories(USER_MEMORY_STORE, [path], ids=False, root=root).rstrip()
    if len(text) <= PINNED_MAX_CHARS:
        return text, False
    return text[:PINNED_MAX_CHARS].rstrip() + " …", True


def _index(memory_store: str, listing: dict) -> tuple[str, bool]:
    """The index of *listing*, its newest `INDEX_MAX_MEMORIES` memories first, and whether some were left out."""
    memories = sorted(listing["memories"], key=lambda entry: entry.get("updated") or "", reverse=True)
    shown = {**listing, "memories": memories[:INDEX_MAX_MEMORIES]}
    return f"## Index of `{memory_store}`\n{store.render_list(shown).rstrip()}", len(memories) > INDEX_MAX_MEMORIES


def memory_prompt(root: Path, project_memory_store: str | None) -> str:
    """Create `user` and the project's memory_store when missing, then render the system-prompt memory block.

    A project name the store cannot address (e.g. a folder with spaces) leaves the project out.
    Pinned memories and indexes are bounded (`PINNED_MAX_CHARS`, `INDEX_MAX_MEMORIES`); a cut adds `CUT_NOTICE`.
    """
    store.create_memory_store(USER_MEMORY_STORE, root=root)
    if project_memory_store is not None:
        try:
            store.create_memory_store(project_memory_store, root=root)
        except store.MemoryStoreError:
            project_memory_store = None

    user_listing = store.list_memories(USER_MEMORY_STORE, root=root)
    present = {entry["path"] for entry in user_listing["memories"] if not entry.get("error")}
    parts = [_pinned(root, path) for path in PINNED_USER_MEMORIES if path in present]
    parts.append(_index(USER_MEMORY_STORE, user_listing))
    names = f"`{USER_MEMORY_STORE}` (about the user, across chats)"
    if project_memory_store is not None:
        parts.append(_index(project_memory_store, store.list_memories(project_memory_store, root=root)))
        names += f" and `{project_memory_store}` (this project)"
    sections = ["# Memory", *(text for text, _ in parts)]
    if any(cut for _, cut in parts):
        sections.append(CUT_NOTICE)
    sections.append(f"Use the memory_stores {names}. Read a memory before you answer from it.")
    sections.append(MEMORY_RULES)
    return "\n\n".join(sections)
