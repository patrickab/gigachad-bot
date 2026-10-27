"""Prompt store backed by the database-native ``prompt`` namespace.

Reads ``prompts/*.md`` files with YAML frontmatter and resolves ``{{block}}``
includes from ``prompts/_blocks/*.md``.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cached_property
import logging
from pathlib import Path
import re
import sys

import yaml

from lib.data_store import DataStore, overwrite_text, read_text, read_text_or
from lib.storage_namespace import PROMPT

log = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_INCLUDE_RE = re.compile(r"\{\{(\S+?)\}\}")
ORDER_KEY = f"{PROMPT}/_order"


def _parse_frontmatter(raw: str) -> tuple[dict[str, object], str]:
    """Return (metadata-dict, body) from a markdown-with-frontmatter string."""
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, raw[m.end():]


def seed_prompts(store: DataStore) -> None:
    """Initialize editable prompts once from the shipped defaults.

    An existing collection is authoritative, including deleted prompts. Defaults
    remain source assets; all subsequent editor writes go to persistent storage.
    """
    source = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2])) / "prompts"
    if not source.is_dir() or store.exists(PROMPT):
        return
    store.mkdir(PROMPT)
    for path in source.rglob("*"):
        relative = path.relative_to(source).as_posix()
        key = f"{PROMPT}/{relative}"
        if path.is_dir():
            store.mkdir(key)
        else:
            store.write_bytes(key, path.read_bytes())


class PromptStore:
    """Reads, resolves, and manages stored prompt definitions. The library is read on first use and after every write."""

    def __init__(self, data_store: DataStore) -> None:
        self._store = data_store

    @staticmethod
    def _key(slug: str) -> str:
        return f"{PROMPT}/{slug}.md"

    def reload(self) -> None:
        """Drop the cached library so the next resolved view reads it again."""
        for name in ("_order", "_block_cache", "_prompt_cache"):
            self.__dict__.pop(name, None)

    @cached_property
    def _order(self) -> list[str]:
        order = read_text_or(self._store, ORDER_KEY)
        return order.split() if order is not None else []

    @cached_property
    def _block_cache(self) -> dict[str, str]:
        return {stem: raw.strip() for stem, raw in self._md_entries(f"{PROMPT}/_blocks")}

    @cached_property
    def _prompt_cache(self) -> dict[str, str]:
        """Resolved text by prompt name."""
        return {str(meta.get("name", stem)): self._resolve(body).strip() for _, stem, meta, body in self._ranked_prompts()}

    def _rank(self, stem: str) -> tuple:
        """Sort key: position in the ``_order`` manifest; unlisted prompts fall to the end, alphabetical."""
        return (0, self._order.index(stem)) if stem in self._order else (1, stem.lower())

    def _md_entries(self, prefix: str) -> Iterator[tuple[str, str]]:
        """Yield `(stem, raw text)` of every `.md` file directly under *prefix*, ordered by key."""
        for entry in sorted(self._store.list(prefix), key=lambda item: item.key):
            if not entry.is_dir and entry.key.endswith(".md"):
                yield Path(entry.key).stem, read_text(self._store, entry.key)[0]

    def _ranked_prompts(self) -> list[tuple[tuple, str, dict[str, object], str]]:
        """Every prompt as `(rank, stem, frontmatter, body)` in display order."""
        parsed = []
        for stem, raw in self._md_entries(PROMPT):
            meta, body = _parse_frontmatter(raw)
            parsed.append((self._rank(stem), stem, meta, body))
        return sorted(parsed)

    def _resolve(self, body: str) -> str:
        def _repl(m: re.Match[str]) -> str:
            name = m.group(1)
            if name in self._block_cache:
                return self._block_cache[name]
            log.warning("prompt include {{%s}} not found in _blocks/", name)
            return m.group(0)
        return _INCLUDE_RE.sub(_repl, body)

    def prompt_map(self) -> dict[str, str]:
        return dict(self._prompt_cache)

    def blocks(self) -> dict[str, str]:
        return dict(self._block_cache)

    def get_raw(self, slug: str) -> str | None:
        """Return the raw (unresolved) file content for a prompt by filename slug."""
        return read_text_or(self._store, self._key(slug))

    def save(self, slug: str, content: str) -> str:
        """Write a prompt file. Returns the resolved name."""
        overwrite_text(self._store, self._key(slug), content)
        self.reload()
        meta, _ = _parse_frontmatter(content)
        return str(meta.get("name", slug))

    def delete(self, slug: str) -> bool:
        key = self._key(slug)
        if not self._store.exists(key):
            return False
        self._store.delete(key)
        if slug in self._order:
            self.set_order([s for s in self._order if s != slug])
        else:
            self.reload()
        return True

    def set_order(self, slugs: list[str]) -> None:
        """Persist the prompt display order. Slugs not listed fall to the end, alphabetical."""
        overwrite_text(self._store, ORDER_KEY, "\n".join(slugs) + "\n")
        self.reload()

    def list_prompts(self) -> list[dict[str, object]]:
        """Return metadata for all prompts (for the UI)."""
        return [
            {
                "slug": stem,
                "name": meta.get("name", stem),
                "includes": _INCLUDE_RE.findall(body),  # derived from body, not frontmatter
            }
            for _, stem, meta, body in self._ranked_prompts()
        ]
