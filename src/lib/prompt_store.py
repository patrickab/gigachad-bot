"""Filesystem-backed prompt store.

Reads ``prompts/*.md`` files with YAML frontmatter, resolves ``{{block}}``
includes from ``prompts/_blocks/*.md``, and exposes a dict-like interface
for the rest of the app.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re

import yaml

from lib.data_store import DataStore, LocalDataStore, StorageNotFoundError, read_text, write_text

log = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_INCLUDE_RE = re.compile(r"\{\{(\S+?)\}\}")


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


class PromptStore:
    """Reads, resolves, and manages markdown prompt files on disk."""

    def __init__(self, base_dir: Path, *, data_store: DataStore | None = None) -> None:
        base_dir = base_dir.expanduser().resolve()
        self._store = data_store or LocalDataStore(base_dir.parent)
        self._prefix = base_dir.name
        self._block_cache: dict[str, str] = {}
        self._prompt_cache: dict[str, tuple[str, str]] = {}  # name -> (resolved_text, filename)
        self._order: list[str] = []
        self.reload()

    def reload(self) -> None:
        self._block_cache.clear()
        self._prompt_cache.clear()
        try:
            order, _ = read_text(self._store, f"{self._prefix}/_order")
            self._order = order.split()
        except StorageNotFoundError:
            self._order = []
        self._load_blocks()
        self._load_prompts()

    def _rank(self, stem: str) -> tuple:
        """Sort key: position in the ``_order`` manifest; unlisted prompts fall to the end, alphabetical."""
        return (0, self._order.index(stem)) if stem in self._order else (1, stem.lower())

    def _load_blocks(self) -> None:
        prefix = f"{self._prefix}/_blocks"
        for entry in sorted(self._store.list(prefix), key=lambda item: item.key):
            if entry.is_dir or not entry.key.endswith(".md"):
                continue
            raw, _ = read_text(self._store, entry.key)
            self._block_cache[Path(entry.key).stem] = raw.strip()

    def _resolve(self, body: str) -> str:
        def _repl(m: re.Match[str]) -> str:
            name = m.group(1)
            if name in self._block_cache:
                return self._block_cache[name]
            log.warning("prompt include {{%s}} not found in _blocks/", name)
            return m.group(0)
        return _INCLUDE_RE.sub(_repl, body)

    def _load_prompts(self) -> None:
        parsed = []
        for entry in self._store.list(self._prefix):
            if entry.is_dir or not entry.key.endswith(".md"):
                continue
            raw, _ = read_text(self._store, entry.key)
            stem = Path(entry.key).stem
            meta, body = _parse_frontmatter(raw)
            parsed.append((self._rank(stem), meta, body, stem))
        for _, meta, body, stem in sorted(parsed):
            name = str(meta.get("name", stem))
            self._prompt_cache[name] = (self._resolve(body).strip(), stem)

    def prompt_map(self) -> dict[str, str]:
        return {name: text for name, (text, _) in self._prompt_cache.items()}

    def blocks(self) -> dict[str, str]:
        return dict(self._block_cache)

    def get_raw(self, slug: str) -> str | None:
        """Return the raw (unresolved) file content for a prompt by filename slug."""
        key = f"{self._prefix}/{slug}.md"
        try:
            content, _ = read_text(self._store, key)
        except StorageNotFoundError:
            return None
        return content

    def save(self, slug: str, content: str) -> str:
        """Write a prompt file and reload. Returns the resolved name."""
        key = f"{self._prefix}/{slug}.md"
        try:
            _, revision = read_text(self._store, key)
        except StorageNotFoundError:
            revision = None
        write_text(self._store, key, content, expected=revision)
        self.reload()
        meta, _ = _parse_frontmatter(content)
        return str(meta.get("name", slug))

    def delete(self, slug: str) -> bool:
        key = f"{self._prefix}/{slug}.md"
        if not self._store.exists(key):
            return False
        self._store.delete(key)
        if slug in self._order:
            self.set_order([s for s in self._order if s != slug])  # reloads
        else:
            self.reload()
        return True

    def set_order(self, slugs: list[str]) -> None:
        """Persist the prompt display order. Slugs not listed fall to the end, alphabetical."""
        key = f"{self._prefix}/_order"
        try:
            _, revision = read_text(self._store, key)
        except StorageNotFoundError:
            revision = None
        write_text(self._store, key, "\n".join(slugs) + "\n", expected=revision)
        self.reload()

    def list_prompts(self) -> list[dict[str, object]]:
        """Return metadata for all prompts (for the UI)."""
        parsed = []
        for entry in self._store.list(self._prefix):
            if entry.is_dir or not entry.key.endswith(".md"):
                continue
            raw, _ = read_text(self._store, entry.key)
            stem = Path(entry.key).stem
            meta, body = _parse_frontmatter(raw)
            parsed.append((self._rank(stem), stem, meta, body))
        return [
            {
                "slug": stem,
                "name": meta.get("name", stem),
                "includes": _INCLUDE_RE.findall(body),  # derived from body, not frontmatter
            }
            for _, stem, meta, body in sorted(parsed)
        ]
