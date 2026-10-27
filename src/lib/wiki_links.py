"""Obsidian-style markdown to standard markdown: `![[embed]]`, `[[link]]` and relative images point at mounted files."""

from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING
from urllib.parse import quote

from lib.mime import IMAGE_SUFFIXES

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def _find(name: str, source_dir: Path, roots: Iterable[Path], contains: Callable[[Path], bool]) -> Path | None:
    """A file by name: the source directory first, then every root."""
    # ponytail: rglob(name) breaks if name has glob metacharacters
    name = name.split("#")[0].strip()
    if not name:
        return None
    candidates = [name] if Path(name).suffix else [name + ".md", name]
    if source_dir.is_dir():
        for candidate in candidates:
            path = (source_dir / candidate).resolve()
            if path.is_file() and contains(path):
                return path
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in candidates:
            for hit in root.rglob(candidate):
                if hit.is_file():
                    return hit
    return None


def resolve_wiki_content(content: str, source_path: str, roots: Iterable[Path], contains: Callable[[Path], bool]) -> str:
    """Rewrite the wiki links of *content*, read at *source_path*, against *roots*; *contains* admits a resolved path."""
    source_dir = Path(source_path).parent

    def _repl_embed(m: re.Match[str]) -> str:
        name = m.group(1).split("|")[0].strip()
        found = _find(name, source_dir, roots, contains)
        if not found:
            return m.group(0)
        if found.suffix.lower() in IMAGE_SUFFIXES:
            return f"![{name}](/api/fileviewer/raw?path={quote(str(found))})"
        return f"[{name}](#file:{quote(str(found))})"

    # Embeds run first, since `![[x]]` would otherwise match the plain `[[x]]` link pass.
    content = re.sub(r"!\[\[([^\]]+)\]\]", _repl_embed, content)

    def _repl_link(m: re.Match[str]) -> str:
        parts = m.group(1).split("|", 1)
        name = parts[0].strip()
        display = parts[1].strip() if len(parts) > 1 else name
        found = _find(name, source_dir, roots, contains)
        if found:
            return f"[{display}](#file:{quote(str(found))})"
        return f"[{display}](#)"

    content = re.sub(r"\[\[([^\]]+)\]\]", _repl_link, content)

    def _repl_rel_img(m: re.Match[str]) -> str:
        alt, src = m.group(1), m.group(2)
        if src.startswith(("/", "http://", "https://", "data:")):
            return m.group(0)
        resolved = (source_dir / src).resolve()
        if resolved.is_file() and contains(resolved):
            return f"![{alt}](/api/fileviewer/raw?path={quote(str(resolved))})"
        return m.group(0)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _repl_rel_img, content)
