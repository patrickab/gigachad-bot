"""Read-only access seam for Obsidian vaults.

A user can register several vault roots (e.g. one per Obsidian vault). The list
of roots is maintained in ``chat_histories/obsidian-roots.json`` — the single
source of truth for vault locations (nothing is configured in ``config.py``).
When that file is absent the vault is simply empty until the user adds a root.
All paths handed to the UI are absolute, and every read is validated to live
under one of the roots (mirroring the path-traversal guard used by ``ChatStore``).

Named ``ObsidianVault`` (not ``VaultStore``) because "vault" already refers to
the sidebar projects/histories tree (``VaultTree``) elsewhere in the codebase.
"""

import re
from pathlib import Path
from urllib.parse import quote as _url_quote

from config import DIRECTORY_CHAT_HISTORIES
from lib.json_io import safe_read_json, safe_write_json

# Directories that hold Obsidian internals or noise rather than notes.
_SKIP_DIRS = {".obsidian", ".trash", ".git", "node_modules"}
_IMG_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"})

ROOTS_FILE = DIRECTORY_CHAT_HISTORIES / "obsidian-roots.json"


def _is_untitled(name: str) -> bool:
    """True for Obsidian's default ``Untitled*.md`` scratch notes."""
    return Path(name).stem.lower().startswith("untitled")


class ObsidianVault:
    """All read access to the user's Obsidian vault roots goes through this class."""

    def __init__(self, roots_file: Path | None = None) -> None:
        self._roots_file = roots_file or ROOTS_FILE
        self._roots = self._load_roots()

    def _load_roots(self) -> list[Path]:
        raw = safe_read_json(self._roots_file, {"roots": []}).get("roots") or []
        roots: list[Path] = []
        for p in raw:
            resolved = Path(p).expanduser().resolve()
            if resolved not in roots:
                roots.append(resolved)
        return roots

    def _save_roots(self) -> None:
        safe_write_json(self._roots_file, {"roots": [str(r) for r in self._roots]})

    # ─── roots management ───

    @property
    def enabled(self) -> bool:
        return any(r.is_dir() for r in self._roots)

    def roots(self) -> list[str]:
        return [str(r) for r in self._roots]

    def add_root(self, path: str) -> None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"Not a directory: {path}")
        if resolved not in self._roots:
            self._roots.append(resolved)
            self._save_roots()

    def remove_root(self, path: str) -> None:
        resolved = Path(path).expanduser().resolve()
        self._roots = [r for r in self._roots if r != resolved]
        self._save_roots()

    # ─── path safety ───

    def _root_for(self, path: str) -> Path:
        """Resolve *path*, rejecting anything that lives outside all roots."""
        target = Path(path).expanduser().resolve()
        if any(target.is_relative_to(root) for root in self._roots):
            return target
        raise ValueError(f"Path outside vault roots: {path}")

    def contains(self, abs_path: Path) -> bool:
        try:
            self._root_for(str(abs_path))
            return True
        except ValueError:
            return False

    # ─── listing ───

    def list_markdown(self) -> list[dict[str, str]]:
        """Flat list of every note as ``{path, name}`` (absolute paths), Untitled filtered.

        Kept for the existing Obsidian picker/attach flow, which wants a flat,
        searchable list rather than the nested tree.
        """
        files: list[dict[str, str]] = []
        for root in self._roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*.md"):
                if not path.is_file():
                    continue
                rel = path.relative_to(root)
                if any(part in _SKIP_DIRS for part in rel.parts):
                    continue
                if _is_untitled(path.name):
                    continue
                files.append({"path": str(path), "name": path.name})
        files.sort(key=lambda f: f["path"].lower())
        return files

    def tree(self) -> list[dict]:
        """Nested folder/file tree per root for the sidebar VaultTree.

        Each root is a ``vault`` node; directories are ``folder`` nodes, ``.md``
        files are ``file`` leaves. Empty folders (no notes underneath) are pruned.
        """
        out: list[dict] = []
        for root in self._roots:
            if not root.is_dir():
                continue
            out.append({
                "name": root.name,
                "path": str(root),
                "type": "vault",
                "children": self._build_children(root),
            })
        return out

    def _build_children(self, directory: Path) -> list[dict]:
        # Folders first, then files, each alphabetical — matches a file browser.
        children: list[dict] = []
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return children
        for entry in entries:
            if entry.name in _SKIP_DIRS:
                continue
            if entry.is_dir():
                sub = self._build_children(entry)
                if sub:  # prune folders with no notes
                    children.append({"name": entry.name, "path": str(entry), "type": "folder", "children": sub})
            elif entry.is_file() and entry.suffix == ".md" and not _is_untitled(entry.name):
                children.append({"name": entry.stem, "path": str(entry), "type": "file"})
        return children

    def read(self, path: str) -> str:
        """Return the text content of a markdown note."""
        resolved = self._root_for(path)
        if not resolved.is_file():
            raise FileNotFoundError(path)
        return resolved.read_text(encoding="utf-8")

    def write(self, path: str, content: str) -> None:
        """Overwrite a markdown note in place (validated under a vault root)."""
        resolved = self._root_for(path)
        resolved.write_text(content, encoding="utf-8")

    # ─── wiki-link resolution ───

    def find_in_vault(self, name: str, source_dir: Path | None = None) -> Path | None:
        """Find a file by name. Checks source directory first, then all roots."""
        # ponytail: rglob(name) — breaks if name has glob metacharacters
        name = name.split("#")[0].strip()
        if not name:
            return None
        candidates = [name] if Path(name).suffix else [name + ".md", name]
        if source_dir and source_dir.is_dir():
            for c in candidates:
                p = (source_dir / c).resolve()
                if p.is_file() and self.contains(p):
                    return p
        for root in self._roots:
            if not root.is_dir():
                continue
            for c in candidates:
                for hit in root.rglob(c):
                    if hit.is_file():
                        return hit
        return None

    def resolve_wiki_content(self, content: str, source_path: str) -> str:
        """Rewrite ``![[embed]]``, ``[[link]]``, and relative images to standard markdown."""
        source_dir = Path(source_path).parent

        def _resolve(name: str) -> Path | None:
            return self.find_in_vault(name, source_dir)

        def _repl_embed(m: re.Match[str]) -> str:
            name = m.group(1).split("|")[0].strip()
            found = _resolve(name)
            if not found:
                return m.group(0)
            if found.suffix.lower() in _IMG_SUFFIXES:
                return f"![{name}](/api/fileviewer/raw?path={_url_quote(str(found))})"
            return f"[{name}](#obsidian:{_url_quote(str(found))})"

        content = re.sub(r"!\[\[([^\]]+)\]\]", _repl_embed, content)

        def _repl_link(m: re.Match[str]) -> str:
            parts = m.group(1).split("|", 1)
            name = parts[0].strip()
            display = parts[1].strip() if len(parts) > 1 else name
            found = _resolve(name)
            if found:
                return f"[{display}](#obsidian:{_url_quote(str(found))})"
            return f"[{display}](#)"

        content = re.sub(r"\[\[([^\]]+)\]\]", _repl_link, content)

        def _repl_rel_img(m: re.Match[str]) -> str:
            alt, src = m.group(1), m.group(2)
            if src.startswith(("/", "http://", "https://", "data:")):
                return m.group(0)
            resolved = (source_dir / src).resolve()
            if resolved.is_file() and self.contains(resolved):
                return f"![{alt}](/api/fileviewer/raw?path={_url_quote(str(resolved))})"
            return m.group(0)

        return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _repl_rel_img, content)
