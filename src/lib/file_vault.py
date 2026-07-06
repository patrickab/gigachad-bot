"""Access seam for file vaults — directories of Attachment-compatible files.

A user can register several vault roots (an Obsidian vault is just one flavor
of file vault). The list of roots is maintained in
``chat_histories/file-vault-roots.json`` — the single source of truth for vault
locations (nothing is configured in ``config.py``). When that file is absent
the vault is simply empty until the user adds a root.

Each root may carry additional **mountpoints** — external directories (e.g. a
``/mnt`` drive) attached to that vault. Mountpoint files appear as folder nodes
inside the parent vault's tree rather than as separate top-level vaults, so a
mounted drive is browsed as part of the vault it was attached to.

Vault files are used as **live references**: attaching one to a chat stores its
path, never a copy — edits through the app write back to the actual file.

All paths handed to the UI are absolute, and every read/write is validated to
live under one of the roots or their mountpoints (mirroring the path-traversal
guard used by ``ChatStore``).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from urllib.parse import quote as _url_quote

from config import DIRECTORY_CHAT_HISTORIES
from lib.json_io import safe_read_json, safe_write_json

# Directories that hold app internals (e.g. Obsidian's) or noise rather than files.
_SKIP_DIRS = {".obsidian", ".trash", ".git", "node_modules"}
_IMG_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"})
# Attachment-compatible file types surfaced by listings.
VAULT_SUFFIXES = frozenset({".md", ".txt", ".tex", ".pdf"})

ROOTS_FILE = DIRECTORY_CHAT_HISTORIES / "file-vault-roots.json"
_LEGACY_ROOTS_FILE = DIRECTORY_CHAT_HISTORIES / "obsidian-roots.json"


def _is_untitled(name: str) -> bool:
    """True for Obsidian's default ``Untitled*.md`` scratch notes."""
    return Path(name).stem.lower().startswith("untitled")


class FileVault:
    """All access to the user's file-vault roots goes through this class."""

    def __init__(self, roots_file: Path | None = None) -> None:
        self._roots_file = roots_file or ROOTS_FILE
        self._roots: list[Path] = []
        self._mountpoints: dict[Path, list[Path]] = defaultdict(list)
        # A root may be mounted to a project (slug) instead of the global Vaults section.
        self._project_tags: dict[Path, str] = {}
        self._reload()

    def _reload(self) -> None:
        source = self._roots_file
        # One-time migration: fall back to the pre-rename obsidian-roots.json.
        if not source.is_file() and self._roots_file == ROOTS_FILE and _LEGACY_ROOTS_FILE.is_file():
            source = _LEGACY_ROOTS_FILE
        raw = safe_read_json(source, {"roots": []}).get("roots") or []
        self._roots = []
        self._mountpoints = defaultdict(list)
        self._project_tags = {}
        for entry in raw:
            # Backward compat: a bare string entry is a root with no mountpoints.
            if isinstance(entry, str):
                path = Path(entry).expanduser().resolve()
                if path not in self._roots:
                    self._roots.append(path)
                continue
            if not isinstance(entry, dict):
                continue
            path = Path(str(entry.get("path", ""))).expanduser().resolve()
            if not path:
                continue
            if path not in self._roots:
                self._roots.append(path)
            if entry.get("project"):
                self._project_tags[path] = str(entry["project"])
            for mp in entry.get("mountpoints") or []:
                resolved_mp = Path(str(mp)).expanduser().resolve()
                if resolved_mp not in self._mountpoints[path]:
                    self._mountpoints[path].append(resolved_mp)

    def _save_roots(self) -> None:
        payload = {
            "roots": [
                {
                    "path": str(r),
                    "mountpoints": [str(m) for m in self._mountpoints.get(r, [])],
                    **({"project": self._project_tags[r]} if r in self._project_tags else {}),
                }
                for r in self._roots
            ]
        }
        safe_write_json(self._roots_file, payload)

    def _all_mountpoints(self) -> list[Path]:
        out: list[Path] = []
        for mps in self._mountpoints.values():
            for m in mps:
                if m not in out:
                    out.append(m)
        return out

    # ─── roots management ───

    @property
    def enabled(self) -> bool:
        return any(r.is_dir() for r in self._roots)

    def roots(self) -> list[str]:
        return [str(r) for r in self._roots]

    def add_root(self, path: str, project: str | None = None) -> None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"Not a directory: {path}")
        if resolved not in self._roots:
            self._roots.append(resolved)
        if project:
            self._project_tags[resolved] = project
        self._save_roots()

    def remove_root(self, path: str) -> None:
        resolved = Path(path).expanduser().resolve()
        self._roots = [r for r in self._roots if r != resolved]
        self._mountpoints.pop(resolved, None)
        self._project_tags.pop(resolved, None)
        self._save_roots()

    # ─── mountpoint management ───

    def mountpoints(self, vault_path: str) -> list[str]:
        resolved = Path(vault_path).expanduser().resolve()
        return [str(m) for m in self._mountpoints.get(resolved, [])]

    def add_mountpoint(self, vault_path: str, mountpoint_path: str) -> None:
        vault = Path(vault_path).expanduser().resolve()
        if vault not in self._roots:
            raise ValueError(f"Unknown vault: {vault_path}")
        mountpoint = Path(mountpoint_path).expanduser().resolve()
        if not mountpoint.is_dir():
            raise ValueError(f"Not a directory: {mountpoint_path}")
        if mountpoint == vault:
            raise ValueError("Mountpoint cannot be the vault itself")
        # Reject if this path is already a root or a mountpoint of another vault.
        if mountpoint in self._roots:
            raise ValueError(f"Already a vault root: {mountpoint_path}")
        for v, mps in self._mountpoints.items():
            if v != vault and mountpoint in mps:
                raise ValueError(f"Already mounted under another vault: {mountpoint_path}")
        if mountpoint not in self._mountpoints[vault]:
            self._mountpoints[vault].append(mountpoint)
            self._save_roots()

    def remove_mountpoint(self, vault_path: str, mountpoint_path: str) -> None:
        vault = Path(vault_path).expanduser().resolve()
        mountpoint = Path(mountpoint_path).expanduser().resolve()
        self._mountpoints[vault] = [m for m in self._mountpoints.get(vault, []) if m != mountpoint]
        if not self._mountpoints[vault]:
            self._mountpoints.pop(vault, None)
        self._save_roots()

    # ─── path safety ───

    def _root_for(self, path: str) -> Path:
        """Resolve *path*, rejecting anything that lives outside all roots and mountpoints."""
        target = Path(path).expanduser().resolve()
        for root in self._roots:
            if target.is_relative_to(root):
                return target
        for mountpoint in self._all_mountpoints():
            if target.is_relative_to(mountpoint):
                return target
        raise ValueError(f"Path outside vault roots: {path}")

    def contains(self, abs_path: Path) -> bool:
        try:
            self._root_for(str(abs_path))
            return True
        except ValueError:
            return False

    # ─── listing ───

    def list_files(self) -> list[dict[str, str]]:
        """Flat list of every vault file as ``{path, name}`` (absolute paths), Untitled filtered.

        Kept for the picker/attach flow, which wants a flat, searchable list
        rather than the nested tree.
        """
        files: list[dict[str, str]] = []
        for root in self._roots:
            if root.is_dir():
                self._collect_files(root, files)
        files.sort(key=lambda f: f["path"].lower())
        return files

    def list_files_for_project(self, slug: str) -> list[dict[str, str]]:
        """Flat ``{path, name}`` list of files under vaults mounted to *slug*.

        Includes files from each such vault's mountpoints so a mounted drive's
        contents surface alongside the vault's own files.
        """
        files: list[dict[str, str]] = []
        for root in self._roots:
            if not root.is_dir():
                continue
            if self._project_tags.get(root) != slug:
                continue
            self._collect_files(root, files)
            for mp in self._mountpoints.get(root, []):
                if mp.is_dir():
                    self._collect_files(mp, files)
        files.sort(key=lambda f: f["path"].lower())
        return files

    @staticmethod
    def _collect_files(base: Path, out: list[dict[str, str]]) -> None:
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in VAULT_SUFFIXES:
                continue
            rel = path.relative_to(base)
            if any(part in _SKIP_DIRS for part in rel.parts):
                continue
            if _is_untitled(path.name):
                continue
            out.append({"path": str(path), "name": path.name})

    def tree(self) -> list[dict]:
        """Nested folder/file tree per root for the sidebar VaultTree.

        Each root is a ``vault`` node; directories are ``folder`` nodes,
        Attachment-compatible files are ``file`` leaves. Empty folders (no vault
        files underneath) are pruned. Mountpoints attached to a vault appear as
        folder nodes inside that vault.
        """
        out: list[dict] = []
        for root in self._roots:
            if not root.is_dir():
                continue
            children = self._build_children(root)
            for mp in self._mountpoints.get(root, []):
                if not mp.is_dir():
                    continue
                sub = self._build_children(mp)
                if not sub:
                    # Surface empty mountpoints too so the user can see them.
                    children.append({"name": mp.name, "path": str(mp), "type": "folder", "children": []})
                else:
                    children.append({"name": mp.name, "path": str(mp), "type": "folder", "children": sub})
            out.append({
                "name": root.name,
                "path": str(root),
                "type": "vault",
                "project": self._project_tags.get(root),
                "children": children,
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
                if sub:  # prune folders with no vault files
                    children.append({"name": entry.name, "path": str(entry), "type": "folder", "children": sub})
            elif entry.is_file() and entry.suffix.lower() in VAULT_SUFFIXES and not _is_untitled(entry.name):
                children.append({"name": entry.stem, "path": str(entry), "type": "file"})
        return children

    def resolve(self, path: str) -> Path:
        """Validate *path* against the roots and require it to be an existing file."""
        resolved = self._root_for(path)
        if not resolved.is_file():
            raise FileNotFoundError(path)
        return resolved

    def read(self, path: str) -> str:
        """Return the text content of a vault file."""
        return self.resolve(path).read_text(encoding="utf-8")

    def write(self, path: str, content: str) -> None:
        """Overwrite a vault file in place (validated under a vault root)."""
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
        for mountpoint in self._all_mountpoints():
            if not mountpoint.is_dir():
                continue
            for c in candidates:
                for hit in mountpoint.rglob(c):
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
            return f"[{name}](#vault:{_url_quote(str(found))})"

        content = re.sub(r"!\[\[([^\]]+)\]\]", _repl_embed, content)

        def _repl_link(m: re.Match[str]) -> str:
            parts = m.group(1).split("|", 1)
            name = parts[0].strip()
            display = parts[1].strip() if len(parts) > 1 else name
            found = _resolve(name)
            if found:
                return f"[{display}](#vault:{_url_quote(str(found))})"
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
