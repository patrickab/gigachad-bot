"""Access seam for file roots: directories of Attachment-compatible files.

A user can register several file roots (an Obsidian vault is just one flavor
of file root). The list of roots is the single source of truth for file
locations (nothing is configured in ``config.py``): it lives in ``vault_roots``
rows for the authenticated user. With no registry the roots are simply empty
until the user adds one.

Listings (flat list, tree, folder expansion) honor ``.gitignore`` files found
from each root downward, with git's precedence: the last matching pattern wins
and deeper files override shallower ones. Ignored folders are never entered.
Explicit paths (attach, read, write) are not filtered, only validated against
the roots. A root that is a git clone also admits its worktrees (``git_worktree``).

Root files are used as **live references**: attaching one to a chat stores its
path, never a copy. Edits through the app write back to the actual file.

All paths handed to the UI are absolute, and every read/write is validated to
live under one of the roots (mirroring the path-traversal guard used by
``ChatStore``).
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Iterable, Protocol
from urllib.parse import quote as _url_quote
from uuid import UUID

from pathspec import GitIgnoreSpec
from psycopg_pool import ConnectionPool

from lib import git_worktree

# Directories that hold app internals (e.g. Obsidian's) or noise rather than files, skipped even without a .gitignore.
_SKIP_DIRS = frozenset({".obsidian", ".trash", ".git", "node_modules"})
# Each layer is one directory's `.gitignore`: (directory path, compiled patterns), outermost first.
_IgnoreLayers = tuple[tuple[str, GitIgnoreSpec], ...]
_IMG_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"})


def _with_gitignore(directory: Path | str, inherited: _IgnoreLayers) -> _IgnoreLayers:
    """*inherited* plus *directory*'s own `.gitignore`, if it has a readable one."""
    try:
        lines = (Path(directory) / ".gitignore").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return inherited
    return (*inherited, (str(directory), GitIgnoreSpec.from_lines(lines)))


def _is_ignored(path: str, is_dir: bool, layers: _IgnoreLayers) -> bool:
    """Git precedence: the last matching pattern wins, and deeper `.gitignore` files override shallower ones."""
    ignored = False
    for base, spec in layers:
        relative = Path(path).relative_to(base).as_posix()
        verdict = spec.check_file(f"{relative}/" if is_dir else relative).include
        if verdict is not None:
            ignored = verdict
    return ignored


def _visible_entries(directory: Path | str, inherited: _IgnoreLayers) -> tuple[list[tuple[os.DirEntry, bool]], _IgnoreLayers]:
    """Entries of *directory* as ``(entry, is_dir)`` minus skipped and git-ignored ones, plus the layers for its children.

    An ignored folder is dropped whole, as in git, so nothing inside it can be re-included.
    """
    layers = _with_gitignore(directory, inherited)
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return [], layers
    visible = []
    for entry in entries:
        is_dir = entry.is_dir(follow_symlinks=False)
        # A linked worktree's `.git` is a file pointing at the clone, skipped like the folder.
        if (is_dir and entry.name in _SKIP_DIRS) or entry.name == ".git" or _is_ignored(entry.path, is_dir, layers):
            continue
        visible.append((entry, is_dir))
    return visible, layers


class FileRootRepository(Protocol):
    """Where the *root registry* lives. File root file contents never pass through here."""

    def load(self) -> list[str]:
        """Registered root paths in registration order."""

    def save(self, roots: list[str]) -> None:
        """Replace the registry with *roots*."""


class PostgresFileRootRepository:
    """The registry is ``vault_roots`` rows for one authenticated user.

    Stored paths are whatever spelling the caller registered; ``FileRoots``
    re-resolves them on load, so a row migrated from another host is resolved
    against *this* host's filesystem.
    """

    def __init__(self, pool: ConnectionPool, user_id: UUID, *, device_id: UUID | None = None) -> None:
        self._pool = pool
        self._user_id = user_id
        self._device_id = device_id

    def load(self) -> list[str]:
        with self._pool.connection() as connection:
            rows = connection.execute(
                "SELECT path FROM vault_roots WHERE user_id = %s ORDER BY created_at, path",
                (self._user_id,),
            ).fetchall()
        return [path for (path,) in rows]

    def _change(self, connection, path: str, operation: str) -> None:
        connection.execute(
            """
            INSERT INTO changes (user_id, resource_kind, resource_key, operation, device_id)
            VALUES (%s, 'vault_root', %s, %s, %s)
            """,
            (self._user_id, path, operation, self._device_id),
        )

    def save(self, roots: list[str]) -> None:
        # ponytail: whole-registry replace, mirroring the JSON file it replaces,
        # one statement per root. A user keeps a handful of roots; if that ever
        # grows, give FileRoots per-root add/remove hooks instead of diffing here.
        with self._pool.connection() as connection, connection.transaction():
            for path in roots:
                inserted = connection.execute(
                    "INSERT INTO vault_roots (user_id, path) VALUES (%s, %s) ON CONFLICT (user_id, path) DO NOTHING",
                    (self._user_id, path),
                ).rowcount
                if inserted:
                    self._change(connection, path, "write")
            removed = connection.execute(
                "DELETE FROM vault_roots WHERE user_id = %s AND NOT (path = ANY(%s::text[])) RETURNING path",
                (self._user_id, roots),
            ).fetchall()
            for (path,) in removed:
                self._change(connection, path, "delete")


def list_directories(path: str) -> tuple[Path, list[str]]:
    """Resolve *path* (`~` expanded) and list its subdirectory names, hidden ones included.

    - Used to pick a new root, so it browses the whole host, not only registered roots
    - Raises NotADirectoryError for a missing or non-directory path, PermissionError when unreadable
    """
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(path)
    with os.scandir(directory) as entries:
        names = [entry.name for entry in entries if entry.is_dir()]
    return directory, sorted(names, key=str.casefold)


class FileRoots:
    """All access to the user's file-root roots goes through this class."""

    def __init__(self, *, repository: FileRootRepository) -> None:
        self._repository = repository
        self._roots: list[Path] = []
        self._worktrees: dict[Path, list[Path]] | None = None
        self._reload()

    def _reload(self) -> None:
        self._roots = []
        for entry in self._repository.load():
            path = Path(entry).expanduser().resolve()
            if path not in self._roots:
                self._roots.append(path)

    def _save_roots(self) -> None:
        self._repository.save([str(root) for root in self._roots])

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

    def worktrees_of(self, root: Path) -> list[Path]:
        """Linked worktrees of a registered clone root, empty for other roots. Listed once per instance."""
        if self._worktrees is None:
            self._worktrees = {}
            for candidate in self._roots:
                if not git_worktree.is_clone(candidate):
                    continue
                try:
                    listed = git_worktree.list_worktrees(candidate)
                except git_worktree.WorktreeError:
                    continue
                self._worktrees[candidate] = [Path(worktree.path) for worktree in listed if not worktree.is_main]
        return self._worktrees.get(root, [])

    def _accessible_roots(self, target: Path) -> list[Path]:
        """Registered roots holding *target*, else the clone worktrees holding it, so worktrees cost git calls only when used."""
        holding = [root for root in self._roots if target.is_relative_to(root)]
        return holding or [worktree for root in self._roots for worktree in self.worktrees_of(root) if target.is_relative_to(worktree)]

    def _root_for(self, path: str) -> Path:
        """Resolve *path*, rejecting anything that lives outside all roots and their worktrees."""
        target = Path(path).expanduser().resolve()
        if self._accessible_roots(target):
            return target
        raise ValueError(f"Path outside file roots: {path}")

    def contains(self, abs_path: Path) -> bool:
        try:
            self._root_for(str(abs_path))
            return True
        except ValueError:
            return False

    def root_of(self, path: str) -> Path:
        """Innermost registered root or clone worktree holding *path*, ValueError outside all of them."""
        target = Path(path).expanduser().resolve()
        holding = self._accessible_roots(target)
        if not holding:
            raise ValueError(f"Path outside file roots: {path}")
        return max(holding, key=lambda root: len(root.parts))

    # ─── listing ───

    def list_files(self) -> list[dict[str, str]]:
        """Flat list of every file-root file as ``{path, name}`` (absolute paths).

        Kept for the picker/attach flow, which wants a flat, searchable list
        rather than the nested tree.
        """
        files: list[dict[str, str]] = []
        for root in self._roots:
            if root.is_dir():
                self._collect_files(root, files)
        files.sort(key=lambda f: f["path"].lower())
        return files

    @staticmethod
    def _collect_files(base: Path | str, out: list[dict[str, str]], inherited: _IgnoreLayers = ()) -> None:
        """Append listed files below *base*, skipping ignored entries and not following directory symlinks."""
        entries, layers = _visible_entries(base, inherited)
        for entry, is_dir in entries:
            if is_dir:
                FileRoots._collect_files(entry.path, out, layers)
            elif entry.is_file():
                out.append({"path": entry.path, "name": entry.name})

    def files_under(self, paths: list[str]) -> list[Path]:
        """Files among *paths*, expanding folders recursively. Deduplicated, order kept."""
        found: dict[Path, None] = {}
        for path in paths:
            target = self._root_for(path)
            if target.is_dir():
                entries: list[dict[str, str]] = []
                self._collect_files(target, entries)
                found.update(dict.fromkeys(Path(entry["path"]) for entry in entries))
            elif target.is_file():
                found[target] = None
        return list(found)

    def tree(self, active_worktrees: Iterable[str] = ()) -> list[dict]:
        """Nested folder/file tree per root for the sidebar FileTree. Empty folders are pruned.

        - A root listed with one of its worktrees in *active_worktrees* shows that worktree's files and path
        - `mount` always holds the registered root path
        """
        active = {Path(path).expanduser().resolve() for path in active_worktrees}
        nodes = []
        for root in self._roots:
            if not root.is_dir():
                continue
            shown = next((worktree for worktree in self.worktrees_of(root) if worktree in active), root) if active else root
            nodes.append({"name": root.name, "path": str(shown), "mount": str(root), "type": "root", "children": self._build_children(shown)})
        return nodes

    @staticmethod
    def _build_children(directory: Path | str, inherited: _IgnoreLayers = ()) -> list[dict]:
        # Folders first, then files, each alphabetical, matches a file browser.
        entries, layers = _visible_entries(directory, inherited)
        children: list[dict] = []
        for entry, is_dir in sorted(entries, key=lambda item: (not item[1], item[0].name.lower())):
            if is_dir:
                sub = FileRoots._build_children(entry.path, layers)
                if sub:  # prune folders with no file-root files
                    children.append({"name": entry.name, "path": entry.path, "type": "folder", "children": sub})
            elif entry.is_file():
                children.append({"name": entry.name, "path": entry.path, "type": "file"})
        return children

    def resolve(self, path: str) -> Path:
        """Validate *path* against the roots and require it to be an existing file."""
        resolved = self._root_for(path)
        if not resolved.is_file():
            raise FileNotFoundError(path)
        return resolved

    def read(self, path: str) -> str:
        """Return the text content of a file-root file."""
        return self.resolve(path).read_text(encoding="utf-8")

    def write(self, path: str, content: str) -> None:
        """Overwrite a file-root file in place (validated under a file root)."""
        resolved = self._root_for(path)
        resolved.write_text(content, encoding="utf-8")

    # ─── wiki-link resolution ───

    def find_in_roots(self, name: str, source_dir: Path | None = None) -> Path | None:
        """Find a file by name. Checks source directory first, then all roots."""
        # ponytail: rglob(name) breaks if name has glob metacharacters
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
            return self.find_in_roots(name, source_dir)

        def _repl_embed(m: re.Match[str]) -> str:
            name = m.group(1).split("|")[0].strip()
            found = _resolve(name)
            if not found:
                return m.group(0)
            if found.suffix.lower() in _IMG_SUFFIXES:
                return f"![{name}](/api/fileviewer/raw?path={_url_quote(str(found))})"
            return f"[{name}](#file:{_url_quote(str(found))})"

        # Embeds run first, since `![[x]]` would otherwise match the plain `[[x]]` link pass.
        content = re.sub(r"!\[\[([^\]]+)\]\]", _repl_embed, content)

        def _repl_link(m: re.Match[str]) -> str:
            parts = m.group(1).split("|", 1)
            name = parts[0].strip()
            display = parts[1].strip() if len(parts) > 1 else name
            found = _resolve(name)
            if found:
                return f"[{display}](#file:{_url_quote(str(found))})"
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
