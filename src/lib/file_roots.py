"""Access seam for file roots: directories of Attachment-compatible files.

A user can register several file roots (an Obsidian vault is just one flavor of file root). The list of roots is the
single source of truth for file locations (nothing is configured in ``config.py``): it lives in ``vault_roots`` rows
for the authenticated user (``file_roots_repository``). With no registry the roots are empty until the user adds one.

Listings (flat list, tree, relocation) honor ``.gitignore`` files (``file_walk``). Explicit paths (attach, read, write)
are not filtered, only validated against the roots. A root that is a git clone also admits its worktrees (``git_worktree``).

Root files are used as **live references**: attaching one to a chat stores its path, never a copy. Edits through the
app write back to the actual file. All paths handed to the UI are absolute, and every read/write is validated to live
under one of the roots. Logins with host access also reach their home directory outside the roots, except credential
stores (``host_access``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lib import git_worktree
from lib.file_walk import build_tree, walk_files
from lib.git import GitError
from lib.host_access import host_readable, linked_worktree_of
from lib.wiki_links import resolve_wiki_content

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lib.file_roots_repository import FileRootRepository


class FileRoots:
    """All access to the user's file-root roots goes through this class."""

    def __init__(self, *, repository: FileRootRepository, host_access: bool = False) -> None:
        self._repository = repository
        self._host_access = host_access
        self._roots: list[Path] = []
        self._worktrees: dict[Path, list[Path]] | None = None
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
                except GitError:
                    continue
                self._worktrees[candidate] = [Path(worktree.path) for worktree in listed if not worktree.is_main]
        return self._worktrees.get(root, [])

    def _accessible_roots(self, target: Path) -> list[Path]:
        """Registered roots holding *target*, else the clone worktrees holding it, so worktrees cost git calls only when used."""
        holding = [root for root in self._roots if target.is_relative_to(root)]
        return holding or [worktree for root in self._roots for worktree in self.worktrees_of(root) if target.is_relative_to(worktree)]

    def _root_for(self, path: str) -> Path:
        """Resolve *path*, rejecting anything outside all roots and their worktrees, unless host access reaches it."""
        target = Path(path).expanduser().resolve()
        if self._accessible_roots(target) or (self._host_access and host_readable(target)):
            return target
        raise ValueError(f"Path outside file roots: {path}")

    def contains(self, abs_path: Path) -> bool:
        try:
            self._root_for(str(abs_path))
            return True
        except ValueError:
            return False

    def root_of(self, path: str) -> Path:
        """Innermost registered root or clone worktree, else (host access) any linked git worktree holding *path*, else ValueError."""
        target = Path(path).expanduser().resolve()
        holding = self._accessible_roots(target)
        if not holding and self._host_access and (worktree := linked_worktree_of(target)):
            holding = [worktree]
        if not holding:
            raise ValueError(f"Path outside file roots: {path}")
        return max(holding, key=lambda root: len(root.parts))

    # ─── listing ───

    def list_files(self) -> list[dict[str, str]]:
        """Flat, searchable list of every file-root file as ``{path, name}`` (absolute paths) for the picker/attach flow."""
        files = [{"path": entry.path, "name": entry.name} for root in self._roots if root.is_dir() for entry in walk_files(root)]
        files.sort(key=lambda f: f["path"].lower())
        return files

    def files_under(self, paths: list[str]) -> list[Path]:
        """Files among *paths*, expanding folders recursively. Deduplicated, order kept."""
        found: dict[Path, None] = {}
        for path in paths:
            target = self._root_for(path)
            if target.is_dir():
                found.update(dict.fromkeys(Path(entry.path) for entry in walk_files(target)))
            elif target.is_file():
                found[target] = None
        return list(found)

    def relocate(self, path: str) -> Path | None:
        """Where the missing file *path* moved: the one visible file with its name in the root that held it, else None.

        - Several files with that name are ambiguous, and a path outside every root has no mount to search, so None rather than a guess
        """
        try:
            root = self.root_of(path)
        except ValueError:
            return None
        name = Path(path).name
        matches = {Path(entry.path) for entry in walk_files(root) if entry.name == name} if root.is_dir() else set()
        return matches.pop() if len(matches) == 1 else None

    def resolve_reference(self, path: str) -> Path:
        """`resolve` *path*, following the file to its new place when it moved. FileNotFoundError when it is gone or ambiguous."""
        try:
            return self.resolve(path)
        except FileNotFoundError:
            moved = self.relocate(path)
            if moved is None:
                raise
            return self.resolve(str(moved))

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
            children = build_tree(shown)
            nodes.append({"name": root.name, "path": str(shown), "mount": str(root), "type": "root", "children": children})
        return nodes

    # ─── file access ───

    def resolve(self, path: str) -> Path:
        """Validate *path* against the roots and require it to be an existing file."""
        resolved = self._root_for(path)
        if not resolved.is_file():
            raise FileNotFoundError(path)
        return resolved

    def resolve_folder(self, path: str) -> Path:
        """Validate *path* against the roots and require it to be an existing directory."""
        resolved = self._root_for(path)
        if not resolved.is_dir():
            raise NotADirectoryError(path)
        return resolved

    def read(self, path: str) -> str:
        """Return the text content of a file-root file."""
        return self.resolve(path).read_text(encoding="utf-8")

    def write(self, path: str, content: str) -> None:
        """Overwrite a file-root file in place (validated under a file root)."""
        self._root_for(path).write_text(content, encoding="utf-8")

    def resolve_wiki_content(self, content: str, source_path: str) -> str:
        """Rewrite ``![[embed]]``, ``[[link]]``, and relative images of *content* to standard markdown links to root files."""
        return resolve_wiki_content(content, source_path, self._roots, self.contains)
