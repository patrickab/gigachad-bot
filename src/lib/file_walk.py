"""Ignore-aware listing of a directory, as the file tree and the flat file lists show it.

`.gitignore` files count from the listed directory downward with git's precedence: the last matching pattern wins
and deeper files override shallower ones. Ignored folders are never entered and directory symlinks never followed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from pathspec import GitIgnoreSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

# Directories that hold app internals (e.g. Obsidian's) or noise rather than files, skipped even without a .gitignore.
_SKIP_DIRS = frozenset({".obsidian", ".trash", ".git", "node_modules"})
# Each layer is one directory's `.gitignore`: (directory path, compiled patterns), outermost first.
_IgnoreLayers = tuple[tuple[str, GitIgnoreSpec], ...]


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


def walk_files(base: Path | str, inherited: _IgnoreLayers = ()) -> Iterator[os.DirEntry]:
    """Yield listed files below *base*, skipping ignored entries and not following directory symlinks."""
    entries, layers = _visible_entries(base, inherited)
    for entry, is_dir in entries:
        if is_dir:
            yield from walk_files(entry.path, layers)
        elif entry.is_file():
            yield entry


def build_tree(directory: Path | str, inherited: _IgnoreLayers = ()) -> list[dict]:
    """Folder and file nodes below *directory*, folders without listed files pruned."""
    # Folders first, then files, each alphabetical, matches a file browser.
    entries, layers = _visible_entries(directory, inherited)
    children: list[dict] = []
    for entry, is_dir in sorted(entries, key=lambda item: (not item[1], item[0].name.lower())):
        if is_dir:
            sub = build_tree(entry.path, layers)
            if sub:
                children.append({"name": entry.name, "path": entry.path, "type": "folder", "children": sub})
        elif entry.is_file():
            children.append({"name": entry.name, "path": entry.path, "type": "file"})
    return children
