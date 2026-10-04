"""Storage for LikeC4 architecture workspaces.

A workspace is a location (a database prefix or a directory on disk) and every
``*.c4`` file from there downward. Files may nest (``backend/backend.c4``);
LikeC4 merges them into one model. Saved positions are LikeC4's own manual
layouts, ``.likec4/<viewId>.likec4.snap`` at the workspace root. Parsing,
layout, and edits happen in the C4 service; this module only reads the files
and commits what changed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Protocol

from lib.data_store import (
    DataStore,
    Entry,
    InvalidStorageKey,
    Revision,
    StorageConflictError,
    StorageError,
    StorageNotFoundError,
    read_text,
    validate_key,
)
from lib.storage_namespace import architecture_prefix

SOURCE_SUFFIX = ".c4"
# LikeC4's default manual-layout location (its `manualLayouts.outDir`).
SNAPSHOT_DIR = ".likec4/"
SNAPSHOT_SUFFIX = ".likec4.snap"


class ArchitectureError(ValueError):
    """A request the workspace cannot honour."""


class ArchitectureNotFound(ArchitectureError, FileNotFoundError):
    """The project has no architecture workspace yet."""


@dataclass(frozen=True)
class Snapshot:
    """Workspace contents plus the revisions a commit must still match."""

    sources: dict[str, str]
    # Workspace-relative path -> text of every saved layout, e.g. ``.likec4/index.likec4.snap``.
    snapshots: dict[str, str]
    revisions: dict[str, Revision]


class WorkspaceFiles(Protocol):
    """The storage a workspace needs; ``PostgresDataStore`` and ``DirectoryFiles`` both provide it."""

    def read_bytes(self, key: str) -> tuple[bytes, Revision]: ...
    def write_many(self, writes: list[tuple[str, bytes, Revision | None]]) -> list[Revision]: ...
    def list(self, prefix: str = "", *, recursive: bool = False) -> list[Entry]: ...
    def delete(self, key: str) -> None: ...


class DirectoryFiles:
    """A directory on disk as workspace storage, e.g. a checked-out repository.

    Revisions are content digests. ``write_many`` checks every revision before
    writing anything, but unlike the database the writes are not one transaction.
    Inside a git work tree, listing follows git: ignored files and ``.git`` are
    skipped, exactly as ``git status`` sees them. Symlinks are never listed.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def _path(self, key: str) -> Path:
        path = (self._root / validate_key(key)).resolve()
        if not path.is_relative_to(self._root):
            raise InvalidStorageKey(f"{key} leaves the workspace directory")
        return path

    @staticmethod
    def _revision(content: bytes) -> Revision:
        return Revision(hashlib.sha256(content).hexdigest())

    def read_bytes(self, key: str) -> tuple[bytes, Revision]:
        try:
            content = self._path(key).read_bytes()
        except (FileNotFoundError, IsADirectoryError) as exc:
            raise StorageNotFoundError(key) from exc
        return content, self._revision(content)

    def write_many(self, writes: list[tuple[str, bytes, Revision | None]]) -> list[Revision]:
        for key, _, expected in writes:
            if expected is None:
                continue
            try:
                current = self.read_bytes(key)[1]
            except StorageNotFoundError:
                current = None
            if current != expected:
                raise StorageConflictError(f"{key} changed on disk; reload before saving")
        revisions = []
        for key, content, _ in writes:
            path = self._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            staged = path.with_name(f".{path.name}.tmp")
            staged.write_bytes(content)
            os.replace(staged, path)
            revisions.append(self._revision(content))
        return revisions

    def _files_under(self, base: Path) -> list[Path]:
        """Every file below *base* git does not ignore; every file when *base* is outside a work tree."""
        if not base.is_dir():
            return []
        listed = subprocess.run(
            ["git", "-C", str(base), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            capture_output=True,
            check=False,
        )
        if listed.returncode == 0:
            # Tracked files deleted from disk are still listed; they are not part of the workspace.
            candidates = [base / name for name in sorted(listed.stdout.decode().split("\0")) if name]
        elif b"not a git repository" in listed.stderr:
            candidates = sorted(Path(directory) / name for directory, _, names in os.walk(base) for name in names)
        else:
            raise StorageError(f"git cannot list {base}: {listed.stderr.decode().strip()}")
        return [path for path in candidates if path.is_file() and not path.is_symlink()]

    def list(self, prefix: str = "", *, recursive: bool = False) -> list[Entry]:
        base = self._path(prefix) if prefix else self._root
        entries: dict[str, Entry] = {}
        for path in self._files_under(base):
            relative = path.relative_to(base)
            if len(relative.parts) > 1 and not recursive:
                child = (base / relative.parts[0]).relative_to(self._root).as_posix()
                entries.setdefault(child, Entry(child, is_dir=True))
                continue
            key = path.relative_to(self._root).as_posix()
            entries[key] = Entry(key, is_dir=False, size=path.stat().st_size)
        return list(entries.values())

    def delete(self, key: str) -> None:
        try:
            self._path(key).unlink()
        except FileNotFoundError as exc:
            raise StorageNotFoundError(key) from exc


class ArchitectureStore:
    """Reads and commits the workspace rooted at one location of *files*."""

    def __init__(self, files: WorkspaceFiles, root: str = "") -> None:
        self._files = files
        self._root = validate_key(root, allow_empty=True)

    @classmethod
    def for_project(cls, docs: DataStore, slug: str) -> ArchitectureStore:
        """The project's workspace in the database, under ``graph/<slug>/``."""
        return cls(docs, architecture_prefix(slug))

    def key(self, relative: str) -> str:
        """Storage key of a workspace-relative path."""
        return f"{self._root}/{relative}" if self._root else relative

    def source_key(self, relative: str) -> str:
        try:
            normalized = validate_key(relative)
        except InvalidStorageKey as exc:
            raise ArchitectureError(f"Not a workspace source file: {relative}") from exc
        if normalized != relative or not relative.endswith(SOURCE_SUFFIX):
            raise ArchitectureError(f"Not a workspace source file: {relative}")
        return self.key(relative)

    def _entries(self) -> list[tuple[str, Entry]]:
        """``(workspace-relative path, entry)`` for every file from the root downward."""
        skip = len(self._root) + 1 if self._root else 0
        return [(entry.key[skip:], entry) for entry in self._files.list(self._root, recursive=True) if not entry.is_dir]

    def source_paths(self) -> list[str]:
        """Storage keys of every ``.c4`` source, in path order."""
        return sorted(entry.key for relative, entry in self._entries() if relative.endswith(SOURCE_SUFFIX))

    def read(self, *, allow_empty: bool = False) -> Snapshot:
        """Every source plus the saved layouts; a workspace without sources is not found unless *allow_empty*."""
        sources: dict[str, str] = {}
        snapshots: dict[str, str] = {}
        revisions: dict[str, Revision] = {}
        for relative, entry in self._entries():
            if relative.endswith(SOURCE_SUFFIX):
                sources[relative], revisions[entry.key] = read_text(self._files, entry.key)
            elif relative.startswith(SNAPSHOT_DIR) and relative.endswith(SNAPSHOT_SUFFIX):
                snapshots[relative], revisions[entry.key] = read_text(self._files, entry.key)
        if not sources and not allow_empty:
            raise ArchitectureNotFound("This project has no architecture yet")
        return Snapshot(sources=sources, snapshots=snapshots, revisions=revisions)

    def commit(self, snapshot: Snapshot, sources: dict[str, str], snapshots: dict[str, str], deleted: Iterable[str] = ()) -> None:
        """Write what changed since *snapshot* in one transaction; StorageConflictError when a changed file moved on.

        Only files that existed in *snapshot* and changed are revision-checked. New files
        overwrite whatever appeared meanwhile. The *deleted* sources and the layouts of
        views that no longer exist are deleted afterwards, one by one and unchecked. A
        leftover layout is harmless, as nothing reads one without its view.
        """
        gone = [self.source_key(relative) for relative in sorted(deleted)]
        gone += [self.key(relative) for relative in sorted(set(snapshot.snapshots) - set(snapshots))]
        writes: list[tuple[str, bytes, Revision | None]] = []
        for relative, text in sorted(sources.items()):
            if snapshot.sources.get(relative) != text:
                key = self.source_key(relative)
                writes.append((key, text.encode("utf-8"), snapshot.revisions.get(key)))
        for relative, text in sorted(snapshots.items()):
            if not (relative.startswith(SNAPSHOT_DIR) and relative.endswith(SNAPSHOT_SUFFIX)):
                raise ArchitectureError(f"Not a layout file: {relative}")
            if snapshot.snapshots.get(relative) != text:
                key = self.key(validate_key(relative))
                writes.append((key, text.encode("utf-8"), snapshot.revisions.get(key)))
        if writes:
            self._files.write_many(writes)
        for key in gone:
            try:
                self._files.delete(key)
            except StorageNotFoundError:
                pass

