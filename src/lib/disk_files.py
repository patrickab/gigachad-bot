"""Writes of file-root files that tell the user's other devices.

Every write through the app records one ``document`` change keyed by the resolved path. Disk files carry no version, so
the change row's version is None.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lib.changes import UserScope
    from lib.file_roots import FileRoots


class DiskFiles:
    """Creates and overwrites files under the file roots and records the change."""

    def __init__(self, roots: FileRoots, scope: UserScope) -> None:
        self._roots = roots
        self._scope = scope

    def create(self, path: str | Path, data: str | bytes) -> Path:
        """Create *path* exclusively (FileExistsError when taken, ValueError outside the roots) and record it."""
        created = self._roots.create(str(path), data)
        self.record(created)
        return created

    def write(self, path: str, content: str) -> Path:
        """Overwrite *path* (ValueError outside the roots) and record it."""
        written = self._roots.write(path, content)
        self.record(written)
        return written

    def record(self, path: Path) -> None:
        """Record the change of *path* that a writer already made."""
        with self._scope.connection() as connection, connection.transaction():
            self._scope.record(connection, kind="document", key=str(path), operation="write")
