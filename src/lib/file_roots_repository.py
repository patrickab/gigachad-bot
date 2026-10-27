"""Where a user's file root registry lives. File root file contents never pass through here."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from lib.changes import UserScope


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

    def __init__(self, scope: UserScope) -> None:
        self._scope = scope

    def load(self) -> list[str]:
        with self._scope.connection() as connection:
            rows = connection.execute(
                "SELECT path FROM vault_roots WHERE user_id = %s ORDER BY created_at, path",
                (self._scope.user_id,),
            ).fetchall()
        return [path for (path,) in rows]

    def save(self, roots: list[str]) -> None:
        # ponytail: whole-registry replace, mirroring the JSON file it replaces,
        # one statement per root. A user keeps a handful of roots; if that ever
        # grows, give FileRoots per-root add/remove hooks instead of diffing here.
        with self._scope.connection() as connection, connection.transaction():
            for path in roots:
                inserted = connection.execute(
                    "INSERT INTO vault_roots (user_id, path) VALUES (%s, %s) ON CONFLICT (user_id, path) DO NOTHING",
                    (self._scope.user_id, path),
                ).rowcount
                if inserted:
                    self._scope.record(connection, kind="vault_root", key=path, operation="write")
            removed = connection.execute(
                "DELETE FROM vault_roots WHERE user_id = %s AND NOT (path = ANY(%s::text[])) RETURNING path",
                (self._scope.user_id, roots),
            ).fetchall()
            for (path,) in removed:
                self._scope.record(connection, kind="vault_root", key=path, operation="delete")
