"""PostgreSQL implementation of the application DataStore contract."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from psycopg.types.json import Jsonb

from lib.changes import UserScope, like_prefix
from lib.data_store import (
    Entry,
    Meta,
    Revision,
    StorageConflictError,
    StorageNotFoundError,
    validate_key,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from psycopg import Connection


class PostgresDataStore:
    """A document store permanently scoped to one authenticated user."""

    def __init__(self, scope: UserScope) -> None:
        self._scope = scope

    def _entry(self, key: str, is_dir: bool, size: int, version: int) -> Entry:
        return Entry(key, is_dir=bool(is_dir), size=None if is_dir else size, revision=None if is_dir else self._revision(version))

    @staticmethod
    def _revision(version: int) -> Revision:
        return Revision(str(version))

    def read_bytes(self, key: str) -> tuple[bytes, Revision]:
        key = validate_key(key)
        with self._scope.connection() as connection:
            # Binary results move bytea as raw bytes instead of hex text, half the size and no decoding.
            row = connection.execute(
                "SELECT content, version FROM documents WHERE user_id = %s AND key = %s AND NOT is_dir",
                (self._scope.user_id, key),
                binary=True,
            ).fetchone()
        if row is None:
            raise StorageNotFoundError(key)
        return bytes(row[0]), self._revision(row[1])

    def read_many(self, keys: Iterable[str]) -> dict[str, tuple[bytes, Revision]]:
        """Content and revision of every file among *keys* in one query. Missing keys and directories are left out."""
        keys = [validate_key(key) for key in keys]
        if not keys:
            return {}
        with self._scope.connection() as connection:
            rows = connection.execute(
                "SELECT key, content, version FROM documents WHERE user_id = %s AND key = ANY(%s) AND NOT is_dir",
                (self._scope.user_id, keys),
                binary=True,
            ).fetchall()
        return {key: (bytes(content), self._revision(version)) for key, content, version in rows}

    def list_meta(self, prefix: str) -> dict[str, Meta | None]:
        """Meta of every file under *prefix* in listing order, None when none was stored with its current content."""
        prefix = validate_key(prefix, allow_empty=True)
        with self._scope.connection() as connection:
            # Meta is stamped with the version it was written for: a writer that leaves the column alone
            # (code predating it, project rename, canvas mutations) bumps the version and so retires it.
            rows = connection.execute(
                """
                SELECT key, CASE WHEN meta -> 'version' = to_jsonb(version) THEN meta - 'version' END
                FROM documents
                WHERE user_id = %s AND key LIKE %s ESCAPE '\\' AND NOT is_dir
                ORDER BY key
                """,
                (self._scope.user_id, like_prefix(prefix)),
            ).fetchall()
        return dict(rows)

    def write_meta(self, entries: Mapping[str, tuple[Meta, str]]) -> None:
        """Store ``(meta, sha256)`` per key on each file whose content still has that hex SHA-256, without a new revision."""
        params = [
            (stored, self._scope.user_id, validate_key(key), bytes.fromhex(sha256))
            for key, (meta, sha256) in entries.items()
            if (stored := _jsonb(meta)) is not None
        ]
        if not params:
            return
        # Guarded by content, not version, so meta derived from replaced content never lands on its successor.
        # Neither the version nor the change log moves: this is a cache fill, not a write.
        with self._scope.connection() as connection, connection.transaction(), connection.cursor() as cursor:
            cursor.executemany(
                """
                UPDATE documents SET meta = %s::jsonb || jsonb_build_object('version', version)
                WHERE user_id = %s AND key = %s AND NOT is_dir AND sha256 = %s
                """,
                params,
            )

    def _write(self, connection: Connection, key: str, content: bytes, expected: Revision | None, meta: Meta | None) -> Revision:
        """Upsert when *expected* is None, otherwise update only if the stored version matches, else raise StorageConflictError."""
        key = validate_key(key)
        stored = _jsonb(meta)
        # Meta is stamped with the version this statement writes, see `read_meta`.
        if expected is None:
            row = connection.execute(
                """
                INSERT INTO documents (user_id, key, content, is_dir, meta)
                VALUES (%s, %s, %s, false, %s::jsonb || jsonb_build_object('version', 1))
                ON CONFLICT (user_id, key) DO UPDATE
                SET content = EXCLUDED.content, is_dir = false,
                    version = documents.version + 1, updated_at = now(),
                    meta = EXCLUDED.meta || jsonb_build_object('version', documents.version + 1)
                RETURNING version
                """,
                (self._scope.user_id, key, content, stored),
            ).fetchone()
        else:
            try:
                expected_version = int(expected.token)
            except ValueError as exc:
                raise StorageConflictError(f"Invalid revision for {key}") from exc
            row = connection.execute(
                """
                UPDATE documents
                SET content = %s, is_dir = false, version = version + 1, updated_at = now(),
                    meta = %s::jsonb || jsonb_build_object('version', version + 1)
                WHERE user_id = %s AND key = %s AND NOT is_dir AND version = %s
                RETURNING version
                """,
                (content, stored, self._scope.user_id, key, expected_version),
            ).fetchone()
            if row is None:
                raise StorageConflictError(f"{key} changed on another device; reload before saving")
        version = row[0]
        self._scope.record(connection, kind="document", key=key, operation="write", version=version)
        return self._revision(version)

    def _delete(self, connection, key: str, recursive: bool) -> None:
        key = validate_key(key)
        if not recursive:
            child = connection.execute(
                "SELECT 1 FROM documents WHERE user_id = %s AND key LIKE %s ESCAPE '\\' LIMIT 1",
                (self._scope.user_id, like_prefix(key)),
            ).fetchone()
            if child is not None:
                raise OSError(f"Directory not empty: {key}")
        # Non-recursive deletes only the exact key, since an empty LIKE pattern matches nothing.
        connection.execute(
            "DELETE FROM documents WHERE user_id = %s AND (key = %s OR key LIKE %s ESCAPE '\\')",
            (self._scope.user_id, key, like_prefix(key) if recursive else ""),
        )
        self._scope.record(connection, kind="document", key=key, operation="delete")

    def write_bytes(self, key: str, content: bytes, *, expected: Revision | None = None, meta: Meta | None = None) -> Revision:
        with self._scope.connection() as connection, connection.transaction():
            return self._write(connection, key, content, expected, meta)

    def apply(
        self, writes: list[tuple[str, bytes, Revision | None]], deletes: list[str], *, meta: Mapping[str, Meta] | None = None
    ) -> list[Revision]:
        meta = meta or {}
        with self._scope.connection() as connection, connection.transaction():
            revisions = [self._write(connection, key, content, expected, meta.get(key)) for key, content, expected in writes]
            for key in deletes:
                self._delete(connection, key, recursive=False)
            return revisions

    def list(self, prefix: str = "", *, recursive: bool = False) -> list[Entry]:
        prefix = validate_key(prefix, allow_empty=True)
        with self._scope.connection() as connection:
            rows = connection.execute(
                """
                SELECT key, is_dir, octet_length(content), version
                FROM documents
                WHERE user_id = %s AND key LIKE %s ESCAPE '\\'
                ORDER BY key
                """,
                (self._scope.user_id, like_prefix(prefix)),
            ).fetchall()

        # Directories can be implicit (only descendants stored), so synthesize missing ancestors.
        entries: dict[str, Entry] = {}
        for key, is_dir, size, version in rows:
            parts = PurePosixPath(key).relative_to(prefix).parts
            if recursive:
                for index in range(1, len(parts)):
                    ancestor = PurePosixPath(prefix, *parts[:index]).as_posix()
                    entries.setdefault(ancestor, Entry(ancestor, is_dir=True))
                entries[key] = self._entry(key, is_dir, size, version)
            else:
                child = PurePosixPath(prefix, parts[0]).as_posix()
                if len(parts) == 1:
                    entries[child] = self._entry(child, is_dir, size, version)
                else:
                    entries.setdefault(child, Entry(child, is_dir=True))
        return list(entries.values())

    def is_file(self, key: str) -> bool:
        key = validate_key(key)
        with self._scope.connection() as connection:
            return connection.execute(
                "SELECT EXISTS (SELECT 1 FROM documents WHERE user_id = %s AND key = %s AND NOT is_dir)",
                (self._scope.user_id, key),
            ).fetchone()[0]

    def is_dir(self, key: str) -> bool:
        key = validate_key(key)
        with self._scope.connection() as connection:
            return connection.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM documents WHERE user_id = %s AND (key = %s AND is_dir OR key LIKE %s ESCAPE '\\')
                )
                """,
                (self._scope.user_id, key, like_prefix(key)),
            ).fetchone()[0]

    def exists(self, key: str) -> bool:
        """A file or directory at *key*, or any descendant of it: is_file or is_dir in one query."""
        key = validate_key(key)
        with self._scope.connection() as connection:
            return connection.execute(
                "SELECT EXISTS (SELECT 1 FROM documents WHERE user_id = %s AND (key = %s OR key LIKE %s ESCAPE '\\'))",
                (self._scope.user_id, key, like_prefix(key)),
            ).fetchone()[0]

    def stat(self, key: str) -> Entry | None:
        """The file or directory at *key*, or None: `exists` plus the file's size, revision and last write, without its content."""
        key = validate_key(key)
        with self._scope.connection() as connection:
            row = connection.execute(
                """
                SELECT key = %s, is_dir, octet_length(content), version, updated_at
                FROM documents
                WHERE user_id = %s AND (key = %s OR key LIKE %s ESCAPE '\\')
                ORDER BY key = %s DESC
                LIMIT 1
                """,
                (key, self._scope.user_id, key, like_prefix(key), key),
            ).fetchone()
        if row is None:
            return None
        exact, is_dir, size, version, modified = row
        if not exact or is_dir:
            return Entry(key, is_dir=True)
        return Entry(key, is_dir=False, size=size, revision=self._revision(version), modified=modified)

    def read_range(self, key: str, start: int, length: int) -> bytes:
        """Up to *length* bytes from offset *start*, sliced in the database: uncompressed TOAST fetches only the chunks it covers."""
        key = validate_key(key)
        with self._scope.connection() as connection:
            row = connection.execute(
                "SELECT substring(content FROM %s FOR %s) FROM documents WHERE user_id = %s AND key = %s AND NOT is_dir",
                (start + 1, length, self._scope.user_id, key),
                binary=True,
            ).fetchone()
        if row is None:
            raise StorageNotFoundError(key)
        return bytes(row[0])

    def sha256(self, key: str) -> str:
        """Hex SHA-256 of the file at *key*, kept by the database on every write, so the content is never read."""
        key = validate_key(key)
        with self._scope.connection() as connection:
            row = connection.execute(
                "SELECT encode(sha256, 'hex') FROM documents WHERE user_id = %s AND key = %s AND NOT is_dir",
                (self._scope.user_id, key),
            ).fetchone()
        if row is None:
            raise StorageNotFoundError(key)
        return row[0]

    def mkdir(self, key: str) -> None:
        key = validate_key(key)
        with self._scope.connection() as connection, connection.transaction():
            row = connection.execute(
                "SELECT is_dir FROM documents WHERE user_id = %s AND key = %s FOR UPDATE", (self._scope.user_id, key)
            ).fetchone()
            if row is not None:
                if not row[0]:
                    raise FileExistsError(key)
                return
            connection.execute(
                "INSERT INTO documents (user_id, key, content, is_dir) VALUES (%s, %s, '', true)",
                (self._scope.user_id, key),
            )
            self._scope.record(connection, kind="document", key=key, operation="write", version=1)

    def delete(self, key: str, *, recursive: bool = False) -> None:
        with self._scope.connection() as connection, connection.transaction():
            self._delete(connection, key, recursive)

    def move(self, source: str, destination: str) -> None:
        source, destination = validate_key(source), validate_key(destination)
        with self._scope.connection() as connection, connection.transaction():
            rows = connection.execute(
                "SELECT key FROM documents WHERE user_id = %s AND (key = %s OR key LIKE %s ESCAPE '\\') FOR UPDATE",
                (self._scope.user_id, source, like_prefix(source)),
            ).fetchall()
            if not rows:
                raise StorageNotFoundError(source)
            collision = connection.execute(
                "SELECT 1 FROM documents WHERE user_id = %s AND (key = %s OR key LIKE %s ESCAPE '\\') LIMIT 1",
                (self._scope.user_id, destination, like_prefix(destination)),
            ).fetchone()
            if collision is not None:
                raise StorageConflictError(f"Destination already exists: {destination}")
            # Rekey the source and every descendant under the destination.
            for (key,) in rows:
                suffix = key.removeprefix(source).lstrip("/")
                replacement = f"{destination}/{suffix}" if suffix else destination
                connection.execute(
                    "UPDATE documents SET key = %s, updated_at = now() WHERE user_id = %s AND key = %s",
                    (replacement, self._scope.user_id, key),
                )
            self._scope.record(connection, kind="document", key=source, operation="move")
            self._scope.record(connection, kind="document", key=destination, operation="move")


def _jsonb(meta: Meta | None) -> Jsonb | None:
    """*meta* as a jsonb parameter, None when absent or when jsonb would not hand its values back unchanged."""
    # Top-level key order is free (readers pick fields by name), nested values must survive exactly.
    if meta is None or not all(isinstance(key, str) and _jsonb_exact(key) and _jsonb_exact(value) for key, value in meta.items()):
        return None
    return Jsonb(meta)


def _jsonb_exact(value: object) -> bool:
    """Whether *value* survives a jsonb round trip as the same Python value with the same key order."""
    # jsonb rejects NUL and lone surrogates, may turn a float into an int, and sorts object keys
    # (shorter first, then bytewise). Such meta is not stored: readers derive it from the content.
    if value is None or isinstance(value, (bool, int)):
        return True
    if isinstance(value, str):
        return "\x00" not in value and (value.isascii() or _encodes(value))
    if isinstance(value, list):
        return all(_jsonb_exact(item) for item in value)
    if isinstance(value, dict):
        keys = list(value)
        return (
            all(isinstance(key, str) and _jsonb_exact(key) for key in keys)
            and keys == sorted(keys, key=lambda key: (len(key.encode()), key.encode()))
            and all(_jsonb_exact(item) for item in value.values())
        )
    return False


def _encodes(text: str) -> bool:
    """Whether *text* encodes as UTF-8, which a lone surrogate does not."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True
