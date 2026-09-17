"""PostgreSQL implementation of the application DataStore contract."""

from __future__ import annotations

from pathlib import PurePosixPath
from uuid import UUID

from psycopg_pool import ConnectionPool

from lib.data_store import (
    Entry,
    Revision,
    StorageConflictError,
    StorageNotFoundError,
    validate_key,
)


class PostgresDataStore:
    """A document store permanently scoped to one authenticated user."""

    def __init__(self, pool: ConnectionPool, user_id: UUID, *, device_id: UUID | None = None) -> None:
        self._pool = pool
        self._user_id = user_id
        self._device_id = device_id

    @staticmethod
    def _prefix(key: str) -> str:
        escaped = key.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return f"{escaped}/%" if escaped else "%"

    @staticmethod
    def _revision(version: int) -> Revision:
        return Revision(str(version))

    def _change(self, connection, key: str, version: int | None, operation: str) -> None:
        connection.execute(
            """
            INSERT INTO changes (user_id, resource_kind, resource_key, version, operation, device_id)
            VALUES (%s, 'document', %s, %s, %s, %s)
            """,
            (self._user_id, key, version, operation, self._device_id),
        )

    def read_bytes(self, key: str) -> tuple[bytes, Revision]:
        key = validate_key(key)
        with self._pool.connection() as connection:
            row = connection.execute(
                "SELECT content, version FROM documents WHERE user_id = %s AND key = %s AND NOT is_dir",
                (self._user_id, key),
            ).fetchone()
        if row is None:
            raise StorageNotFoundError(key)
        return bytes(row[0]), self._revision(row[1])

    def write_bytes(self, key: str, content: bytes, *, expected: Revision | None = None) -> Revision:
        key = validate_key(key)
        with self._pool.connection() as connection, connection.transaction():
            if expected is None:
                row = connection.execute(
                    """
                    INSERT INTO documents (user_id, key, content, is_dir)
                    VALUES (%s, %s, %s, false)
                    ON CONFLICT (user_id, key) DO UPDATE
                    SET content = EXCLUDED.content, is_dir = false,
                        version = documents.version + 1, updated_at = now()
                    RETURNING version
                    """,
                    (self._user_id, key, content),
                ).fetchone()
            else:
                try:
                    expected_version = int(expected.token)
                except ValueError as exc:
                    raise StorageConflictError(f"Invalid revision for {key}") from exc
                row = connection.execute(
                    """
                    UPDATE documents
                    SET content = %s, is_dir = false, version = version + 1, updated_at = now()
                    WHERE user_id = %s AND key = %s AND NOT is_dir AND version = %s
                    RETURNING version
                    """,
                    (content, self._user_id, key, expected_version),
                ).fetchone()
                if row is None:
                    raise StorageConflictError(f"{key} changed on another device; reload before saving")
            version = row[0]
            self._change(connection, key, version, "write")
        return self._revision(version)

    def list(self, prefix: str = "", *, recursive: bool = False) -> list[Entry]:
        prefix = validate_key(prefix, allow_empty=True)
        with self._pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT key, is_dir, octet_length(content), version
                FROM documents
                WHERE user_id = %s AND key LIKE %s ESCAPE '\\'
                ORDER BY key
                """,
                (self._user_id, self._prefix(prefix)),
            ).fetchall()

        entries: dict[str, Entry] = {}
        for key, is_dir, size, version in rows:
            relative = PurePosixPath(key).relative_to(prefix) if prefix else PurePosixPath(key)
            parts = relative.parts
            if recursive:
                for index in range(1, len(parts)):
                    ancestor = (PurePosixPath(prefix) / PurePosixPath(*parts[:index])).as_posix() if prefix else PurePosixPath(*parts[:index]).as_posix()
                    entries.setdefault(ancestor, Entry(ancestor, is_dir=True))
                entries[key] = Entry(key, is_dir=bool(is_dir), size=None if is_dir else size, revision=None if is_dir else self._revision(version))
            else:
                child = (PurePosixPath(prefix) / parts[0]).as_posix() if prefix else parts[0]
                if len(parts) == 1:
                    entries[child] = Entry(child, is_dir=bool(is_dir), size=None if is_dir else size, revision=None if is_dir else self._revision(version))
                else:
                    entries.setdefault(child, Entry(child, is_dir=True))
        return list(entries.values())

    def is_file(self, key: str) -> bool:
        key = validate_key(key)
        with self._pool.connection() as connection:
            return connection.execute(
                "SELECT EXISTS (SELECT 1 FROM documents WHERE user_id = %s AND key = %s AND NOT is_dir)",
                (self._user_id, key),
            ).fetchone()[0]

    def is_dir(self, key: str) -> bool:
        key = validate_key(key)
        with self._pool.connection() as connection:
            return connection.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM documents WHERE user_id = %s AND (key = %s AND is_dir OR key LIKE %s ESCAPE '\\')
                )
                """,
                (self._user_id, key, self._prefix(key)),
            ).fetchone()[0]

    def exists(self, key: str) -> bool:
        return self.is_file(key) or self.is_dir(key)

    def mkdir(self, key: str) -> None:
        key = validate_key(key)
        with self._pool.connection() as connection, connection.transaction():
            row = connection.execute(
                "SELECT is_dir FROM documents WHERE user_id = %s AND key = %s FOR UPDATE", (self._user_id, key)
            ).fetchone()
            if row is not None:
                if not row[0]:
                    raise FileExistsError(key)
                return
            connection.execute(
                "INSERT INTO documents (user_id, key, content, is_dir) VALUES (%s, %s, '', true)",
                (self._user_id, key),
            )
            self._change(connection, key, 1, "write")

    def delete(self, key: str, *, recursive: bool = False) -> None:
        key = validate_key(key)
        with self._pool.connection() as connection, connection.transaction():
            if not recursive:
                directory = connection.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM documents
                        WHERE user_id = %s AND (key = %s AND is_dir OR key LIKE %s ESCAPE '\\')
                    )
                    """,
                    (self._user_id, key, self._prefix(key)),
                ).fetchone()[0]
                child = connection.execute(
                    "SELECT 1 FROM documents WHERE user_id = %s AND key LIKE %s ESCAPE '\\' LIMIT 1",
                    (self._user_id, self._prefix(key)),
                ).fetchone()
                if directory and child is not None:
                    raise OSError(f"Directory not empty: {key}")
            connection.execute(
                "DELETE FROM documents WHERE user_id = %s AND (key = %s OR key LIKE %s ESCAPE '\\')",
                (self._user_id, key, self._prefix(key) if recursive else ""),
            )
            self._change(connection, key, None, "delete")

    def move(self, source: str, destination: str) -> None:
        source, destination = validate_key(source), validate_key(destination)
        with self._pool.connection() as connection, connection.transaction():
            rows = connection.execute(
                "SELECT key FROM documents WHERE user_id = %s AND (key = %s OR key LIKE %s ESCAPE '\\') FOR UPDATE",
                (self._user_id, source, self._prefix(source)),
            ).fetchall()
            if not rows:
                raise StorageNotFoundError(source)
            collision = connection.execute(
                "SELECT 1 FROM documents WHERE user_id = %s AND (key = %s OR key LIKE %s ESCAPE '\\') LIMIT 1",
                (self._user_id, destination, self._prefix(destination)),
            ).fetchone()
            if collision is not None:
                raise StorageConflictError(f"Destination already exists: {destination}")
            for (key,) in rows:
                suffix = key.removeprefix(source).lstrip("/")
                replacement = f"{destination}/{suffix}" if suffix else destination
                connection.execute(
                    "UPDATE documents SET key = %s, updated_at = now() WHERE user_id = %s AND key = %s",
                    (replacement, self._user_id, key),
                )
            self._change(connection, source, None, "move")
            self._change(connection, destination, None, "move")
