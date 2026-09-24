"""PostgreSQL storage for binary application state (uploads, PDFs, drawings).

Assets are the counterpart to ``PostgresDataStore``'s text documents: opaque
bytes the app owns, addressed by a logical path and replicated to devices
through the ``changes`` log. Like the document store, an instance is
permanently scoped to one authenticated user — no operation takes a user id.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from uuid import UUID

from psycopg_pool import ConnectionPool

from lib.data_store import StorageConflictError, StorageNotFoundError, validate_key
from lib.document_library import mime_for

ASSET_KINDS = ("upload", "pdf", "mineru_markdown", "mineru_image", "drawing", "sandbox")

_COLUMNS = "id, kind, logical_path, mime, sha256, size_bytes, version"

_MIRROR_ROOTS = {
    "pdf": "PDFs",
    "mineru_markdown": "Mineru",
    "mineru_image": "Mineru",
}


@dataclass(frozen=True)
class Asset:
    id: UUID
    kind: str
    logical_path: str
    mime: str
    sha256: str
    size_bytes: int
    version: int
    content: bytes | None = None


class AssetStore:
    """A binary asset store permanently scoped to one authenticated user."""

    def __init__(self, pool: ConnectionPool, user_id: UUID, *, device_id: UUID | None = None) -> None:
        self._pool = pool
        self._user_id = user_id
        self._device_id = device_id

    @staticmethod
    def _escape_like(value: str) -> str:
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    @staticmethod
    def _asset(row, content: bytes | None = None) -> Asset:
        return Asset(
            id=row[0],
            kind=row[1],
            logical_path=row[2],
            mime=row[3],
            sha256=row[4],
            size_bytes=row[5],
            version=row[6],
            content=content,
        )

    def _change(self, connection, logical_path: str, version: int | None, operation: str) -> None:
        connection.execute(
            """
            INSERT INTO changes (user_id, resource_kind, resource_key, version, operation, device_id)
            VALUES (%s, 'asset', %s, %s, %s, %s)
            """,
            (self._user_id, logical_path, version, operation, self._device_id),
        )

    def write(self, kind: str, logical_path: str, content: bytes, *, mime: str | None = None) -> Asset:
        if kind not in ASSET_KINDS:
            raise ValueError(f"Unknown asset kind: {kind}")
        logical_path = validate_key(logical_path)
        mime = mime or mime_for(logical_path)
        sha256 = hashlib.sha256(content).hexdigest()
        with self._pool.connection() as connection, connection.transaction():
            row = connection.execute(
                """
                INSERT INTO assets (user_id, kind, logical_path, content, mime, sha256, size_bytes)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, logical_path) DO UPDATE
                SET kind = EXCLUDED.kind, content = EXCLUDED.content, mime = EXCLUDED.mime,
                    sha256 = EXCLUDED.sha256, size_bytes = EXCLUDED.size_bytes,
                    version = assets.version + 1, updated_at = now()
                RETURNING id, version
                """,
                (self._user_id, kind, logical_path, content, mime, sha256, len(content)),
            ).fetchone()
            self._change(connection, logical_path, row[1], "write")
        return Asset(
            id=row[0],
            kind=kind,
            logical_path=logical_path,
            mime=mime,
            sha256=sha256,
            size_bytes=len(content),
            version=row[1],
            content=content,
        )

    def read(self, logical_path: str) -> Asset:
        logical_path = validate_key(logical_path)
        with self._pool.connection() as connection:
            row = connection.execute(
                f"SELECT {_COLUMNS}, content FROM assets WHERE user_id = %s AND logical_path = %s",
                (self._user_id, logical_path),
            ).fetchone()
        if row is None:
            raise StorageNotFoundError(logical_path)
        return self._asset(row, bytes(row[7]))

    def read_by_id(self, asset_id: UUID) -> Asset:
        with self._pool.connection() as connection:
            row = connection.execute(
                f"SELECT {_COLUMNS}, content FROM assets WHERE user_id = %s AND id = %s",
                (self._user_id, asset_id),
            ).fetchone()
        if row is None:
            raise StorageNotFoundError(str(asset_id))
        return self._asset(row, bytes(row[7]))

    def list(self, prefix: str = "", *, kind: str | None = None) -> list[Asset]:
        prefix = validate_key(prefix, allow_empty=True)
        clauses = [f"SELECT {_COLUMNS} FROM assets WHERE user_id = %s"]
        params: list[object] = [self._user_id]
        if prefix:
            # Directory semantics: "notes" matches "notes" and "notes/a.md", never "notes-old/a.md".
            clauses.append("AND (logical_path = %s OR logical_path LIKE %s ESCAPE '\\')")
            params += [prefix, f"{self._escape_like(prefix)}/%"]
        if kind is not None:
            clauses.append("AND kind = %s")
            params.append(kind)
        clauses.append("ORDER BY logical_path")
        with self._pool.connection() as connection:
            rows = connection.execute(" ".join(clauses), params).fetchall()
        return [self._asset(row) for row in rows]

    def delete(self, logical_path: str) -> None:
        logical_path = validate_key(logical_path)
        with self._pool.connection() as connection, connection.transaction():
            deleted = connection.execute(
                "DELETE FROM assets WHERE user_id = %s AND logical_path = %s",
                (self._user_id, logical_path),
            ).rowcount
            if deleted:
                self._change(connection, logical_path, None, "delete")

    def move(self, source: str, destination: str) -> Asset:
        """Rename one asset while retaining its content and identity."""
        source, destination = validate_key(source), validate_key(destination)
        if source == destination:
            return self.read(source)
        with self._pool.connection() as connection, connection.transaction():
            exists = connection.execute(
                "SELECT 1 FROM assets WHERE user_id = %s AND logical_path = %s FOR UPDATE",
                (self._user_id, source),
            ).fetchone()
            if exists is None:
                raise StorageNotFoundError(source)
            collision = connection.execute(
                "SELECT 1 FROM assets WHERE user_id = %s AND logical_path = %s",
                (self._user_id, destination),
            ).fetchone()
            if collision is not None:
                raise StorageConflictError(f"Destination already exists: {destination}")
            row = connection.execute(
                f"""
                UPDATE assets SET logical_path = %s, version = version + 1, updated_at = now()
                WHERE user_id = %s AND logical_path = %s
                RETURNING {_COLUMNS}, content
                """,
                (destination, self._user_id, source),
            ).fetchone()
            self._change(connection, source, None, "move")
            self._change(connection, destination, row[6], "move")
        return self._asset(row, bytes(row[7]))

    def mirror(self, asset: Asset, root: Path) -> Path:
        """Atomically write an allowed PDF or MinerU asset beneath *root*.

        Other application state remains database-only. Never call this inside a
        database transaction: it reads missing content back through its own
        connection and then blocks on disk I/O.
        """
        mirror_root = _MIRROR_ROOTS.get(asset.kind)
        if mirror_root is None or not asset.logical_path.startswith(f"{mirror_root}/"):
            raise ValueError("Only PDF and MinerU assets may be mirrored to disk")
        content = asset.content if asset.content is not None else self.read_by_id(asset.id).content
        path = root.joinpath(*validate_key(asset.logical_path).split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(content)
            Path(temporary).replace(path)
        except OSError:
            Path(temporary).unlink(missing_ok=True)
            raise
        return path
