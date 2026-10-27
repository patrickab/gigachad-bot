"""PostgreSQL storage for binary application state (uploads, PDFs, drawings).

Assets are the counterpart to ``PostgresDataStore``'s text documents: opaque
bytes the app owns, addressed by a logical path and replicated to devices
through the ``changes`` log. Like the document store, an instance is
permanently scoped to one authenticated user: no operation takes a user id.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from uuid import UUID

from lib.changes import UserScope, like_prefix
from lib.data_store import StorageNotFoundError, validate_key
from lib.mime import mime_for
from lib.storage_namespace import MINERU, PDFS

ASSET_KINDS = ("upload", "pdf", "mineru_markdown", "mineru_image", "drawing", "sandbox")

_COLUMNS = "id, kind, logical_path, mime, sha256, size_bytes, version"

# Top-level path segment each mirrorable kind must live under on disk.
_MIRROR_ROOTS = {
    "pdf": PDFS,
    "mineru_markdown": MINERU,
    "mineru_image": MINERU,
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

    def __init__(self, scope: UserScope) -> None:
        self._scope = scope

    # Row layout is ``_COLUMNS`` order, optionally followed by ``content``.
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

    def write(self, kind: str, logical_path: str, content: bytes, *, mime: str | None = None) -> Asset:
        """Upsert by path. Overwriting bumps ``version``, and the replicated change carries it."""
        if kind not in ASSET_KINDS:
            raise ValueError(f"Unknown asset kind: {kind}")
        logical_path = validate_key(logical_path)
        mime = mime or mime_for(logical_path)
        sha256 = hashlib.sha256(content).hexdigest()
        with self._scope.connection() as connection, connection.transaction():
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
                (self._scope.user_id, kind, logical_path, content, mime, sha256, len(content)),
            ).fetchone()
            self._scope.record(connection, kind="asset", key=logical_path, operation="write", version=row[1])
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

    def _fetch(self, column: str, value: object, missing: str) -> Asset:
        with self._scope.connection() as connection:
            # Binary results move bytea as raw bytes instead of hex text, half the size and no decoding.
            row = connection.execute(
                f"SELECT {_COLUMNS}, content FROM assets WHERE user_id = %s AND {column} = %s",
                (self._scope.user_id, value),
                binary=True,
            ).fetchone()
        if row is None:
            raise StorageNotFoundError(missing)
        return self._asset(row, bytes(row[7]))

    def read(self, logical_path: str) -> Asset:
        logical_path = validate_key(logical_path)
        return self._fetch("logical_path", logical_path, logical_path)

    def read_by_id(self, asset_id: UUID) -> Asset:
        return self._fetch("id", asset_id, str(asset_id))

    def stat(self, logical_path: str) -> Asset | None:
        """The asset at *logical_path* without its content (size, hash and version only), or None when it is not stored."""
        logical_path = validate_key(logical_path)
        with self._scope.connection() as connection:
            row = connection.execute(
                f"SELECT {_COLUMNS} FROM assets WHERE user_id = %s AND logical_path = %s",
                (self._scope.user_id, logical_path),
            ).fetchone()
        return None if row is None else self._asset(row)

    def read_range(self, logical_path: str, start: int, length: int) -> bytes:
        """Up to *length* bytes from offset *start*, sliced in the database: uncompressed TOAST fetches only the chunks it covers."""
        logical_path = validate_key(logical_path)
        with self._scope.connection() as connection:
            row = connection.execute(
                "SELECT substring(content FROM %s FOR %s) FROM assets WHERE user_id = %s AND logical_path = %s",
                (start + 1, length, self._scope.user_id, logical_path),
                binary=True,
            ).fetchone()
        if row is None:
            raise StorageNotFoundError(logical_path)
        return bytes(row[0])

    def list(self, prefix: str = "", *, kind: str | None = None) -> list[Asset]:
        """Assets under *prefix* (directory semantics) without content, ordered by path."""
        prefix = validate_key(prefix, allow_empty=True)
        clauses = [f"SELECT {_COLUMNS} FROM assets WHERE user_id = %s"]
        params: list[object] = [self._scope.user_id]
        if prefix:
            # Directory semantics: "notes" matches "notes" and "notes/a.md", never "notes-old/a.md".
            clauses.append("AND (logical_path = %s OR logical_path LIKE %s ESCAPE '\\')")
            params += [prefix, like_prefix(prefix)]
        if kind is not None:
            clauses.append("AND kind = %s")
            params.append(kind)
        clauses.append("ORDER BY logical_path")
        with self._scope.connection() as connection:
            rows = connection.execute(" ".join(clauses), params).fetchall()
        return [self._asset(row) for row in rows]

    def delete(self, logical_path: str) -> None:
        logical_path = validate_key(logical_path)
        with self._scope.connection() as connection, connection.transaction():
            deleted = connection.execute(
                "DELETE FROM assets WHERE user_id = %s AND logical_path = %s",
                (self._scope.user_id, logical_path),
            ).rowcount
            if deleted:
                self._scope.record(connection, kind="asset", key=logical_path, operation="delete")

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
