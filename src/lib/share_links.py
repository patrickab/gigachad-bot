"""Share links: a revocable secret granting one stored resource of its owner to whoever holds it.

- Only the secret's SHA-256 is stored: the secret is shown once, at creation
- One link per resource: creating another rotates the secret, revoking deletes it
- The row references its resource, so a link follows renames and dies with the resource
- How a link opens and what it allows follow from its resource (`ShareTarget.view`, `.access`):
  canvases are edited together, all else is viewed
- A new resource kind is a `Kind`, a `_KINDS` entry and its referencing column in `share_links`
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import secrets
from typing import TYPE_CHECKING, Literal

from psycopg.errors import ForeignKeyViolation

from lib.asset_store import AssetStore
from lib.canvas_store import is_canvas_key
from lib.mime import mime_for
from lib.postgres_data_store import PostgresDataStore

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from psycopg_pool import ConnectionPool

    from lib.changes import UserScope

Kind = Literal["document", "asset"]
Access = Literal["view", "edit"]
View = Literal["canvas", "content"]


class ShareTargetNotFound(LookupError):
    """The owner has no such resource to share."""


@dataclass(frozen=True)
class _ResourceKind:
    column: str  # the `share_links` column referencing this kind's resource. Fixed, never user input
    read: Callable[[UserScope, str], tuple[bytes, str]]  # the resource's bytes and MIME type


def _read_document(scope: UserScope, key: str) -> tuple[bytes, str]:
    return PostgresDataStore(scope).read_bytes(key)[0], mime_for(key)


def _read_asset(scope: UserScope, key: str) -> tuple[bytes, str]:
    asset = AssetStore(scope).read(key)
    return asset.content, asset.mime


_KINDS: dict[Kind, _ResourceKind] = {
    "document": _ResourceKind("document_key", _read_document),
    "asset": _ResourceKind("asset_path", _read_asset),
}


@dataclass(frozen=True)
class ShareTarget:
    """One shareable resource of a user: a document key or an asset path."""

    kind: Kind
    key: str

    @property
    def view(self) -> View:
        return "canvas" if self.kind == "document" and is_canvas_key(self.key) else "content"

    @property
    def access(self) -> Access:
        return "edit" if self.view == "canvas" else "view"

    def read_content(self, owner: UserScope) -> tuple[bytes, str]:
        """The resource's bytes and MIME type, read as its owner."""
        return _KINDS[self.kind].read(owner, self.key)


@dataclass(frozen=True)
class ShareGrant:
    """What a live secret grants: its owner's resource, acted on with the owner's authority."""

    owner: UUID
    target: ShareTarget


def _hash(secret: str) -> bytes:
    # 256 random bits cannot be brute-forced, so a fast unsalted hash suffices.
    return hashlib.sha256(secret.encode()).digest()


class ShareLinks:
    """The owner's side: create, inspect and revoke the links of their own resources."""

    def __init__(self, scope: UserScope) -> None:
        self._scope = scope

    def is_shared(self, target: ShareTarget) -> bool:
        with self._scope.connection() as connection:
            return connection.execute(
                f"SELECT 1 FROM share_links WHERE user_id = %s AND {_KINDS[target.kind].column} = %s",
                (self._scope.user_id, target.key),
            ).fetchone() is not None

    def create(self, target: ShareTarget) -> str:
        """Issue the resource's secret, invalidating any previous link to it."""
        secret = secrets.token_urlsafe(32)
        column = _KINDS[target.kind].column
        try:
            with self._scope.connection() as connection:
                connection.execute(
                    f"""
                    INSERT INTO share_links (secret_hash, user_id, {column}) VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, {column}) DO UPDATE SET secret_hash = EXCLUDED.secret_hash, created_at = now()
                    """,
                    (_hash(secret), self._scope.user_id, target.key),
                )
        except ForeignKeyViolation as exc:  # the referenced resource does not exist
            raise ShareTargetNotFound(f"No {target.kind} {target.key!r} to share") from exc
        return secret

    def revoke(self, target: ShareTarget) -> None:
        with self._scope.connection() as connection:
            connection.execute(
                f"DELETE FROM share_links WHERE user_id = %s AND {_KINDS[target.kind].column} = %s",
                (self._scope.user_id, target.key),
            )


def find_grant(pool: ConnectionPool, secret: str) -> ShareGrant | None:
    """What a live secret grants, None for an unknown, rotated or revoked one."""
    kinds = list(_KINDS)
    with pool.connection() as connection:
        row = connection.execute(
            f"SELECT user_id, {', '.join(_KINDS[kind].column for kind in kinds)} FROM share_links WHERE secret_hash = %s",
            (_hash(secret),),
        ).fetchone()
    if row is None:
        return None
    owner, *keys = row
    # The table's CHECK leaves exactly one target column set.
    kind, key = next((kind, key) for kind, key in zip(kinds, keys, strict=True) if key is not None)
    return ShareGrant(owner, ShareTarget(kind, key))
