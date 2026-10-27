"""Per-user change log: `UserScope` (pool, user, device) writes rows, `latest_changes` and `head_cursor` read them for sync."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from uuid import UUID

    from psycopg import Connection
    from psycopg_pool import ConnectionPool

EVENT_KEYS = ("seq", "resource_kind", "resource_key", "version", "device_id")


def like_prefix(key: str) -> str:
    """LIKE pattern for descendants of *key* with wildcards escaped. Empty key matches everything."""
    escaped = key.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}/%" if escaped else "%"


@dataclass(frozen=True)
class UserScope:
    """One authenticated user (and the device acting for them) over the shared connection pool."""

    pool: ConnectionPool
    user_id: UUID
    device_id: UUID | None = None

    def connection(self) -> AbstractContextManager[Connection]:
        return self.pool.connection()

    def record(self, connection: Connection, *, kind: str, key: str, operation: str, version: int | None = None) -> None:
        """Insert one change row and prune the rows it supersedes. ``version`` is None for deletes, vacated sources and vault roots."""
        # Replay sends only the latest row per resource, so older rows of it carry nothing.
        # Rows another open transaction holds are skipped, not waited for: the next write of the resource prunes them.
        connection.execute(
            """
            WITH superseded AS (
                SELECT seq FROM changes
                WHERE user_id = %(user_id)s AND resource_kind = %(kind)s AND resource_key = %(key)s
                FOR UPDATE SKIP LOCKED
            ), pruned AS (
                DELETE FROM changes WHERE seq IN (SELECT seq FROM superseded)
            )
            INSERT INTO changes (user_id, resource_kind, resource_key, version, operation, device_id)
            VALUES (%(user_id)s, %(kind)s, %(key)s, %(version)s, %(operation)s, %(device_id)s)
            """,
            {"user_id": self.user_id, "kind": kind, "key": key, "version": version, "operation": operation,
             "device_id": self.device_id},
        )


def latest_changes(pool: ConnectionPool, user_id: UUID, since: int) -> list[dict]:
    """Return this user's latest change per resource after `since`, oldest first, keyed by `EVENT_KEYS`.

    - Clients reload by resource, so superseded rows add replay size and no information
    - `UserScope.record` prunes them, this query still collapses rows a concurrent write left behind
    """
    with pool.connection() as connection:
        rows = connection.execute(
            """
            SELECT seq, resource_kind, resource_key, version, device_id FROM (
                SELECT DISTINCT ON (resource_kind, resource_key)
                    seq, resource_kind, resource_key, version, device_id
                FROM changes
                WHERE user_id = %s AND seq > %s
                ORDER BY resource_kind, resource_key, seq DESC
            ) latest
            ORDER BY seq
            """,
            (user_id, since),
        ).fetchall()
    return [dict(zip(EVENT_KEYS, row, strict=True)) for row in rows]


def head_cursor(pool: ConnectionPool, user_id: UUID) -> int:
    """Return this user's highest change sequence, 0 before their first change."""
    with pool.connection() as connection:
        return connection.execute("SELECT coalesce(max(seq), 0) FROM changes WHERE user_id = %s", (user_id,)).fetchone()[0]
