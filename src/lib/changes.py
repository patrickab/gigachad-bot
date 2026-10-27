"""Append rows to the per-user ``changes`` log that drives cross-device sync."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from psycopg import Connection


def record_change(
    connection: Connection, *, user_id: UUID, device_id: UUID | None, kind: str, key: str, operation: str, version: int | None = None,
) -> None:
    """Insert one change row. ``version`` is None for deletes, vacated move sources and vault roots."""
    connection.execute(
        """
        INSERT INTO changes (user_id, resource_kind, resource_key, version, operation, device_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (user_id, kind, key, version, operation, device_id),
    )
