"""Database users and browser devices behind a trusted login."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from psycopg_pool import ConnectionPool


def resolve_user(pool: ConnectionPool, login: str, device_id: UUID | None) -> UUID:
    """Upsert the user owning *login*, register *device_id* to them when given, and return the user id."""
    with pool.connection() as connection, connection.transaction():
        user_id = connection.execute(
            """
            INSERT INTO users (tailscale_login) VALUES (%s)
            ON CONFLICT (tailscale_login) DO UPDATE SET tailscale_login = EXCLUDED.tailscale_login
            RETURNING id
            """,
            (login,),
        ).fetchone()[0]
        if device_id is not None:
            connection.execute(
                """
                INSERT INTO devices (id, user_id) VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET user_id = EXCLUDED.user_id, last_seen_at = now()
                """,
                (device_id, user_id),
            )
    return user_id
