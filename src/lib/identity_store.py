"""Database users and browser devices behind a trusted login."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from psycopg_pool import ConnectionPool

# A known device's `last_seen_at` is refreshed at most this often, so steady traffic only reads.
DEVICE_SEEN_INTERVAL = timedelta(minutes=1)


def resolve_user(pool: ConnectionPool, login: str, device_id: UUID | None) -> UUID:
    """Return the user owning *login*, registering the user and *device_id* when new, reassigned or not seen recently."""
    # Every request resolves its identity. Upserting each time made every request write and commit,
    # and concurrent requests queued on the same users row lock, so the common case only reads.
    with pool.connection() as connection:
        known = connection.execute(
            """
            SELECT users.id, devices.user_id IS NOT DISTINCT FROM users.id AND devices.last_seen_at > now() - %s
            FROM users LEFT JOIN devices ON devices.id = %s
            WHERE users.tailscale_login = %s
            """,
            (DEVICE_SEEN_INTERVAL, device_id, login),
        ).fetchone()
    if known is not None and (device_id is None or known[1]):
        return known[0]
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
