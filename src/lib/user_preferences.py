"""Per-user display preferences, columns of the `users` row."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from psycopg_pool import ConnectionPool


class UserPreferences:
    """The animated app background setting of one user, on by default."""

    def __init__(self, pool: ConnectionPool, user_id: UUID) -> None:
        self._pool = pool
        self._user_id = user_id

    def animated_background(self) -> bool:
        with self._pool.connection() as connection:
            row = connection.execute("SELECT animated_background FROM users WHERE id = %s", (self._user_id,)).fetchone()
        return True if row is None else row[0]

    def set_animated_background(self, enabled: bool) -> None:
        with self._pool.connection() as connection:
            connection.execute("UPDATE users SET animated_background = %s WHERE id = %s", (enabled, self._user_id))
