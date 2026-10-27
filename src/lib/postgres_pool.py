"""The process-wide PostgreSQL connection pool."""

from __future__ import annotations

import os

from psycopg_pool import ConnectionPool

from lib.db_schema import database_url

_postgres_pool: ConnectionPool | None = None


def get_postgres_pool() -> ConnectionPool:
    """Return the process-wide connection pool for Postgres storage."""
    global _postgres_pool
    if _postgres_pool is None:
        _postgres_pool = ConnectionPool(
            database_url(),
            min_size=int(os.environ.get("GIGACHAD_PG_POOL_MIN_SIZE", "1")),
            max_size=int(os.environ.get("GIGACHAD_PG_POOL_MAX_SIZE", "10")),
        )
    return _postgres_pool


def close_postgres_pool() -> None:
    """Release database connections during backend shutdown."""
    global _postgres_pool
    if _postgres_pool is not None:
        _postgres_pool.close()
        _postgres_pool = None
