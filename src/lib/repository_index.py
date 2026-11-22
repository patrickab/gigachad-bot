"""Where Pi sessions may start: the folders `pi_host.scan_repositories` finds, stored in the `repositories` table.

- Requests read the stored list, so none walks `~` (a scan takes seconds on a full home folder)
- `sweep` scans the disk and writes the difference: new folders are added, vanished ones removed. The backend sweeps
  once at startup in the background, serving the list of the last run meanwhile
- A name the stored list lacks sweeps once more before it is refused (a folder created since), and a clone is added at once
- Tests swap in a store through `use`, None is the database
"""

from __future__ import annotations

import logging
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Protocol

from lib import pi_host
from lib.pi_host import Repository

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool

log = logging.getLogger(__name__)


class RepositoryStore(Protocol):
    def load(self) -> list[Repository]:
        """Stored repositories in scan order."""

    def replace(self, repositories: list[Repository]) -> tuple[int, int]:
        """Store exactly *repositories*, returning how many were added and removed."""

    def add(self, repository: Repository) -> None:
        """Store *repository* after the others, replacing one of the same name."""


class PostgresRepositoryStore:
    """The `repositories` rows, host-wide: every user with host access sees the same folders."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def load(self) -> list[Repository]:
        with self._pool.connection() as connection:
            rows = connection.execute("SELECT name, path FROM repositories ORDER BY position").fetchall()
        return [Repository(name=name, path=path) for name, path in rows]

    def replace(self, repositories: list[Repository]) -> tuple[int, int]:
        names = [repository.name for repository in repositories]
        with self._pool.connection() as connection, connection.transaction():
            removed = connection.execute("DELETE FROM repositories WHERE NOT (name = ANY(%s::text[]))", (names,)).rowcount
            added = 0
            for position, repository in enumerate(repositories):
                (inserted,) = connection.execute(
                    """
                    INSERT INTO repositories (name, path, position) VALUES (%s, %s, %s)
                    ON CONFLICT (name) DO UPDATE SET path = excluded.path, position = excluded.position
                    RETURNING xmax = 0
                    """,
                    (repository.name, repository.path, position),
                ).fetchone()
                added += inserted
        return added, removed

    def add(self, repository: Repository) -> None:
        with self._pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO repositories (name, path, position)
                VALUES (%s, %s, (SELECT coalesce(max(position) + 1, 0) FROM repositories))
                ON CONFLICT (name) DO UPDATE SET path = excluded.path
                """,
                (repository.name, repository.path),
            )


_store: RepositoryStore | None = None
_sweeping = threading.Lock()


def use(store: RepositoryStore | None) -> None:
    """Read and write *store* from now on, None for the database."""
    global _store
    _store = store


def _current() -> RepositoryStore:
    if _store is None:
        from lib.postgres_pool import get_postgres_pool

        use(PostgresRepositoryStore(get_postgres_pool()))
    assert _store is not None
    return _store


def sweep() -> list[Repository]:
    """Scan the disk and store what it holds now. Concurrent sweeps run one after the other."""
    with _sweeping:
        found = pi_host.scan_repositories()
        added, removed = _current().replace(found)
    if added or removed:
        log.info("Repositories: %d added, %d removed, %d stored", added, removed, len(found))
    return found


def sweep_quietly() -> None:
    """`sweep` for the startup task: a failure is logged, the stored list stays."""
    try:
        sweep()
    except Exception:
        log.exception("Repository sweep failed, keeping the stored list")


def list_repositories() -> list[Repository]:
    """The stored repositories, scanned first when none are stored (the first start)."""
    return _current().load() or sweep()


def resolve(name: str) -> Path | None:
    """The folder of the repository *name*, sweeping once when it is not stored or gone. None for an unknown name."""
    for repositories in (_current().load, sweep):
        folder = next((Path(repository.path) for repository in repositories() if repository.name == name), None)
        if folder is not None and folder.is_dir():
            return folder
    return None


def add(folder: Path) -> None:
    """Store *folder*, a repository just created on disk, without a sweep. A folder outside `~` is no start folder."""
    try:
        name = pi_host.repository_name(folder)
    except ValueError:
        return
    _current().add(Repository(name=name, path=str(folder)))
