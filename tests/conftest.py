"""Shared pytest fixtures."""

import os
from pathlib import Path
import subprocess
from uuid import UUID, uuid4

from psycopg_pool import ConnectionPool
import pytest

from lib.db_schema import upgrade

TRUNCATE_ALL = "TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE"


def create_user(pool: ConnectionPool, login: str | None = None) -> UUID:
    """Insert a user (a unique throwaway login by default) and return its id."""
    login = login or f"test-{uuid4()}@example.test"
    with pool.connection() as connection, connection.transaction():
        return connection.execute(
            "INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", (login,)
        ).fetchone()[0]


def create_device(pool: ConnectionPool, user_id: UUID) -> UUID:
    device_id = uuid4()
    with pool.connection() as connection, connection.transaction():
        connection.execute("INSERT INTO devices (id, user_id) VALUES (%s, %s)", (device_id, user_id))
    return device_id


def create_user_and_device(pool: ConnectionPool, login: str = "alice@example.test") -> tuple[UUID, UUID]:
    user_id = create_user(pool, login)
    return user_id, create_device(pool, user_id)


def git(directory, *args: str) -> str:
    """Run git in *directory* with a fixed committer identity; returns stripped stdout."""
    return subprocess.run(
        ["git", "-C", str(directory), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture(scope="module")
def database_url() -> str:
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for Postgres-backed tests")
    upgrade(url)
    return url


@pytest.fixture(scope="module")
def postgres_pool(database_url: str):
    pool = ConnectionPool(database_url, min_size=1, max_size=4)
    yield pool
    pool.close()


@pytest.fixture
def clean_database(postgres_pool: ConnectionPool) -> None:
    """Empty every user-owned table. Modules that want it for every test apply it with
    ``pytestmark = pytest.mark.usefixtures("clean_database")``."""
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute(TRUNCATE_ALL)


@pytest.fixture
def documents_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the Nextcloud mirror root at a tmp dir; ``attachment_materialize`` reads it at call time."""
    import lib.attachment_materialize as attachment_materialize

    root = tmp_path / "Documents"
    monkeypatch.setattr(attachment_materialize, "DOCUMENTS", root, raising=True)
    return root


@pytest.fixture
def no_enqueue(monkeypatch: pytest.MonkeyPatch):
    """Replace extract_queue.enqueue with a recorder so tests never touch the
    real asyncio queue or run MinerU. Records the PDF name each call queued."""
    calls: list[str] = []
    monkeypatch.setattr(
        "lib.attachment_materialize.extract_queue.enqueue",
        lambda name, content, assets: calls.append(name),
    )
    return calls

