"""Shared pytest fixtures."""

from collections.abc import Callable, Mapping
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from psycopg_pool import ConnectionPool
import pytest

from backend.identity import RequestIdentity
from lib.asset_store import AssetStore
from lib.changes import UserScope
from lib.db_schema import upgrade
from lib.file_roots import FileRoots
from lib.postgres_data_store import PostgresDataStore

TRUNCATE_ALL = "TRUNCATE changes, assets, vault_roots, devices, documents, users, model_reasoning_levels CASCADE"


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


def make_asset_store(pool: ConnectionPool, login: str = "alice@example.test") -> tuple[AssetStore, UUID]:
    """Register *login* with one device; return that user's asset store and the device id."""
    user_id, device_id = create_user_and_device(pool, login)
    return AssetStore(UserScope(pool, user_id, device_id)), device_id


def mounted_roots(*directories: Path | str, host_access: bool = False) -> FileRoots:
    """File roots whose saved mounts are *directories*; saving more mounts is discarded."""
    repository = SimpleNamespace(load=lambda: [str(directory) for directory in directories], save=lambda roots: None)
    return FileRoots(repository=repository, host_access=host_access)


def request_identity(has_host_access: bool = True) -> RequestIdentity:
    """A fresh identity for me@example.test without a device."""
    return RequestIdentity(uuid4(), "me@example.test", None, has_host_access)


def _returning(value: Any) -> Callable[[], Any]:
    return lambda: value


def route_client(*routers: APIRouter, overrides: Mapping[Callable[..., Any], Any] | None = None) -> TestClient:
    """A client for *routers* mounted on a bare app, each dependency in *overrides* answering its fixed value."""
    app = FastAPI()
    for router in routers:
        app.include_router(router)
    for dependency, value in (overrides or {}).items():
        app.dependency_overrides[dependency] = _returning(value)
    return TestClient(app)


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
def data_store(postgres_pool: ConnectionPool, clean_database: None) -> PostgresDataStore:
    """A new user's document store in an emptied database."""
    return PostgresDataStore(UserScope(postgres_pool, create_user(postgres_pool)))


@pytest.fixture
def asset_store(postgres_pool: ConnectionPool, clean_database: None) -> AssetStore:
    """alice's asset store in an emptied database."""
    return make_asset_store(postgres_pool)[0]


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
        lambda name, content, store: calls.append(name),
    )
    return calls

