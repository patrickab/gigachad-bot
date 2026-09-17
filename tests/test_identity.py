import os
from uuid import uuid4

from fastapi import HTTPException
from psycopg_pool import ConnectionPool
import pytest

from backend import identity
from lib.db_schema import upgrade


@pytest.fixture(scope="module")
def postgres_pool():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for identity tests")
    upgrade(url)
    pool = ConnectionPool(url, min_size=1, max_size=4)
    yield pool
    pool.close()


@pytest.fixture(autouse=True)
def clean_database(postgres_pool):
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute("TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE")


@pytest.fixture
def postgres_identity(monkeypatch, postgres_pool):
    monkeypatch.setattr(identity, "get_postgres_pool", lambda: postgres_pool)


def test_postgres_identity_requires_tailscale_login(postgres_identity):
    with pytest.raises(HTTPException) as exc:
        identity.get_request_identity(tailscale_login=None, device_id=str(uuid4()))

    assert exc.value.status_code == 401


def test_postgres_identity_requires_uuid_device_id(postgres_identity):
    with pytest.raises(HTTPException) as exc:
        identity.get_request_identity(tailscale_login="alice@example.test", device_id="not-a-uuid")

    assert exc.value.status_code == 400


def test_postgres_identity_registers_one_user_and_device(postgres_identity):
    device_id = str(uuid4())

    first = identity.get_request_identity(tailscale_login="alice@example.test", device_id=device_id)
    second = identity.get_request_identity(tailscale_login="alice@example.test", device_id=device_id)

    assert first is not None
    assert second == first
