from uuid import uuid4

from fastapi import HTTPException
import pytest

from backend import identity


@pytest.fixture
def postgres_identity(monkeypatch, postgres_pool, clean_database):
    monkeypatch.setattr(identity, "get_postgres_pool", lambda: postgres_pool)


def test_host_access_needs_the_granted_capability_except_for_the_local_dev_login(monkeypatch):
    granted = f'{{"{identity.HOST_CAPABILITY}":[{{}}]}}'
    monkeypatch.setenv("GIGACHAD_ENV", "production")
    monkeypatch.setenv("GIGACHAD_DEV_USER", "me")

    assert identity.has_host_access("alice@example.test", granted)
    assert not identity.has_host_access("alice@example.test", '{"example.com/cap/other":[{}]}')
    assert not identity.has_host_access("me", None)

    monkeypatch.setenv("GIGACHAD_ENV", "development")
    assert identity.has_host_access("me", None)
    assert not identity.has_host_access("alice@example.test", None)


def test_postgres_identity_requires_tailscale_login(postgres_identity):
    with pytest.raises(HTTPException) as exc:
        identity.get_request_identity(tailscale_login=None, capabilities_header=None, device_id=str(uuid4()))

    assert exc.value.status_code == 401


def test_postgres_identity_requires_uuid_device_id(postgres_identity):
    with pytest.raises(HTTPException) as exc:
        identity.get_request_identity(tailscale_login="alice@example.test", capabilities_header=None, device_id="not-a-uuid")

    assert exc.value.status_code == 400


def test_postgres_identity_registers_one_user_and_device(postgres_identity):
    device_id = str(uuid4())

    first = identity.get_request_identity(tailscale_login="alice@example.test", capabilities_header=None, device_id=device_id)
    second = identity.get_request_identity(tailscale_login="alice@example.test", capabilities_header=None, device_id=device_id)

    assert first is not None
    assert second == first


def _device_row(pool, device_id: str) -> tuple:
    """The device's owner, last_seen_at and row version (xmin changes with every write)."""
    with pool.connection() as connection:
        return connection.execute("SELECT user_id, last_seen_at, xmin::text FROM devices WHERE id = %s", (device_id,)).fetchone()


def test_postgres_identity_reads_a_recently_seen_device_without_writing(postgres_identity, postgres_pool):
    device_id = str(uuid4())
    identity.get_request_identity(tailscale_login="alice@example.test", capabilities_header=None, device_id=device_id)
    before = _device_row(postgres_pool, device_id)

    identity.get_request_identity(tailscale_login="alice@example.test", capabilities_header=None, device_id=device_id)

    assert _device_row(postgres_pool, device_id) == before


def test_postgres_identity_refreshes_a_stale_device_and_moves_a_device_to_its_new_login(postgres_identity, postgres_pool):
    device_id = str(uuid4())
    alice = identity.get_request_identity(tailscale_login="alice@example.test", capabilities_header=None, device_id=device_id)
    with postgres_pool.connection() as connection:
        connection.execute("UPDATE devices SET last_seen_at = now() - interval '1 hour' WHERE id = %s", (device_id,))
    stale = _device_row(postgres_pool, device_id)[1]

    identity.get_request_identity(tailscale_login="alice@example.test", capabilities_header=None, device_id=device_id)
    assert _device_row(postgres_pool, device_id)[1] > stale

    bob = identity.get_request_identity(tailscale_login="bob@example.test", capabilities_header=None, device_id=device_id)
    assert bob.user_id != alice.user_id
    assert _device_row(postgres_pool, device_id)[0] == bob.user_id
