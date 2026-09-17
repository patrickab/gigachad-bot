import asyncio
import json
import os
from uuid import UUID, uuid4

import psycopg
from psycopg_pool import ConnectionPool
import pytest

from backend.identity import RequestIdentity
from backend.routes.sync import replay_changes, stream_changes
from backend.sync import CHANNEL, QUEUE_MAXSIZE, ChangeBroker
from lib.db_schema import upgrade


@pytest.fixture(scope="module")
def database_url():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for sync tests")
    upgrade(url)
    return url


@pytest.fixture(scope="module")
def postgres_pool(database_url):
    pool = ConnectionPool(database_url, min_size=1, max_size=4)
    yield pool
    pool.close()


@pytest.fixture(autouse=True)
def clean_database(postgres_pool):
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute("TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE")


@pytest.fixture
def listener(database_url):
    connection = psycopg.connect(database_url, autocommit=True)
    connection.execute(f"LISTEN {CHANNEL}")
    yield connection
    connection.close()


def _user(pool, login) -> UUID:
    with pool.connection() as connection:
        return connection.execute("INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", (login,)).fetchone()[0]


def _device(pool, user_id) -> UUID:
    device_id = uuid4()
    with pool.connection() as connection:
        connection.execute("INSERT INTO devices (id, user_id) VALUES (%s, %s)", (device_id, user_id))
    return device_id


def _change(pool, user_id, resource_key, *, kind="document", version=1, device_id=None) -> int:
    with pool.connection() as connection:
        return connection.execute(
            """
            INSERT INTO changes (user_id, resource_kind, resource_key, version, operation, device_id)
            VALUES (%s, %s, %s, %s, 'write', %s)
            RETURNING seq
            """,
            (user_id, kind, resource_key, version, device_id),
        ).fetchone()[0]


async def _collect(events) -> list[dict]:
    return [event async for event in events]


def _listener_pids(pool) -> list[int]:
    with pool.connection() as connection:
        return [
            row[0]
            for row in connection.execute(
                "SELECT pid FROM pg_stat_activity WHERE query = %s AND pid <> pg_backend_pid()",
                (f"LISTEN {CHANNEL}",),
            ).fetchall()
        ]


def _terminate_listeners(pool) -> None:
    with pool.connection() as connection:
        connection.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE query = %s AND pid <> pg_backend_pid()",
            (f"LISTEN {CHANNEL}",),
        )


async def _await_listener(pool) -> None:
    """Block until the broker's LISTEN is actually issued.

    A fixed sleep is a bet: a NOTIFY sent before the LISTEN lands is lost, because the
    channel is at-most-once.
    """
    for _ in range(80):
        if await asyncio.to_thread(_listener_pids, pool):
            return
        await asyncio.sleep(0.25)
    pytest.fail("listener connection never issued its LISTEN")


def test_insert_notifies_once_with_metadata_only(postgres_pool, listener):
    user_id = _user(postgres_pool, "alice@example.test")
    device_id = _device(postgres_pool, user_id)

    seq = _change(postgres_pool, user_id, "notes/a.md", version=3, device_id=device_id)

    received = list(listener.notifies(timeout=5, stop_after=1))
    assert [notify.channel for notify in received] == [CHANNEL]
    assert json.loads(received[0].payload) == {
        "seq": seq,
        "user_id": str(user_id),
        "resource_kind": "document",
        "resource_key": "notes/a.md",
        "version": 3,
        "device_id": str(device_id),
    }
    assert list(listener.notifies(timeout=0.5)) == []


def test_replay_returns_rows_after_since_in_seq_order(postgres_pool, monkeypatch):
    monkeypatch.setattr("backend.routes.sync.get_postgres_pool", lambda: postgres_pool)
    user_id = _user(postgres_pool, "alice@example.test")
    seqs = [_change(postgres_pool, user_id, key) for key in ("c.md", "a.md", "b.md")]

    rows = replay_changes(user_id, 0)

    assert [row["seq"] for row in rows] == sorted(seqs)
    assert [row["resource_key"] for row in rows] == ["c.md", "a.md", "b.md"]
    assert replay_changes(user_id, seqs[0]) == rows[1:]
    assert replay_changes(user_id, seqs[-1]) == []


def test_replay_excludes_other_users(postgres_pool, monkeypatch):
    monkeypatch.setattr("backend.routes.sync.get_postgres_pool", lambda: postgres_pool)
    alice = _user(postgres_pool, "alice@example.test")
    bob = _user(postgres_pool, "bob@example.test")
    _change(postgres_pool, bob, "bob.md")
    alice_seq = _change(postgres_pool, alice, "alice.md")

    assert [(row["seq"], row["resource_key"]) for row in replay_changes(alice, 0)] == [(alice_seq, "alice.md")]


async def test_broker_delivers_only_the_subscribed_user(postgres_pool, database_url):
    alice = _user(postgres_pool, "alice@example.test")
    bob = _user(postgres_pool, "bob@example.test")
    broker = ChangeBroker()
    await broker.start(database_url)
    try:
        async with broker.subscribe(alice) as events:
            await _await_listener(postgres_pool)
            await asyncio.to_thread(_change, postgres_pool, bob, "bob.md")
            seq = await asyncio.to_thread(_change, postgres_pool, alice, "alice.md")
            event = await asyncio.wait_for(anext(events), timeout=5)
    finally:
        await broker.stop()

    assert (event["seq"], event["resource_key"], event["user_id"]) == (seq, "alice.md", str(alice))


async def test_broker_closes_subscriber_that_stops_reading():
    user_id = uuid4()
    broker = ChangeBroker()

    async with broker.subscribe(user_id) as events:
        for seq in range(QUEUE_MAXSIZE + 5):
            broker.dispatch(json.dumps({"seq": seq, "user_id": str(user_id), "resource_key": "a.md"}))
        delivered = await asyncio.wait_for(_collect(events), timeout=5)

    assert len(delivered) <= QUEUE_MAXSIZE


async def test_stream_replays_missed_changes_then_live_ones(postgres_pool, database_url, monkeypatch):
    broker = ChangeBroker()
    monkeypatch.setattr("backend.routes.sync.get_postgres_pool", lambda: postgres_pool)
    monkeypatch.setattr("backend.routes.sync.get_change_broker", lambda: broker)
    user_id = _user(postgres_pool, "alice@example.test")
    missed = _change(postgres_pool, user_id, "old.md")
    identity = RequestIdentity(user_id=user_id, login="alice@example.test", device_id=_device(postgres_pool, user_id))
    await broker.start(database_url)
    try:
        response = await stream_changes(since=0, identity=identity)
        events = response.body_iterator
        replayed = await asyncio.wait_for(anext(events), timeout=5)
        await _await_listener(postgres_pool)
        live_seq = await asyncio.to_thread(_change, postgres_pool, user_id, "new.md", version=2)
        live = await asyncio.wait_for(anext(events), timeout=5)
        await events.aclose()
    finally:
        await broker.stop()

    assert replayed["event"] == "change"
    assert json.loads(replayed["data"]) == {
        "seq": missed,
        "resource_kind": "document",
        "resource_key": "old.md",
        "version": 1,
        "device_id": None,
    }
    assert json.loads(live["data"]) == {
        "seq": live_seq,
        "resource_kind": "document",
        "resource_key": "new.md",
        "version": 2,
        "device_id": None,
    }


async def test_broker_reconnects_after_its_connection_is_terminated(postgres_pool, database_url):
    user_id = _user(postgres_pool, "alice@example.test")
    broker = ChangeBroker()
    await broker.start(database_url)
    try:
        async with broker.subscribe(user_id) as events:
            await _await_listener(postgres_pool)
            before = set(await asyncio.to_thread(_listener_pids, postgres_pool))
            await asyncio.to_thread(_terminate_listeners, postgres_pool)
            # A terminated backend lingers in pg_stat_activity, so "a listener exists"
            # can match the dying row and let us write before the broker has reconnected
            # — a NOTIFY sent then is simply lost. Wait for a pid we have not seen yet.
            for _ in range(120):  # bounded backoff delays the reconnect by seconds
                await asyncio.sleep(0.25)
                if set(await asyncio.to_thread(_listener_pids, postgres_pool)) - before:
                    break
            else:
                pytest.fail("listener did not reconnect after its connection dropped")

            # NOTIFY is at-most-once, so a write that races the re-LISTEN is lost by
            # design and the client recovers with ?since=. Keep writing until one lands
            # rather than betting the test on which side of that race it is.
            pending = asyncio.ensure_future(anext(events))
            written: list[int] = []
            for _ in range(20):
                written.append(await asyncio.to_thread(_change, postgres_pool, user_id, "after-drop.md"))
                done, _ = await asyncio.wait({pending}, timeout=1)
                if done:
                    event = pending.result()
                    break
            else:
                pending.cancel()
                pytest.fail("no change arrived after the listener reconnected")
    finally:
        await broker.stop()

    assert event["resource_key"] == "after-drop.md"
    assert event["seq"] in written
