import asyncio
import json
import random
from uuid import uuid4

import psycopg
import pytest

from backend.identity import RequestIdentity
from backend.routes.sync import stream_changes
from backend.sync import CHANNEL, QUEUE_MAXSIZE, ChangeBroker
from lib.changes import UserScope, head_cursor, latest_changes
from lib.db_schema import SCHEMA_DIRECTORY
from tests.conftest import create_device, create_user

pytestmark = pytest.mark.usefixtures("clean_database")


@pytest.fixture
def listener(database_url):
    connection = psycopg.connect(database_url, autocommit=True)
    connection.execute(f"LISTEN {CHANNEL}")
    yield connection
    connection.close()


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
    user_id = create_user(postgres_pool, "alice@example.test")
    device_id = create_device(postgres_pool, user_id)

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


def test_replay_returns_rows_after_since_in_seq_order(postgres_pool):
    user_id = create_user(postgres_pool, "alice@example.test")
    seqs = [_change(postgres_pool, user_id, key) for key in ("c.md", "a.md", "b.md")]

    rows = latest_changes(postgres_pool, user_id, 0)

    assert [row["seq"] for row in rows] == sorted(seqs)
    assert [row["resource_key"] for row in rows] == ["c.md", "a.md", "b.md"]
    assert latest_changes(postgres_pool, user_id, seqs[0]) == rows[1:]
    assert latest_changes(postgres_pool, user_id, seqs[-1]) == []


def test_replay_keeps_only_the_latest_change_per_resource(postgres_pool):
    user_id = create_user(postgres_pool, "alice@example.test")
    first = _change(postgres_pool, user_id, "a.md")
    other = _change(postgres_pool, user_id, "b.md")
    latest = _change(postgres_pool, user_id, "a.md", version=2)

    rows = latest_changes(postgres_pool, user_id, 0)

    assert [(row["seq"], row["resource_key"]) for row in rows] == [(other, "b.md"), (latest, "a.md")]
    assert latest_changes(postgres_pool, user_id, first) == rows
    assert latest_changes(postgres_pool, user_id, other) == rows[1:]


def test_replay_excludes_other_users(postgres_pool):
    alice = create_user(postgres_pool, "alice@example.test")
    bob = create_user(postgres_pool, "bob@example.test")
    _change(postgres_pool, bob, "bob.md")
    alice_seq = _change(postgres_pool, alice, "alice.md")

    assert [(row["seq"], row["resource_key"]) for row in latest_changes(postgres_pool, alice, 0)] == [(alice_seq, "alice.md")]


def _rows(pool) -> list[tuple]:
    with pool.connection() as connection:
        return connection.execute("SELECT user_id, resource_kind, resource_key FROM changes ORDER BY seq").fetchall()


def test_pruning_superseded_rows_keeps_replay_identical_for_every_cursor(postgres_pool):
    users = [create_user(postgres_pool, "alice@example.test"), create_user(postgres_pool, "bob@example.test")]
    shuffle = random.Random(13)
    for version in range(1, 121):
        user, kind, key = shuffle.choice(users), shuffle.choice(("document", "asset")), shuffle.choice(("a", "b", "c", "d"))
        _change(postgres_pool, user, key, kind=kind, version=version)
    head = max(head_cursor(postgres_pool, user) for user in users)
    before = {(user, since): latest_changes(postgres_pool, user, since) for user in users for since in range(head + 1)}

    with postgres_pool.connection() as connection:
        connection.execute((SCHEMA_DIRECTORY / "0013_prune_superseded_changes.sql").read_text())

    remaining = _rows(postgres_pool)
    assert len(remaining) == len(set(remaining))
    assert before == {(user, since): latest_changes(postgres_pool, user, since) for user in users for since in range(head + 1)}


def test_record_prunes_only_superseded_rows_of_the_same_resource(postgres_pool):
    alice = create_user(postgres_pool, "alice@example.test")
    bob = create_user(postgres_pool, "bob@example.test")
    alice_scope, bob_scope = UserScope(postgres_pool, alice), UserScope(postgres_pool, bob)
    with postgres_pool.connection() as connection:
        bob_scope.record(connection, kind="document", key="a.md", operation="write", version=1)
        alice_scope.record(connection, kind="asset", key="a.md", operation="write", version=1)
        for version in (1, 2, 3):
            alice_scope.record(connection, kind="document", key="a.md", operation="write", version=version)
        alice_scope.record(connection, kind="document", key="b.md", operation="write", version=1)
        alice_scope.record(connection, kind="document", key="a.md", operation="delete")

    assert _rows(postgres_pool) == [
        (bob, "document", "a.md"), (alice, "asset", "a.md"), (alice, "document", "b.md"), (alice, "document", "a.md"),
    ]
    assert [(row["resource_key"], row["version"]) for row in latest_changes(postgres_pool, alice, 0)][-1] == ("a.md", None)


def test_record_skips_rows_another_transaction_holds(postgres_pool, database_url):
    user_id = create_user(postgres_pool, "alice@example.test")
    scope = UserScope(postgres_pool, user_id)
    held = _change(postgres_pool, user_id, "a.md")
    with psycopg.connect(database_url) as holder:
        holder.execute("SELECT seq FROM changes WHERE seq = %s FOR UPDATE", (held,))
        with postgres_pool.connection() as connection:
            connection.execute("SET LOCAL lock_timeout = '2s'")
            scope.record(connection, kind="document", key="a.md", operation="write", version=2)
        holder.rollback()

    assert len(_rows(postgres_pool)) == 2
    assert [row["version"] for row in latest_changes(postgres_pool, user_id, 0)] == [2]
    with postgres_pool.connection() as connection:
        scope.record(connection, kind="document", key="a.md", operation="write", version=3)
    assert len(_rows(postgres_pool)) == 1



async def test_broker_delivers_only_the_subscribed_user(postgres_pool, database_url):
    alice = create_user(postgres_pool, "alice@example.test")
    bob = create_user(postgres_pool, "bob@example.test")
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
    user_id = create_user(postgres_pool, "alice@example.test")
    missed = _change(postgres_pool, user_id, "old.md")
    identity = RequestIdentity(user_id=user_id, login="alice@example.test", device_id=create_device(postgres_pool, user_id))
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


async def test_stream_without_since_announces_the_head_then_only_later_changes(postgres_pool, database_url, monkeypatch):
    broker = ChangeBroker()
    monkeypatch.setattr("backend.routes.sync.get_postgres_pool", lambda: postgres_pool)
    monkeypatch.setattr("backend.routes.sync.get_change_broker", lambda: broker)
    user_id = create_user(postgres_pool, "alice@example.test")
    _change(postgres_pool, user_id, "old.md")
    head = _change(postgres_pool, user_id, "older.md")
    identity = RequestIdentity(user_id=user_id, login="alice@example.test", device_id=create_device(postgres_pool, user_id))
    await broker.start(database_url)
    try:
        response = await stream_changes(since=None, identity=identity)
        events = response.body_iterator
        cursor = await asyncio.wait_for(anext(events), timeout=5)
        await _await_listener(postgres_pool)
        live_seq = await asyncio.to_thread(_change, postgres_pool, user_id, "new.md")
        live = await asyncio.wait_for(anext(events), timeout=5)
        await events.aclose()
    finally:
        await broker.stop()

    assert cursor == {"event": "cursor", "data": json.dumps({"seq": head})}
    assert (json.loads(live["data"])["seq"], json.loads(live["data"])["resource_key"]) == (live_seq, "new.md")


async def test_broker_reconnects_after_its_connection_is_terminated(postgres_pool, database_url):
    user_id = create_user(postgres_pool, "alice@example.test")
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
