"""Focused PostgreSQL coverage for canvas mutation sequencing and delivery."""

import asyncio
import json
import os
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from psycopg_pool import ConnectionPool

from backend.canvas import CanvasMutation, CanvasMutationStore, InvalidCanvasSnapshot
from backend.identity import RequestIdentity
from backend.routes.canvases import CanvasMutationBatchRequest, stream_canvas, submit_mutations
from backend.sync import ChangeBroker
from lib.db_schema import upgrade


@pytest.fixture(scope="module")
def database_url() -> str:
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for canvas mutation tests")
    upgrade(url)
    return url


@pytest.fixture(scope="module")
def postgres_pool(database_url: str):
    pool = ConnectionPool(database_url, min_size=1, max_size=4)
    yield pool
    pool.close()


@pytest.fixture(autouse=True)
def clean_database(postgres_pool):
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute("TRUNCATE users CASCADE")


def _user(pool: ConnectionPool, login: str) -> UUID:
    with pool.connection() as connection:
        return connection.execute("INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", (login,)).fetchone()[0]


def _canvas(pool: ConnectionPool, user_id: UUID, key: str, content: bytes | dict) -> None:
    with pool.connection() as connection:
        connection.execute(
            "INSERT INTO documents (user_id, key, content) VALUES (%s, %s, %s)",
            (user_id, key, content if isinstance(content, bytes) else json.dumps(content).encode()),
        )


def _stored(pool: ConnectionPool, user_id: UUID, key: str) -> bytes:
    with pool.connection() as connection:
        return bytes(connection.execute(
            "SELECT content FROM documents WHERE user_id = %s AND key = %s", (user_id, key)
        ).fetchone()[0])


def _text(text: str) -> CanvasMutation:
    return CanvasMutation(uuid4(), "upsert", "text", str(uuid4()), {"x": 1, "y": 2, "width": 240, "height": 80, "text": text})


EMPTY = {"version": 1, "frames": [], "strokes": [], "attachments": [], "texts": []}


def test_blank_canvas_reads_empty_and_accepts_its_first_mutation(postgres_pool):
    user_id = _user(postgres_pool, "blank@example.test")
    key = "notes/blank.canvas"
    _canvas(postgres_pool, user_id, key, b"")
    store = CanvasMutationStore(postgres_pool, user_id)
    mutation = _text("first")

    assert store.snapshot(key) == (EMPTY, 0)
    assert store.apply(key, [mutation]) == 1
    assert store.snapshot(key) == ({**EMPTY, "texts": [{**mutation.value, "id": mutation.entity_id}]}, 1)


def test_malformed_canvas_fails_closed_on_read_and_write(postgres_pool):
    user_id = _user(postgres_pool, "malformed@example.test")
    key = "notes/malformed.canvas"
    _canvas(postgres_pool, user_id, key, b"{")
    store = CanvasMutationStore(postgres_pool, user_id)

    with pytest.raises(InvalidCanvasSnapshot):
        store.snapshot(key)
    with pytest.raises(InvalidCanvasSnapshot):
        store.apply(key, [_text("never written")])
    assert _stored(postgres_pool, user_id, key) == b"{"


def test_retried_mutations_are_acknowledged_at_their_original_revision(postgres_pool):
    user_id = _user(postgres_pool, "alice@example.test")
    key = "notes/board.canvas"
    _canvas(postgres_pool, user_id, key, EMPTY)
    store = CanvasMutationStore(postgres_pool, user_id)
    first, second = _text("first"), _text("second")

    assert store.apply(key, [first]) == 1
    assert store.apply(key, [first, second]) == 2  # partial retry: only `second` is new
    assert store.apply(key, [first]) == 1
    assert store.apply(key, [first, second]) == 2

    assert [(batch.revision, [m.mutation_id for m in batch.mutations]) for batch in store.replay(key, 0)] == [
        (1, [first.mutation_id]),
        (2, [second.mutation_id]),
    ]
    assert [text["text"] for text in store.snapshot(key)[0]["texts"]] == ["first", "second"]


def test_legacy_canvas_reads_with_the_ids_its_first_write_persists(postgres_pool):
    user_id = _user(postgres_pool, "legacy@example.test")
    key = "notes/legacy.canvas"
    legacy_stroke = {"points": [[2, 3], [5, 8]], "color": "#123", "width": 4}
    _canvas(postgres_pool, user_id, key, {
        "version": 1,
        "viewport": {"scale": 1, "centerX": 4, "centerY": 8},
        "pages": [{"id": "page-1", "x": 0, "y": 0}],
        "imageEmbeds": [{"id": "image-1", "path": "images/a.png", "x": 10, "y": 20, "width": 320}],
        "pdfEmbeds": [{"id": "pdf-1", "path": "files/a.pdf", "x": 30, "y": 40, "width": 500}],
        "strokes": [legacy_stroke, legacy_stroke],
        "texts": [{"id": "text-1", "x": 1, "y": 2, "text": "legacy"}],
    })
    store = CanvasMutationStore(postgres_pool, user_id)

    read, _ = store.snapshot(key)
    stroke_ids = [stroke["id"] for stroke in read["strokes"]]
    assert len(set(stroke_ids)) == 2
    # Erasing a legacy stroke by the ID the client read must persist.
    store.apply(key, [CanvasMutation(uuid4(), "delete", "stroke", stroke_ids[0])])

    assert json.loads(_stored(postgres_pool, user_id, key)) == {
        "version": 1,
        "viewport": {"scale": 1, "centerX": 4, "centerY": 8},
        "frames": [
            {"id": "page-1", "kind": "page", "x": 0, "y": 0, "width": 794},
            {"id": "image-1", "kind": "image", "path": "images/a.png", "x": 10, "y": 20, "width": 320},
        ],
        "attachments": [{"id": "pdf-1", "kind": "pdf", "path": "files/a.pdf", "x": 30, "y": 40, "width": 500}],
        "strokes": [{**legacy_stroke, "id": stroke_ids[1]}],
        "texts": [{"id": "text-1", "x": 1, "y": 2, "text": "legacy", "width": 240, "height": 80}],
    }


def test_canvas_access_is_user_scoped(postgres_pool, monkeypatch):
    alice = _user(postgres_pool, "alice@example.test")
    bob = _user(postgres_pool, "bob@example.test")
    key = "notes/private.canvas"
    _canvas(postgres_pool, alice, key, EMPTY)
    monkeypatch.setattr("backend.routes.canvases.get_postgres_pool", lambda: postgres_pool)
    identity = RequestIdentity(user_id=bob, login="bob@example.test", device_id=None)

    for attempt in (
        submit_mutations(key, CanvasMutationBatchRequest(mutations=[]), identity),
        stream_canvas(key, sinceRevision=0, identity=identity),
    ):
        with pytest.raises(HTTPException) as error:
            asyncio.run(attempt)
        assert error.value.status_code == 404
    assert _stored(postgres_pool, alice, key) == json.dumps(EMPTY).encode()


async def test_canvas_stream_replays_then_delivers_live_batches(postgres_pool, database_url, monkeypatch):
    user_id = _user(postgres_pool, "alice@example.test")
    key = "notes/board.canvas"
    _canvas(postgres_pool, user_id, key, EMPTY)
    store = CanvasMutationStore(postgres_pool, user_id)
    replayed = _text("replayed")
    store.apply(key, [replayed])

    broker = ChangeBroker()
    monkeypatch.setattr("backend.routes.canvases.get_postgres_pool", lambda: postgres_pool)
    monkeypatch.setattr("backend.routes.canvases.get_change_broker", lambda: broker)
    identity = RequestIdentity(user_id=user_id, login="alice@example.test", device_id=None)
    await broker.start(database_url)
    try:
        events = (await stream_canvas(key, sinceRevision=0, identity=identity)).body_iterator
        first = json.loads((await asyncio.wait_for(anext(events), timeout=5))["data"])
        for _ in range(40):
            with postgres_pool.connection() as connection:
                if connection.execute(
                    "SELECT 1 FROM pg_stat_activity WHERE datname = current_database() AND query = 'LISTEN gigachad_changes'"
                ).fetchone():
                    break
            await asyncio.sleep(0.25)
        # Park the stream on its live wait so only the change notification can wake it.
        pending = asyncio.ensure_future(anext(events))
        await asyncio.sleep(0.5)
        live = _text("live")
        await asyncio.to_thread(store.apply, key, [live])
        second = json.loads((await asyncio.wait_for(pending, timeout=10))["data"])
        await events.aclose()
    finally:
        await broker.stop()

    assert (first["revision"], first["mutations"][0]["mutationId"]) == (1, str(replayed.mutation_id))
    assert (second["revision"], second["mutations"][0]["mutationId"]) == (2, str(live.mutation_id))
