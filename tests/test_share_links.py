"""Share links: owner management, guest access as the owner, rotation, revocation and lifecycle with the resource."""

import asyncio
import json
from uuid import uuid4

from fastapi import HTTPException
import pytest

from backend.routes.canvases import CanvasMutationBatchRequest
from backend.routes.share_links import (
    ShareTargetRequest,
    create_share,
    describe_shared,
    get_canvas_grant,
    get_grant,
    read_share,
    read_shared_canvas,
    read_shared_content,
    revoke_share,
    stream_shared_canvas,
    submit_shared_mutations,
)
from backend.sync import ChangeBroker
from lib.canvas_store import CanvasMutation, CanvasMutationStore
from lib.changes import UserScope
from lib.share_links import ShareLinks
from tests.conftest import create_user

pytestmark = pytest.mark.usefixtures("clean_database")

EMPTY = {"version": 1, "frames": [], "strokes": [], "attachments": [], "texts": []}


@pytest.fixture
def pool(postgres_pool, monkeypatch):
    monkeypatch.setattr("backend.routes.share_links.get_postgres_pool", lambda: postgres_pool)
    return postgres_pool


def _document(pool, user_id, key, content: bytes) -> None:
    with pool.connection() as connection:
        connection.execute("INSERT INTO documents (user_id, key, content) VALUES (%s, %s, %s)", (user_id, key, content))


def _links(pool, user_id) -> ShareLinks:
    return ShareLinks(UserScope(pool, user_id))


def _share(pool, user_id, kind, key) -> dict[str, str]:
    return create_share(ShareTargetRequest(kind=kind, key=key), _links(pool, user_id))


def _gone(secret: str) -> bool:
    with pytest.raises(HTTPException) as error:
        get_grant(secret)
    return error.value.status_code == 404


def _edit(text: str) -> CanvasMutationBatchRequest:
    mutation = CanvasMutation(uuid4(), "upsert", "text", str(uuid4()), {"x": 1, "y": 2, "width": 240, "height": 80, "text": text})
    return CanvasMutationBatchRequest.model_validate({"mutations": [mutation.event_data()]})


def test_canvas_link_edits_as_owner_until_rotated_or_revoked(pool):
    alice = create_user(pool, "alice@example.test")
    key = "project/notes/board.canvas"
    _document(pool, alice, key, json.dumps(EMPTY).encode())

    first = _share(pool, alice, "document", key)
    grant = get_canvas_grant(get_grant(first["secret"]))
    assert first["access"] == "edit"
    assert describe_shared(grant) == {"name": "board.canvas", "view": "canvas"}
    assert asyncio.run(submit_shared_mutations(_edit("from guest"), grant)) == {"revision": 1}
    assert asyncio.run(read_shared_canvas(grant))["document"]["texts"][0]["text"] == "from guest"
    assert CanvasMutationStore(UserScope(pool, alice)).snapshot(key)[1] == 1

    second = _share(pool, alice, "document", key)
    assert _gone(first["secret"])
    assert get_grant(second["secret"]) == grant
    assert read_share("document", key, _links(pool, alice)) == {"access": "edit"}

    revoke_share("document", key, _links(pool, alice))
    assert _gone(second["secret"])
    assert read_share("document", key, _links(pool, alice)) == {"access": None}


def test_document_and_asset_links_serve_sandboxed_bytes_view_only(pool):
    alice = create_user(pool, "alice@example.test")
    _document(pool, alice, "project/page.html", b"<script>alert(1)</script>")
    with pool.connection() as connection:
        connection.execute(
            "INSERT INTO assets (user_id, kind, logical_path, content, mime, sha256, size_bytes)"
            " VALUES (%s, 'upload', 'attachment/chat/c1/photo.png', %s, 'image/png', 'x', 3)",
            (alice, b"png"),
        )

    page = get_grant(_share(pool, alice, "document", "project/page.html")["secret"])
    response = read_shared_content(page)
    assert (response.body, response.media_type) == (b"<script>alert(1)</script>", "text/html")
    assert describe_shared(page) == {"name": "page.html", "view": "content"}
    assert response.headers["content-security-policy"] == "sandbox"
    assert response.headers["x-content-type-options"] == "nosniff"
    with pytest.raises(HTTPException) as error:
        get_canvas_grant(page)
    assert error.value.status_code == 404

    photo = get_grant(_share(pool, alice, "asset", "attachment/chat/c1/photo.png")["secret"])
    assert (read_shared_content(photo).body, describe_shared(photo)["name"]) == (b"png", "photo.png")


def test_link_follows_its_resource_and_never_reaches_another_users(pool):
    alice, bob = create_user(pool, "alice@example.test"), create_user(pool, "bob@example.test")
    _document(pool, alice, "notes/a.md", b"alice")
    secret = _share(pool, alice, "document", "notes/a.md")["secret"]

    with pool.connection() as connection:
        connection.execute("UPDATE documents SET key = 'notes/b.md' WHERE user_id = %s", (alice,))
    assert read_shared_content(get_grant(secret)).body == b"alice"

    for user, key in ((bob, "notes/b.md"), (alice, "notes/missing.md")):
        with pytest.raises(HTTPException) as error:
            _share(pool, user, "document", key)
        assert error.value.status_code == 404

    with pool.connection() as connection:
        connection.execute("DELETE FROM documents WHERE user_id = %s", (alice,))
    assert _gone(secret)


async def test_revoking_ends_a_live_canvas_stream(pool, database_url, monkeypatch):
    alice = create_user(pool, "alice@example.test")
    key = "notes/board.canvas"
    _document(pool, alice, key, json.dumps(EMPTY).encode())
    secret = _share(pool, alice, "document", key)["secret"]
    grant = get_canvas_grant(get_grant(secret))
    await submit_shared_mutations(_edit("before"), grant)

    broker = ChangeBroker()
    monkeypatch.setattr("backend.routes.canvases.get_change_broker", lambda: broker)
    await broker.start(database_url)
    try:
        events = (await stream_shared_canvas(secret, sinceRevision=0, grant=grant)).body_iterator
        assert json.loads((await asyncio.wait_for(anext(events), timeout=5))["data"])["revision"] == 1
        revoke_share("document", key, _links(pool, alice))
        await submit_shared_mutations(_edit("after"), grant)  # the canvas changes, which wakes the stream
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(events), timeout=5)
    finally:
        await broker.stop()
