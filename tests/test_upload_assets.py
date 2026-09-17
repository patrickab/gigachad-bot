import io
import os
from uuid import uuid4

from fastapi import HTTPException
from psycopg_pool import ConnectionPool
import pytest
from starlette.datastructures import Headers, UploadFile

from backend.routes import files
from backend.routes.assets import get_asset
from lib.asset_store import AssetStore
from lib.db_schema import upgrade


@pytest.fixture(scope="module")
def postgres_pool():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for upload asset tests")
    upgrade(url)
    pool = ConnectionPool(url, min_size=1, max_size=4)
    yield pool
    pool.close()


@pytest.fixture(autouse=True)
def clean_database(request):
    if "postgres_pool" not in request.fixturenames:
        return
    pool = request.getfixturevalue("postgres_pool")
    with pool.connection() as connection, connection.transaction():
        connection.execute("TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE")


def make_store(pool, login="alice@example.test"):
    device_id = uuid4()
    with pool.connection() as connection, connection.transaction():
        user_id = connection.execute("INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", (login,)).fetchone()[0]
        connection.execute("INSERT INTO devices (id, user_id) VALUES (%s, %s)", (device_id, user_id))
    return AssetStore(pool, user_id, device_id=device_id)


@pytest.fixture
def store(postgres_pool):
    return make_store(postgres_pool)


def upload(content: bytes, filename: str, mime: str) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename, headers=Headers({"content-type": mime}))


async def test_upload_stores_asset_at_database_native_path(store):
    chat_id = "chat-1"

    result = await files.upload_file(
        file=upload(b"id,name\n1,a\n", "table.csv", "text/csv"),
        chat_id=chat_id,
        slug=None,
        overwrite=False,
        assets=store,
    )

    assert result.name == "table.csv"
    asset = store.read(f"attachment/chat/{chat_id}/table.csv")
    assert (asset.kind, asset.mime, asset.content) == ("upload", "text/csv", b"id,name\n1,a\n")
    assert result.content == "id,name\n1,a\n"


async def test_project_upload_lands_under_the_project_slug(store):
    await files.upload_file(
        file=upload(b"%PDF-1.7", "paper.pdf", "application/pdf"),
        chat_id="chat-2",
        slug="my-project",
        overwrite=False,
        assets=store,
    )

    assert store.read("attachment/project/my-project/chat/chat-2/paper.pdf").content == b"%PDF-1.7"


async def test_duplicate_filename_is_deduped_not_overwritten(store):
    for body in (b"first", b"second"):
        result = await files.upload_file(
            file=upload(body, "notes.txt", "text/plain"),
            chat_id="chat-3",
            slug=None,
            overwrite=False,
            assets=store,
        )

    assert result.name == "notes (1).txt"
    assert store.read("attachment/chat/chat-3/notes.txt").content == b"first"
    assert store.read("attachment/chat/chat-3/notes (1).txt").content == b"second"


async def test_overwrite_reuses_the_same_logical_path(store):
    for body in (b"first", b"second"):
        result = await files.upload_file(
            file=upload(body, "notes.txt", "text/plain"),
            chat_id="chat-4",
            slug=None,
            overwrite=True,
            assets=store,
        )

    assert result.name == "notes.txt"
    assert [asset.logical_path for asset in store.list("attachment/chat/chat-4")] == [
        "attachment/chat/chat-4/notes.txt"
    ]


async def test_delete_removes_the_asset_and_its_markdown_sidecar(store):
    store.write("upload", "attachment/chat/chat-5/paper.pdf", b"%PDF-1.7")
    store.write("upload", "attachment/chat/chat-5/paper.md", b"# parsed")
    store.write("upload", "attachment/chat/chat-5/keep.txt", b"keep")

    assert await files.delete_single_file("chat-5", "paper.pdf", slug=None, assets=store) == {"status": "ok"}

    assert [asset.logical_path for asset in store.list("attachment/chat/chat-5")] == [
        "attachment/chat/chat-5/keep.txt"
    ]


async def test_delete_of_a_missing_attachment_is_404(store):
    with pytest.raises(HTTPException) as raised:
        await files.delete_single_file("chat-6", "ghost.txt", slug=None, assets=store)

    assert raised.value.status_code == 404


def test_delete_chat_upload_dir_clears_only_that_chats_prefix(store):
    store.write("upload", "attachment/chat/chat-7/a.txt", b"a")
    store.write("upload", "attachment/chat/chat-7/sub/b.txt", b"b")
    store.write("upload", "attachment/chat/chat-70/c.txt", b"c")
    store.write("upload", "attachment/project/proj/chat/chat-7/d.txt", b"d")

    files.delete_chat_upload_dir("chat-7", None, store)

    assert [asset.logical_path for asset in store.list()] == [
        "attachment/chat/chat-70/c.txt",
        "attachment/project/proj/chat/chat-7/d.txt",
    ]


def test_asset_route_serves_owner_bytes_and_hides_other_users(postgres_pool):
    alice = make_store(postgres_pool, "alice@example.test")
    bob = make_store(postgres_pool, "bob@example.test")
    written = alice.write("upload", "attachment/chat/chat-8/photo.png", b"\x89PNG\r\n\x1a\nbody", mime="image/png")

    response = get_asset("attachment/chat/chat-8/photo.png", assets=alice)

    assert response.body == b"\x89PNG\r\n\x1a\nbody"
    assert response.media_type == "image/png"
    assert response.headers["etag"] == f'"{written.version}"'

    for store, path in ((bob, "attachment/chat/chat-8/photo.png"), (alice, "attachment/chat/chat-8/gone.png")):
        with pytest.raises(HTTPException) as raised:
            get_asset(path, assets=store)
        assert raised.value.status_code == 404


def test_asset_route_rejects_traversal(store):
    with pytest.raises(HTTPException) as raised:
        get_asset("../../etc/passwd", assets=store)

    assert raised.value.status_code == 404

