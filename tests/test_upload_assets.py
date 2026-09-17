import io
import os
from uuid import uuid4

from fastapi import HTTPException
import pytest
from psycopg_pool import ConnectionPool
from starlette.datastructures import Headers, UploadFile

from backend.routes import files
from backend.routes import assets as assets_route
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


async def test_upload_stores_asset_at_documented_logical_path(store):
    chat_id = "chat-1"

    result = await files.upload_file(
        file=upload(b"id,name\n1,a\n", "table.csv", "text/csv"),
        chat_id=chat_id,
        slug=None,
        overwrite=False,
        assets=store,
    )

    assert result.name == "table.csv"
    asset = store.read(f"chat_history/_uploads/{chat_id}/table.csv")
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

    assert store.read("chat_history/my-project/_uploads/chat-2/paper.pdf").content == b"%PDF-1.7"


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
    assert store.read("chat_history/_uploads/chat-3/notes.txt").content == b"first"
    assert store.read("chat_history/_uploads/chat-3/notes (1).txt").content == b"second"


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
    assert [asset.logical_path for asset in store.list("chat_history/_uploads/chat-4")] == [
        "chat_history/_uploads/chat-4/notes.txt"
    ]


async def test_delete_removes_the_asset_and_its_markdown_sidecar(store):
    store.write("upload", "chat_history/_uploads/chat-5/paper.pdf", b"%PDF-1.7")
    store.write("upload", "chat_history/_uploads/chat-5/paper.md", b"# parsed")
    store.write("upload", "chat_history/_uploads/chat-5/keep.txt", b"keep")

    assert await files.delete_single_file("chat-5", "paper.pdf", slug=None, assets=store) == {"status": "ok"}

    assert [asset.logical_path for asset in store.list("chat_history/_uploads/chat-5")] == [
        "chat_history/_uploads/chat-5/keep.txt"
    ]


async def test_delete_of_a_missing_attachment_is_404(store):
    with pytest.raises(HTTPException) as raised:
        await files.delete_single_file("chat-6", "ghost.txt", slug=None, assets=store)

    assert raised.value.status_code == 404


def test_delete_chat_upload_dir_clears_only_that_chats_prefix(store):
    store.write("upload", "chat_history/_uploads/chat-7/a.txt", b"a")
    store.write("upload", "chat_history/_uploads/chat-7/sub/b.txt", b"b")
    store.write("upload", "chat_history/_uploads/chat-70/c.txt", b"c")
    store.write("upload", "chat_history/proj/_uploads/chat-7/d.txt", b"d")

    files.delete_chat_upload_dir("chat-7", None, store)

    assert [asset.logical_path for asset in store.list()] == [
        "chat_history/proj/_uploads/chat-7/d.txt",
        "chat_history/_uploads/chat-70/c.txt",
    ]


def test_asset_route_serves_owner_bytes_and_hides_other_users(postgres_pool):
    alice = make_store(postgres_pool, "alice@example.test")
    bob = make_store(postgres_pool, "bob@example.test")
    written = alice.write("upload", "chat_history/_uploads/chat-8/photo.png", b"\x89PNG\r\n\x1a\nbody", mime="image/png")

    response = get_asset("chat_history/_uploads/chat-8/photo.png", assets=alice)

    assert response.body == b"\x89PNG\r\n\x1a\nbody"
    assert response.media_type == "image/png"
    assert response.headers["etag"] == f'"{written.version}"'

    for store, path in ((bob, "chat_history/_uploads/chat-8/photo.png"), (alice, "chat_history/_uploads/chat-8/gone.png")):
        with pytest.raises(HTTPException) as raised:
            get_asset(path, assets=store)
        assert raised.value.status_code == 404


def test_asset_route_rejects_traversal(store):
    with pytest.raises(HTTPException) as raised:
        get_asset("../../etc/passwd", assets=store)

    assert raised.value.status_code == 404


def test_asset_route_serves_local_mode_chat_files_and_nothing_else(tmp_path, monkeypatch):
    monkeypatch.setattr(assets_route, "DOCUMENTS", tmp_path)
    target = tmp_path / "chat_history" / "_uploads" / "chat-9"
    target.mkdir(parents=True)
    (target / "a.txt").write_bytes(b"local bytes")
    (tmp_path / "Prompts").mkdir()
    (tmp_path / "Prompts" / "secret.md").write_bytes(b"not an attachment")

    served = get_asset("chat_history/_uploads/chat-9/a.txt", assets=None)
    assert served.body == b"local bytes"

    # Only the chat tree is reachable: this route never replaces the document API.
    for path in ("Prompts/secret.md", "chat_history/_uploads/chat-9/missing.txt"):
        with pytest.raises(HTTPException) as raised:
            get_asset(path, assets=None)
        assert raised.value.status_code == 404


async def test_local_mode_still_writes_the_chat_upload_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(files, "chat_upload_dir", lambda chat_id, slug=None: tmp_path / (slug or "_uploads") / chat_id)

    first = await files.upload_file(
        file=upload(b"hello", "a.txt", "text/plain"), chat_id="chat-10", slug=None, overwrite=False, assets=None
    )
    second = await files.upload_file(
        file=upload(b"again", "a.txt", "text/plain"), chat_id="chat-10", slug=None, overwrite=False, assets=None
    )

    chat_dir = tmp_path / "_uploads" / "chat-10"
    assert (first.name, second.name) == ("a.txt", "a (1).txt")
    assert (chat_dir / "a.txt").read_bytes() == b"hello"
    assert (chat_dir / "a (1).txt").read_bytes() == b"again"
