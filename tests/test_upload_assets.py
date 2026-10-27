import io

from fastapi import HTTPException
import pytest
from starlette.datastructures import Headers, UploadFile

from backend.routes import files
from backend.routes.assets import get_asset
from lib.attachments import ChatUploads
from tests.conftest import make_asset_store

pytestmark = pytest.mark.usefixtures("clean_database", "documents_root")


def upload(content: bytes, filename: str, mime: str) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename, headers=Headers({"content-type": mime}))


async def test_upload_stores_asset_at_database_native_path(asset_store):
    chat_id = "chat-1"

    result = await files.upload_file(
        file=upload(b"id,name\n1,a\n", "table.csv", "text/csv"),
        chat_id=chat_id,
        slug=None,
        overwrite=False,
        assets=asset_store,
    )

    assert result.name == "table.csv"
    asset = asset_store.read(f"attachment/chat/{chat_id}/table.csv")
    assert (asset.kind, asset.mime, asset.content) == ("upload", "text/csv", b"id,name\n1,a\n")
    assert result.content == "id,name\n1,a\n"


async def test_project_upload_lands_under_the_project_slug(asset_store):
    await files.upload_file(
        file=upload(b"%PDF-1.7", "paper.pdf", "application/pdf"),
        chat_id="chat-2",
        slug="my-project",
        overwrite=False,
        assets=asset_store,
    )

    assert asset_store.read("attachment/project/my-project/chat/chat-2/paper.pdf").content == b"%PDF-1.7"


async def test_duplicate_filename_is_deduped_not_overwritten(asset_store):
    for body in (b"first", b"second"):
        result = await files.upload_file(
            file=upload(body, "notes.txt", "text/plain"),
            chat_id="chat-3",
            slug=None,
            overwrite=False,
            assets=asset_store,
        )

    assert result.name == "notes (1).txt"
    assert asset_store.read("attachment/chat/chat-3/notes.txt").content == b"first"
    assert asset_store.read("attachment/chat/chat-3/notes (1).txt").content == b"second"


async def test_overwrite_reuses_the_same_logical_path(asset_store):
    for body in (b"first", b"second"):
        result = await files.upload_file(
            file=upload(body, "notes.txt", "text/plain"),
            chat_id="chat-4",
            slug=None,
            overwrite=True,
            assets=asset_store,
        )

    assert result.name == "notes.txt"
    assert [asset.logical_path for asset in asset_store.list("attachment/chat/chat-4")] == [
        "attachment/chat/chat-4/notes.txt"
    ]


async def test_delete_removes_the_asset_and_its_markdown_sidecar(asset_store):
    asset_store.write("upload", "attachment/chat/chat-5/paper.pdf", b"%PDF-1.7")
    asset_store.write("upload", "attachment/chat/chat-5/paper.md", b"# parsed")
    asset_store.write("upload", "attachment/chat/chat-5/keep.txt", b"keep")

    assert await files.delete_single_file("chat-5", "paper.pdf", slug=None, assets=asset_store) == {"status": "ok"}

    assert [asset.logical_path for asset in asset_store.list("attachment/chat/chat-5")] == [
        "attachment/chat/chat-5/keep.txt"
    ]


async def test_delete_of_a_missing_attachment_is_404(asset_store):
    with pytest.raises(HTTPException) as raised:
        await files.delete_single_file("chat-6", "ghost.txt", slug=None, assets=asset_store)

    assert raised.value.status_code == 404


def test_delete_all_chat_uploads_clears_only_that_chats_prefix(asset_store):
    asset_store.write("upload", "attachment/chat/chat-7/a.txt", b"a")
    asset_store.write("upload", "attachment/chat/chat-7/sub/b.txt", b"b")
    asset_store.write("upload", "attachment/chat/chat-70/c.txt", b"c")
    asset_store.write("upload", "attachment/project/proj/chat/chat-7/d.txt", b"d")

    ChatUploads(asset_store, "chat-7").delete_all()

    assert [asset.logical_path for asset in asset_store.list()] == [
        "attachment/chat/chat-70/c.txt",
        "attachment/project/proj/chat/chat-7/d.txt",
    ]


def test_asset_route_serves_owner_bytes_and_hides_other_users(postgres_pool):
    alice, _ = make_asset_store(postgres_pool, "alice@example.test")
    bob, _ = make_asset_store(postgres_pool, "bob@example.test")
    written = alice.write("upload", "attachment/chat/chat-8/photo.png", b"\x89PNG\r\n\x1a\nbody", mime="image/png")

    response = get_asset("attachment/chat/chat-8/photo.png", assets=alice)

    assert response.body == b"\x89PNG\r\n\x1a\nbody"
    assert response.media_type == "image/png"
    assert response.headers["etag"] == f'"{written.version}"'

    for store, path in ((bob, "attachment/chat/chat-8/photo.png"), (alice, "attachment/chat/chat-8/gone.png")):
        with pytest.raises(HTTPException) as raised:
            get_asset(path, assets=store)
        assert raised.value.status_code == 404


def test_asset_route_rejects_traversal(asset_store):
    with pytest.raises(HTTPException) as raised:
        get_asset("../../etc/passwd", assets=asset_store)

    assert raised.value.status_code == 404


async def test_pdf_chat_upload_mirrors_to_nextcloud_immediately(asset_store, documents_root):
    """A PDF must reach Nextcloud on upload, whether or not it is ever sent/parsed
    afterward — deferring the mirror to a successful ``/files/parse`` left an
    uploaded-but-unsent (or unparseable) PDF without any Nextcloud copy."""
    await files.upload_file(
        file=upload(b"%PDF-1.7 body", "paper.pdf", "application/pdf"),
        chat_id="chat-9",
        slug=None,
        overwrite=False,
        assets=asset_store,
    )

    asset = asset_store.read("PDFs/paper.pdf")
    assert (asset.kind, asset.content) == ("pdf", b"%PDF-1.7 body")
    assert (documents_root / "PDFs" / "paper.pdf").read_bytes() == b"%PDF-1.7 body"
