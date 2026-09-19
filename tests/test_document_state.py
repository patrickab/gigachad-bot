"""Postgres-mode document routes: state lands in the database, not on disk.

The route functions are called directly with explicit request-scoped stores —
the same objects ``get_document_store``/``get_asset_store`` hand FastAPI — so a
test needs no HTTP client and no environment fiddling.
``documents.DOCUMENTS`` is redirected at a tmp_path so the only filesystem
writes a test can produce are deliberate PDF mirrors. Any other file below that
root is a policy violation.
"""

from io import BytesIO
import os
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from psycopg_pool import ConnectionPool
import pytest

from backend.routes import documents as route
from lib import attachment_materialize
from lib.asset_store import AssetStore
from lib.chat_store import ChatStore
from lib.data_store import StorageNotFoundError
from lib.db_schema import upgrade
from lib.postgres_data_store import PostgresDataStore
from lib.project_store import ProjectStore


@pytest.fixture(scope="module")
def postgres_pool():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for document route tests")
    upgrade(url)
    pool = ConnectionPool(url, min_size=1, max_size=4)
    yield pool
    pool.close()


@pytest.fixture(autouse=True)
def clean_database(postgres_pool):
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute("TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE")


@pytest.fixture(autouse=True)
def documents_root(tmp_path, monkeypatch):
    """Point the mirror root at a tmp dir; ``attachment_materialize.py`` reads it at call time."""
    root = tmp_path / "Documents"
    monkeypatch.setattr(attachment_materialize, "DOCUMENTS", root, raising=True)
    return root


def written_files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) if root.exists() else []


class Stores:
    """The three request-scoped stores a documents route is handed, for one user."""

    def __init__(self, pool, login):
        device_id = uuid4()
        with pool.connection() as connection, connection.transaction():
            self.user_id = connection.execute(
                "INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", (login,)
            ).fetchone()[0]
            connection.execute("INSERT INTO devices (id, user_id) VALUES (%s, %s)", (device_id, self.user_id))
        self.docs = PostgresDataStore(pool, self.user_id, device_id=device_id)
        self.assets = AssetStore(pool, self.user_id, device_id=device_id)
        self.projects = ProjectStore(chat_store=ChatStore(data_store=self.docs), data_store=self.docs)


@pytest.fixture
def alice(postgres_pool):
    return Stores(postgres_pool, "alice@example.test")


@pytest.fixture
def bob(postgres_pool):
    return Stores(postgres_pool, "bob@example.test")


def upload(name, content):
    return UploadFile(file=BytesIO(content), filename=name)


async def test_writing_a_project_document_registers_it_in_database(alice, documents_root):
    slug = alice.projects.create_project("Thesis")["slug"]

    meta = await route.write_document(
        route.WriteDocumentRequest(slug=slug, name="spec.md", content="# spec"), alice.projects, alice.docs, alice.assets
    )

    assert meta.path == f"project/{slug}/document/spec.md"
    assert alice.projects.list_files(slug) == [f"project/{slug}/document/spec.md"]
    assert alice.docs.read_bytes(f"project/{slug}/document/spec.md")[0] == b"# spec"
    assert written_files(documents_root) == []


async def test_writing_a_document_refreshes_the_chat_upload_copies(alice):
    slug = alice.projects.create_project("Thesis")["slug"]
    alice.assets.write("upload", f"attachment/project/{slug}/chat/chat-1/spec.md", b"stale")
    alice.assets.write("upload", f"attachment/project/{slug}/chat/chat-1/other.md", b"untouched")

    await route.write_document(
        route.WriteDocumentRequest(slug=slug, name="spec.md", content="fresh"), alice.projects, alice.docs, alice.assets
    )

    assert alice.assets.read(f"attachment/project/{slug}/chat/chat-1/spec.md").content == b"fresh"
    assert alice.assets.read(f"attachment/project/{slug}/chat/chat-1/other.md").content == b"untouched"


async def test_upload_promotes_into_the_pdf_library_and_mirrors_it(alice, documents_root, no_enqueue):
    slug = alice.projects.create_project("Thesis")["slug"]

    meta = await route.upload_document(upload("paper.pdf", b"%PDF-1.7 body"), slug, alice.projects, alice.assets)

    assert meta.path == "PDFs/paper.pdf"
    asset = alice.assets.read("PDFs/paper.pdf")
    assert (asset.kind, asset.content) == ("pdf", b"%PDF-1.7 body")
    assert alice.projects.list_files(slug) == ["PDFs/paper.pdf"]
    assert written_files(documents_root) == ["PDFs/paper.pdf"]
    # MinerU needs bytes to extract: extraction is queued straight from the upload.
    assert no_enqueue == ["paper.pdf"]


async def test_non_pdf_upload_cannot_enter_the_persisted_pdf_library(alice, documents_root):
    slug = alice.projects.create_project("Thesis")["slug"]

    with pytest.raises(HTTPException) as exc:
        await route.upload_document(upload("notes.txt", b"database-only"), slug, alice.projects, alice.assets)

    assert exc.value.status_code == 400
    assert alice.assets.list() == []
    assert written_files(documents_root) == []


async def test_non_pdf_chat_upload_cannot_enter_the_persisted_pdf_library(alice, documents_root):
    alice.assets.write("upload", "attachment/chat/chat-1/notes.txt", b"database-only")

    with pytest.raises(HTTPException) as exc:
        await route.register_upload(
            route.RegisterUploadRequest(chat_id="chat-1", filename="notes.txt"), None, alice.projects, alice.assets
        )

    assert exc.value.status_code == 400
    assert [asset.logical_path for asset in alice.assets.list()] == ["attachment/chat/chat-1/notes.txt"]
    assert written_files(documents_root) == []


async def test_register_upload_promotes_a_stored_chat_upload(alice, documents_root, no_enqueue):
    alice.assets.write("upload", "attachment/chat/chat-1/paper.pdf", b"%PDF-1.7 attached")

    meta = await route.register_upload(
        route.RegisterUploadRequest(chat_id="chat-1", filename="paper.pdf"), None, alice.projects, alice.assets
    )

    assert meta.path == "PDFs/paper.pdf"
    assert alice.assets.read("PDFs/paper.pdf").content == b"%PDF-1.7 attached"
    assert written_files(documents_root) == ["PDFs/paper.pdf"]
    assert no_enqueue == ["paper.pdf"]


async def test_register_upload_without_a_stored_upload_is_not_found(alice):
    with pytest.raises(HTTPException) as exc:
        await route.register_upload(
            route.RegisterUploadRequest(chat_id="chat-1", filename="ghost.pdf"), None, alice.projects, alice.assets
        )
    assert exc.value.status_code == 404


async def test_move_relocates_the_document_key_leaving_nothing_behind(alice):
    slug = alice.projects.create_project("Thesis")["slug"]
    await route.write_document(
        route.WriteDocumentRequest(slug="", name="sketch.canvas", content="{}"), alice.projects, alice.docs, alice.assets
    )

    meta = await route.move_document(
        route.MoveDocumentRequest(path="note/sketch.canvas", from_slug="", to_slug=slug),
        alice.projects,
        alice.docs,
    )

    assert meta.path == f"project/{slug}/document/sketch.canvas"
    assert alice.docs.read_bytes(f"project/{slug}/document/sketch.canvas")[0] == b"{}"
    assert not alice.docs.exists("note/sketch.canvas")
    assert alice.projects.list_files(slug) == [f"project/{slug}/document/sketch.canvas"]


async def test_move_onto_an_existing_name_is_a_conflict(alice):
    slug = alice.projects.create_project("Thesis")["slug"]
    for request in (
        route.WriteDocumentRequest(slug="", name="spec.md", content="note"),
        route.WriteDocumentRequest(slug=slug, name="spec.md", content="project"),
    ):
        await route.write_document(request, alice.projects, alice.docs, alice.assets)

    with pytest.raises(HTTPException) as exc:
        await route.move_document(
            route.MoveDocumentRequest(path="note/spec.md", from_slug="", to_slug=slug),
            alice.projects,
            alice.docs,
        )
    assert exc.value.status_code == 409
    assert alice.docs.read_bytes("note/spec.md")[0] == b"note"


async def test_delete_refuses_to_touch_a_document_outside_the_project_directory(alice):
    slug = alice.projects.create_project("Thesis")["slug"]
    await route.write_document(
        route.WriteDocumentRequest(slug="", name="idea.md", content="x"), alice.projects, alice.docs, alice.assets
    )

    await route.remove_document(slug, "note/idea.md", alice.projects, alice.docs)

    assert alice.docs.read_bytes("note/idea.md")[0] == b"x"


async def test_another_users_document_is_invisible_and_answers_404(alice, bob):
    slug = alice.projects.create_project("Thesis")["slug"]
    meta = await route.write_document(
        route.WriteDocumentRequest(slug=slug, name="secret.md", content="mine"), alice.projects, alice.docs, alice.assets
    )

    listed = await route.list_all_documents(bob.projects, bob.docs)
    assert listed.documents == []

    for call in (
        route.add_document(route.AddDocumentRequest(path=meta.path), slug, bob.projects, bob.docs, bob.assets),
        route.attach_document(meta.path, "chat-1", None, bob.projects, bob.docs, bob.assets),
    ):
        with pytest.raises(HTTPException) as exc:
            await call
        assert exc.value.status_code == 404

    with pytest.raises(StorageNotFoundError):
        bob.docs.read_bytes(meta.path)
    assert alice.docs.read_bytes(meta.path)[0] == b"mine"


async def test_attaching_a_registered_document_copies_it_into_the_chat_uploads(alice):
    slug = alice.projects.create_project("Thesis")["slug"]
    meta = await route.write_document(
        route.WriteDocumentRequest(slug=slug, name="spec.md", content="# spec"), alice.projects, alice.docs, alice.assets
    )

    result = await route.attach_document(meta.path, "chat-1", slug, alice.projects, alice.docs, alice.assets)

    assert (result.name, result.mime, result.content) == ("spec.md", "text/markdown", "# spec")
    assert alice.assets.read(f"attachment/project/{slug}/chat/chat-1/spec.md").content == b"# spec"
