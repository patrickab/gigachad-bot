"""Postgres-mode document routes: state lands in the database, not on disk.

The route functions are called directly with explicit request-scoped stores —
the same objects ``get_document_store``/``get_asset_store`` hand FastAPI — so a
test needs no HTTP client and no environment fiddling.
The ``documents_root`` fixture redirects the Nextcloud mirror at a tmp_path so the only filesystem
writes a test can produce are deliberate PDF mirrors. Any other file below that
root is a policy violation.
"""

from fastapi import HTTPException
import pytest

from backend.routes import documents as route
from lib import attachment_materialize
from lib.asset_store import AssetStore
from lib.changes import UserScope
from lib.chat_store import ChatStore
from lib.data_store import StorageNotFoundError
from lib.json_docs import JsonDocs
from lib.postgres_data_store import PostgresDataStore
from lib.project_rename import PostgresProjectRenamer
from lib.project_store import ProjectStore
from lib.references import References
from tests.conftest import create_user_and_device

pytestmark = pytest.mark.usefixtures("clean_database", "documents_root")


def written_files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) if root.exists() else []


class Stores:
    """The three request-scoped stores a documents route is handed, for one user."""

    def __init__(self, pool, login):
        self.user_id, device_id = create_user_and_device(pool, login)
        scope = UserScope(pool, self.user_id, device_id)
        self.docs = PostgresDataStore(scope)
        self.assets = AssetStore(scope)
        rename = PostgresProjectRenamer(scope).rename
        self.projects = ProjectStore(ChatStore(data_store=self.docs), rename=rename)


@pytest.fixture
def alice(postgres_pool):
    return Stores(postgres_pool, "alice@example.test")


@pytest.fixture
def bob(postgres_pool):
    return Stores(postgres_pool, "bob@example.test")


def test_writing_a_project_document_registers_it_in_database(alice, documents_root):
    slug = alice.projects.create_project("Thesis")["slug"]

    meta = route.write_document(
        route.WriteDocumentRequest(slug=slug, name="spec.md", content="# spec"), alice.projects, alice.docs, alice.assets
    )

    assert meta.path == f"project/{slug}/document/spec.md"
    assert alice.projects.list_files(slug) == [f"project/{slug}/document/spec.md"]
    assert alice.docs.read_bytes(f"project/{slug}/document/spec.md")[0] == b"# spec"
    assert written_files(documents_root) == []


def test_writing_a_document_refreshes_the_chat_upload_copies(alice):
    slug = alice.projects.create_project("Thesis")["slug"]
    alice.assets.write("upload", f"attachment/project/{slug}/chat/chat-1/spec.md", b"stale")
    alice.assets.write("upload", f"attachment/project/{slug}/chat/chat-1/other.md", b"untouched")

    route.write_document(
        route.WriteDocumentRequest(slug=slug, name="spec.md", content="fresh"), alice.projects, alice.docs, alice.assets
    )

    assert alice.assets.read(f"attachment/project/{slug}/chat/chat-1/spec.md").content == b"fresh"
    assert alice.assets.read(f"attachment/project/{slug}/chat/chat-1/other.md").content == b"untouched"


def test_move_relocates_the_document_key_leaving_nothing_behind(alice):
    slug = alice.projects.create_project("Thesis")["slug"]
    route.write_document(
        route.WriteDocumentRequest(slug="", name="sketch.canvas", content="{}"), alice.projects, alice.docs, alice.assets
    )

    meta = route.move_document(
        route.MoveDocumentRequest(path="note/sketch.canvas", from_slug="", to_slug=slug),
        alice.projects,
        alice.docs,
    )

    assert meta.path == f"project/{slug}/document/sketch.canvas"
    assert alice.docs.read_bytes(f"project/{slug}/document/sketch.canvas")[0] == b"{}"
    assert not alice.docs.exists("note/sketch.canvas")
    assert alice.projects.list_files(slug) == [f"project/{slug}/document/sketch.canvas"]



def test_rename_only_changes_the_name_a_project_document_is_shown_under(alice):
    slug = alice.projects.create_project("Thesis")["slug"]
    key = f"project/{slug}/document/mindmap.md"
    route.write_document(
        route.WriteDocumentRequest(slug=slug, name="mindmap.md", content="# Mind map"), alice.projects, alice.docs, alice.assets
    )

    meta = route.rename_document(route.RenameDocumentRequest(path=key, slug=slug, name="Research map"), alice.projects, alice.docs)

    assert (meta.path, meta.name) == (key, "Research map.md")
    assert alice.docs.read_bytes(key)[0] == b"# Mind map"
    assert not alice.docs.exists(f"project/{slug}/document/Research map.md")
    assert alice.projects.list_files(slug) == [key]
    listed = route.list_documents(slug, alice.projects, _NoArchitecture(), _NoRoots())
    assert [(doc.path, doc.name) for doc in listed.documents] == [(key, "Research map.md")]
    assert alice.projects.list_projects()[0]["file_names"] == {key: "Research map.md"}


def test_renaming_back_to_the_file_name_clears_the_display_name(alice):
    slug = alice.projects.create_project("Thesis")["slug"]
    key = f"project/{slug}/document/mindmap.md"
    route.write_document(route.WriteDocumentRequest(slug=slug, name="mindmap.md"), alice.projects, alice.docs, alice.assets)
    route.rename_document(route.RenameDocumentRequest(path=key, slug=slug, name="Research map"), alice.projects, alice.docs)

    meta = route.rename_document(route.RenameDocumentRequest(path=key, slug=slug, name="mindmap"), alice.projects, alice.docs)

    assert (meta.path, meta.name) == (key, "mindmap.md")
    assert JsonDocs(alice.docs).read(f"chat/{slug}/project.json")["files"] == [key]
    assert alice.projects.list_projects()[0]["file_names"] == {}


def test_renaming_a_library_pdf_names_it_in_one_project_and_keeps_the_pdf(alice, documents_root):
    first = alice.projects.create_project("First")["slug"]
    second = alice.projects.create_project("Second")["slug"]
    original = attachment_materialize.store_library_pdf(alice.assets, "source.pdf", b"%PDF-1.7 body")
    for slug in (first, second):
        route.add_document(route.AddDocumentRequest(path=original.logical_path), slug, alice.projects, alice.docs, alice.assets)
    mirrored = written_files(documents_root)

    renamed = route.rename_document(
        route.RenameDocumentRequest(path=original.logical_path, slug=first, name="renamed"), alice.projects, alice.docs,
    )

    assert (renamed.path, renamed.name) == (original.logical_path, "renamed.pdf")
    assert alice.assets.read(original.logical_path).content == b"%PDF-1.7 body"
    assert alice.projects.list_named_files(first) == [(original.logical_path, "renamed.pdf")]
    assert alice.projects.list_named_files(second) == [(original.logical_path, "source.pdf")]
    assert written_files(documents_root) == mirrored


def test_a_project_still_reads_plain_string_file_entries(alice):
    slug = alice.projects.create_project("Thesis")["slug"]
    project_key = f"chat/{slug}/project.json"
    project = JsonDocs(alice.docs)
    project.overwrite(project_key, {**project.read(project_key), "files": ["PDFs/b.pdf", {"path": "PDFs/a.pdf", "name": "Zeta.pdf"}]})

    assert alice.projects.list_named_files(slug) == [("PDFs/b.pdf", "b.pdf"), ("PDFs/a.pdf", "Zeta.pdf")]


def test_moving_a_renamed_canvas_keeps_the_name_it_is_shown_under(alice):
    first = alice.projects.create_project("First")["slug"]
    second = alice.projects.create_project("Second")["slug"]
    key = f"project/{first}/document/sketch.canvas"
    request = route.WriteDocumentRequest(slug=first, name="sketch.canvas", content="{}")
    route.write_document(request, alice.projects, alice.docs, alice.assets)
    route.rename_document(route.RenameDocumentRequest(path=key, slug=first, name="Plan"), alice.projects, alice.docs)

    moved = route.move_document(route.MoveDocumentRequest(path=key, from_slug=first, to_slug=second), alice.projects, alice.docs)
    assert alice.projects.list_named_files(second) == [(moved.path, "Plan.canvas")]

    note = route.move_document(route.MoveDocumentRequest(path=moved.path, from_slug=second, to_slug=""), alice.projects, alice.docs)
    assert note.path == "note/Plan.canvas"


def test_renaming_a_note_moves_it_to_the_new_name(alice):
    route.write_document(route.WriteDocumentRequest(slug="", name="idea.md", content="x"), alice.projects, alice.docs, alice.assets)

    meta = route.rename_document(route.RenameDocumentRequest(path="note/idea.md", slug="", name="plan"), alice.projects, alice.docs)

    assert (meta.path, meta.name) == ("note/plan.md", "plan.md")
    assert alice.docs.read_bytes("note/plan.md")[0] == b"x"
    assert not alice.docs.exists("note/idea.md")


class _NoArchitecture:
    def source_paths(self) -> list[str]:
        return []


class _NoRoots:
    def relocate(self, path: str) -> None:
        return None


def test_move_onto_an_existing_name_is_a_conflict(alice):
    slug = alice.projects.create_project("Thesis")["slug"]
    for request in (
        route.WriteDocumentRequest(slug="", name="spec.md", content="note"),
        route.WriteDocumentRequest(slug=slug, name="spec.md", content="project"),
    ):
        route.write_document(request, alice.projects, alice.docs, alice.assets)

    with pytest.raises(HTTPException) as exc:
        route.move_document(
            route.MoveDocumentRequest(path="note/spec.md", from_slug="", to_slug=slug),
            alice.projects,
            alice.docs,
        )
    assert exc.value.status_code == 409
    assert alice.docs.read_bytes("note/spec.md")[0] == b"note"


def test_delete_refuses_to_touch_a_document_outside_the_project_directory(alice):
    slug = alice.projects.create_project("Thesis")["slug"]
    route.write_document(
        route.WriteDocumentRequest(slug="", name="idea.md", content="x"), alice.projects, alice.docs, alice.assets
    )

    route.remove_document(slug, "note/idea.md", alice.projects, References(alice.docs, alice.assets))

    assert alice.docs.read_bytes("note/idea.md")[0] == b"x"


def test_another_users_document_is_invisible_and_answers_404(alice, bob):
    slug = alice.projects.create_project("Thesis")["slug"]
    meta = route.write_document(
        route.WriteDocumentRequest(slug=slug, name="secret.md", content="mine"), alice.projects, alice.docs, alice.assets
    )

    listed = route.list_all_documents(bob.projects)
    assert listed.documents == []

    with pytest.raises(HTTPException) as exc:
        route.add_document(route.AddDocumentRequest(path=meta.path), slug, bob.projects, bob.docs, bob.assets)
    assert exc.value.status_code == 404

    with pytest.raises(StorageNotFoundError):
        bob.docs.read_bytes(meta.path)
    assert alice.docs.read_bytes(meta.path)[0] == b"mine"

