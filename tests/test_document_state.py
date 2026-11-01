"""Postgres-mode document routes: state lands in the database, not on disk.

The route functions are called directly with explicit request-scoped stores —
the same objects ``get_document_store``/``get_asset_store`` hand FastAPI — so a
test needs no HTTP client and no environment fiddling.
The ``documents_root`` fixture redirects the Nextcloud mirror at a tmp_path so the only filesystem
writes a test can produce are deliberate PDF mirrors. Any other file below that
root is a policy violation.
"""

import json
from types import SimpleNamespace

from fastapi import HTTPException
import pytest

from backend.routes import documents as route
import config
from lib import attachment_materialize
from lib.asset_store import AssetStore
from lib.changes import UserScope, latest_changes
from lib.chat_store import ChatStore
from lib.data_store import StorageConflictError, StorageNotFoundError
from lib.disk_files import DiskFiles
from lib.json_docs import JsonDocs
from lib.postgres_data_store import PostgresDataStore
from lib.project_rename import PostgresKeyRelocator, PostgresProjectRenamer
from lib.project_store import ProjectStore
from lib.references import References
from lib.storage_namespace import WORKBENCH_SESSIONS
from tests.conftest import create_user_and_device, mounted_roots

pytestmark = pytest.mark.usefixtures("clean_database", "documents_root")


def written_files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) if root.exists() else []


class Stores:
    """The three request-scoped stores a documents route is handed, for one user."""

    def __init__(self, pool, login):
        self.pool = pool
        self.user_id, device_id = create_user_and_device(pool, login)
        scope = self.scope = UserScope(pool, self.user_id, device_id)
        self.docs = PostgresDataStore(scope)
        self.assets = AssetStore(scope)
        rename = PostgresProjectRenamer(scope).rename
        self.projects = ProjectStore(ChatStore(data_store=self.docs), rename=rename)
        self.relocate = PostgresKeyRelocator(scope).relocate


@pytest.fixture
def alice(postgres_pool):
    return Stores(postgres_pool, "alice@example.test")


@pytest.fixture
def bob(postgres_pool):
    return Stores(postgres_pool, "bob@example.test")


def rename(alice, path, slug, name, roots=None):
    """`POST /api/documents/rename` against *roots* (nothing mounted by default)."""
    request = route.RenameDocumentRequest(path=path, slug=slug, name=name)
    roots = roots or mounted_roots()
    return route.rename_document(
        request, alice.projects, alice.docs, alice.assets, roots, alice.relocate, References(alice.docs, alice.assets),
        DiskFiles(roots, alice.scope),
    )


def status_of(call) -> int:
    with pytest.raises(HTTPException) as exc:
        call()
    return exc.value.status_code


def write_project_document(alice, slug, name, content="x"):
    route.write_document(route.WriteDocumentRequest(slug=slug, name=name, content=content), alice.projects, alice.docs, alice.assets)
    return f"project/{slug}/document/{name}"


def mounted_file(tmp_path, name="notes.md", content="hello"):
    """A file in a fresh mounted folder, returning the folder and the file."""
    folder = tmp_path.resolve() / "mount"
    folder.mkdir(exist_ok=True)
    (folder / name).write_text(content, encoding="utf-8")
    return folder, folder / name


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
        alice.relocate,
    )

    assert meta.path == f"project/{slug}/document/sketch.canvas"
    assert alice.docs.read_bytes(f"project/{slug}/document/sketch.canvas")[0] == b"{}"
    assert not alice.docs.exists("note/sketch.canvas")
    assert alice.projects.list_files(slug) == [f"project/{slug}/document/sketch.canvas"]


def test_renaming_a_project_document_moves_its_key_and_every_reference(alice):
    first = alice.projects.create_project("First")["slug"]
    second = alice.projects.create_project("Second")["slug"]
    key = write_project_document(alice, first, "mindmap.md", "# Mind map")
    board = write_project_document(alice, first, "board.canvas", json.dumps({"attachments": [{"path": key}]}))
    alice.projects.add_file(second, key)
    alice.projects.set_file_name(second, key, "Legacy.md")
    alice.docs.write_bytes("note/links.md", f"[map]({key})".encode())

    meta = rename(alice, key, first, "Research map")

    renamed = f"project/{first}/document/Research map.md"
    assert (meta.path, meta.name) == (renamed, "Research map.md")
    assert alice.docs.read_bytes(renamed)[0] == b"# Mind map"
    assert not alice.docs.exists(key)
    assert alice.projects.list_named_files(first) == [(board, "board.canvas"), (renamed, "Research map.md")]
    assert alice.projects.list_named_files(second) == [(renamed, "Research map.md")]
    assert json.loads(alice.docs.read_bytes(board)[0]) == {"attachments": [{"path": renamed}]}
    assert alice.docs.read_bytes("note/links.md")[0] == f"[map]({renamed})".encode()
    assert alice.projects.list_projects()[0]["file_names"] == {}
    moved_from = {row["resource_key"]: row["moved_to"] for row in latest_changes(alice.pool, alice.user_id, 0)}
    assert moved_from[key] == renamed


def test_renaming_a_project_document_to_its_own_name_changes_nothing(alice):
    slug = alice.projects.create_project("Thesis")["slug"]
    key = write_project_document(alice, slug, "mindmap.md", "# Mind map")

    meta = rename(alice, key, slug, "mindmap")

    assert (meta.path, meta.name) == (key, "mindmap.md")
    assert alice.docs.read_bytes(key)[0] == b"# Mind map"
    assert alice.projects.list_files(slug) == [key]


def test_renaming_onto_an_existing_name_is_a_conflict(alice):
    slug = alice.projects.create_project("Thesis")["slug"]
    first = write_project_document(alice, slug, "a.md", "first")
    second = write_project_document(alice, slug, "b.md", "second")

    assert status_of(lambda: rename(alice, first, slug, "b")) == 409

    assert alice.docs.read_bytes(first)[0] == b"first" and alice.docs.read_bytes(second)[0] == b"second"
    assert alice.projects.list_files(slug) == [first, second]


@pytest.mark.parametrize("name", ["", "   ", "a/b", "..", "a\\b"])
def test_renaming_to_an_unusable_name_is_refused(alice, name):
    slug = alice.projects.create_project("Thesis")["slug"]
    key = write_project_document(alice, slug, "a.md")

    assert status_of(lambda: rename(alice, key, slug, name)) == 400

    assert alice.projects.list_files(slug) == [key]


def test_a_library_pdf_cannot_be_renamed(alice):
    slug = alice.projects.create_project("Thesis")["slug"]
    original = attachment_materialize.store_library_pdf(alice.assets, "source.pdf", b"%PDF-1.7 body")
    route.add_document(route.AddDocumentRequest(path=original.logical_path), slug, alice.projects, alice.docs, alice.assets)

    assert status_of(lambda: rename(alice, original.logical_path, slug, "renamed")) == 400

    assert alice.assets.read(original.logical_path).content == b"%PDF-1.7 body"
    assert alice.projects.list_named_files(slug) == [(original.logical_path, "source.pdf")]


def test_renaming_a_stored_pdf_gives_its_library_copy_the_new_name(alice):
    slug = alice.projects.create_project("Thesis")["slug"]
    key = f"project/{slug}/document/paper.pdf"
    alice.docs.write_bytes(key, b"%PDF-1.7 body")
    alice.projects.add_file(slug, key)
    attachment_materialize.store_library_pdf(alice.assets, "paper.pdf", b"%PDF-1.7 body")

    meta = rename(alice, key, slug, "thesis")

    assert meta.path == f"project/{slug}/document/thesis.pdf"
    assert alice.assets.read("PDFs/thesis.pdf").content == b"%PDF-1.7 body"
    assert alice.assets.stat("PDFs/paper.pdf") is None


def test_renaming_a_mounted_file_renames_it_on_disk_and_repoints_every_reference(alice, tmp_path):
    folder, source = mounted_file(tmp_path)
    first = alice.projects.create_project("First")["slug"]
    second = alice.projects.create_project("Second")["slug"]
    for slug in (first, second):
        alice.projects.add_file(slug, str(source))
    alice.projects.set_file_name(second, str(source), "Legacy.md")
    stored_links = {str(source): {"active": "s1", "sessions": ["s1"]}, "/elsewhere/notes.md": "s2"}
    alice.docs.write_bytes(WORKBENCH_SESSIONS, json.dumps(stored_links).encode())
    alice.docs.write_bytes("note/board.canvas", json.dumps({"attachments": [{"path": str(source)}]}).encode())
    alice.docs.write_bytes("note/links.md", f"[notes]({source})".encode())
    target = folder / "ideas.md"

    meta = rename(alice, str(source), first, "ideas", mounted_roots(folder))

    assert (meta.path, meta.name) == (str(target), "ideas.md")
    assert target.read_text(encoding="utf-8") == "hello" and not source.exists()
    assert alice.projects.list_named_files(first) == [(str(target), "ideas.md")]
    assert alice.projects.list_named_files(second) == [(str(target), "ideas.md")]
    assert json.loads(alice.docs.read_bytes(WORKBENCH_SESSIONS)[0]) == {
        str(target): {"active": "s1", "sessions": ["s1"], "moved": ["s1"]}, "/elsewhere/notes.md": "s2",
    }
    assert json.loads(alice.docs.read_bytes("note/board.canvas")[0]) == {"attachments": [{"path": str(target)}]}
    assert alice.docs.read_bytes("note/links.md")[0] == f"[notes]({target})".encode()
    moved_from = {row["resource_key"]: row["moved_to"] for row in latest_changes(alice.pool, alice.user_id, 0)}
    assert moved_from[str(source)] == str(target) and moved_from[str(target)] is None


def test_a_file_outside_any_project_renames_with_an_empty_scope(alice, tmp_path):
    folder, source = mounted_file(tmp_path)

    meta = rename(alice, str(source), "", "ideas", mounted_roots(folder))

    assert meta.path == str(folder / "ideas.md") and (folder / "ideas.md").is_file() and not source.exists()


def test_renaming_a_mounted_file_onto_an_existing_one_is_a_conflict(alice, tmp_path):
    folder, source = mounted_file(tmp_path)
    (folder / "ideas.md").write_text("taken", encoding="utf-8")
    slug = alice.projects.create_project("Thesis")["slug"]
    alice.projects.add_file(slug, str(source))

    assert status_of(lambda: rename(alice, str(source), slug, "ideas", mounted_roots(folder))) == 409

    assert source.read_text(encoding="utf-8") == "hello" and (folder / "ideas.md").read_text(encoding="utf-8") == "taken"
    assert alice.projects.list_files(slug) == [str(source)]


def test_a_project_never_shows_two_files_under_one_name(alice, tmp_path):
    folder, source = mounted_file(tmp_path)
    elsewhere = tmp_path.resolve() / "chapter"
    elsewhere.mkdir()
    (elsewhere / "ideas.md").write_text("taken", encoding="utf-8")
    slug = alice.projects.create_project("Thesis")["slug"]
    alice.projects.add_file(slug, str(source))
    alice.projects.add_file(slug, str(elsewhere / "ideas.md"))
    stored = write_project_document(alice, slug, "draft.md")

    assert status_of(lambda: rename(alice, str(source), slug, "IDEAS", mounted_roots(tmp_path.resolve()))) == 409
    assert status_of(lambda: rename(alice, stored, slug, "ideas")) == 409

    assert source.is_file() and alice.docs.is_file(stored)


def test_a_file_outside_the_roots_is_not_renamed(alice, tmp_path):
    _, source = mounted_file(tmp_path)
    other = tmp_path.resolve() / "other"
    other.mkdir()

    assert status_of(lambda: rename(alice, str(source), "", "ideas", mounted_roots(other))) == 403

    assert source.is_file() and not (source.parent / "ideas.md").exists()


def test_a_project_can_only_rename_files_it_references(alice, tmp_path):
    folder, source = mounted_file(tmp_path)
    slug = alice.projects.create_project("Thesis")["slug"]

    assert status_of(lambda: rename(alice, str(source), slug, "ideas", mounted_roots(folder))) == 404

    assert source.is_file()


def test_a_missing_mounted_file_answers_404(alice, tmp_path):
    folder, source = mounted_file(tmp_path)
    source.unlink()

    assert status_of(lambda: rename(alice, str(source), "", "ideas", mounted_roots(folder))) == 404


def test_a_failed_reference_rewrite_moves_the_file_back(alice, tmp_path):
    folder, source = mounted_file(tmp_path)
    slug = alice.projects.create_project("Thesis")["slug"]
    alice.projects.add_file(slug, str(source))

    def failing(old, new):
        raise StorageConflictError("the database refused")

    alice.relocate = failing
    assert status_of(lambda: rename(alice, str(source), slug, "ideas", mounted_roots(folder))) == 409

    assert source.read_text(encoding="utf-8") == "hello" and not (folder / "ideas.md").exists()
    assert alice.projects.list_files(slug) == [str(source)]


def test_renaming_a_tex_file_discards_the_build_outputs_of_both_names(alice, tmp_path):
    folder, source = mounted_file(tmp_path, "paper.tex", "\\documentclass{article}")
    outputs = folder / ".latex-build"
    outputs.mkdir()
    for name in ("paper.pdf", "paper.view.pdf", "thesis.view.pdf", "other.pdf"):
        (outputs / name).write_bytes(b"%PDF")

    rename(alice, str(source), "", "thesis", mounted_roots(folder))

    assert sorted(path.name for path in outputs.iterdir()) == ["other.pdf"]

def test_a_project_still_reads_plain_string_file_entries(alice):
    slug = alice.projects.create_project("Thesis")["slug"]
    project_key = f"chat/{slug}/project.json"
    project = JsonDocs(alice.docs)
    project.overwrite(project_key, {**project.read(project_key), "files": ["PDFs/b.pdf", {"path": "PDFs/a.pdf", "name": "Zeta.pdf"}]})

    assert alice.projects.list_named_files(slug) == [("PDFs/b.pdf", "b.pdf"), ("PDFs/a.pdf", "Zeta.pdf")]


def test_moving_a_renamed_canvas_keeps_its_name(alice):
    first = alice.projects.create_project("First")["slug"]
    second = alice.projects.create_project("Second")["slug"]
    key = write_project_document(alice, first, "sketch.canvas", "{}")
    renamed = rename(alice, key, first, "Plan").path

    moved = route.move_document(
        route.MoveDocumentRequest(path=renamed, from_slug=first, to_slug=second), alice.projects, alice.docs, alice.relocate
    )
    assert alice.projects.list_named_files(second) == [(moved.path, "Plan.canvas")]

    note = route.move_document(
        route.MoveDocumentRequest(path=moved.path, from_slug=second, to_slug=""), alice.projects, alice.docs, alice.relocate
    )
    assert note.path == "note/Plan.canvas"


def test_renaming_a_note_moves_it_and_the_links_to_it(alice):
    route.write_document(route.WriteDocumentRequest(slug="", name="idea.md", content="x"), alice.projects, alice.docs, alice.assets)
    alice.docs.write_bytes("note/links.md", b"[idea](note/idea.md)")

    meta = rename(alice, "note/idea.md", "", "plan")

    assert (meta.path, meta.name) == ("note/plan.md", "plan.md")
    assert alice.docs.read_bytes("note/plan.md")[0] == b"x"
    assert not alice.docs.exists("note/idea.md")
    assert alice.docs.read_bytes("note/links.md")[0] == b"[idea](note/plan.md)"


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
            alice.relocate,
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


def folder_entry(tmp_path):
    """A mounted LaTeX project folder with a root, a bibliography and build output, returning the mount and the folder."""
    mount = tmp_path.resolve() / "mount"
    folder = mount / "paper"
    (folder / ".latex-build").mkdir(parents=True)
    for name in ("main.tex", "refs.bib", "main.aux", ".latex-build/main.pdf"):
        (folder / name).write_text(name, encoding="utf-8")
    return mount, folder


def list_project(alice, slug, roots):
    return route.list_documents(slug, alice.projects, SimpleNamespace(source_paths=lambda: []), roots).documents


def test_a_folder_entry_lists_its_tree_and_its_files_rename_like_mounted_ones(alice, tmp_path):
    mount, folder = folder_entry(tmp_path)
    slug = alice.projects.create_project("Thesis")["slug"]
    alice.projects.add_file(slug, str(folder))
    alice.docs.write_bytes(WORKBENCH_SESSIONS, json.dumps({str(folder / "refs.bib"): "s1"}).encode())

    [entry] = list_project(alice, slug, mounted_roots(mount))
    renamed = rename(alice, str(folder / "refs.bib"), "", "literature.bib", mounted_roots(mount))

    assert entry.path == str(folder)
    assert [child["name"] for child in entry.children] == ["main.tex", "refs.bib"]
    assert renamed.path == str(folder / "literature.bib") and (folder / "literature.bib").is_file()
    assert alice.projects.list_named_files(slug) == [(str(folder), "paper")]
    assert json.loads(alice.docs.read_bytes(WORKBENCH_SESSIONS)[0]) == {
        str(folder / "literature.bib"): {"active": "s1", "sessions": ["s1"], "moved": ["s1"]},
    }


def test_deleting_a_folder_entry_unlists_it_and_leaves_the_folder_on_disk(alice, tmp_path):
    mount, folder = folder_entry(tmp_path)
    slug = alice.projects.create_project("Thesis")["slug"]
    alice.projects.add_file(slug, str(folder))

    route.remove_document(slug, str(folder), alice.projects, References(alice.docs, alice.assets))

    assert list_project(alice, slug, mounted_roots(mount)) == []
    assert sorted(path.name for path in folder.iterdir()) == [".latex-build", "main.aux", "main.tex", "refs.bib"]


@pytest.fixture
def workbench(tmp_path, monkeypatch):
    root = tmp_path.resolve() / "workbench"
    monkeypatch.setattr(config, "WORKBENCH", root)
    return root


def workbench_document(alice, workbench, slug, folder, name):
    """A listed workbench file `<workbench>/latex/<slug>/<folder>/<name>` with `imgs/` and build output beside it."""
    source = workbench / "latex" / slug / folder / name
    (source.parent / "imgs").mkdir(parents=True)
    (source.parent / ".latex-build").mkdir()
    for path in (source, source.parent / "imgs" / "a.png", source.parent / ".latex-build" / "out.pdf"):
        path.write_text("x", encoding="utf-8")
    alice.projects.add_file(slug, str(source))
    return source


def remove(alice, slug, path):
    route.remove_document(slug, str(path), alice.projects, References(alice.docs, alice.assets))


def test_deleting_a_renamed_workbench_tex_removes_its_folder_and_chat_link(alice, workbench):
    slug = alice.projects.create_project("Thesis")["slug"]
    renamed = workbench_document(alice, workbench, slug, "test2", "test.tex")
    links = {str(renamed): "s1", str(renamed.parent / "other.tex"): "s2", "/elsewhere/a.md": "s3"}
    alice.docs.write_bytes(WORKBENCH_SESSIONS, json.dumps(links).encode())

    remove(alice, slug, renamed)

    assert not renamed.parent.exists() and not renamed.parent.parent.exists() and (workbench / "latex").is_dir()
    assert alice.projects.list_files(slug) == []
    assert json.loads(alice.docs.read_bytes(WORKBENCH_SESSIONS)[0]) == {"/elsewhere/a.md": "s3"}


def test_deleting_a_promoted_folder_entry_removes_the_workbench_folder(alice, workbench):
    slug = alice.projects.create_project("Thesis")["slug"]
    source = workbench_document(alice, workbench, slug, "paper", "paper.tex")
    alice.projects.replace_file(slug, str(source), str(source.parent))

    remove(alice, slug, source.parent)

    assert not source.parent.exists()


def test_the_slug_folder_stays_while_another_document_is_in_it(alice, workbench):
    slug = alice.projects.create_project("Thesis")["slug"]
    first = workbench_document(alice, workbench, slug, "one", "one.tex")
    second = workbench_document(alice, workbench, slug, "two", "two.tex")

    remove(alice, slug, first)
    assert not first.parent.exists() and second.is_file()

    remove(alice, slug, second)
    assert not second.parent.parent.exists()


def test_a_workbench_folder_another_project_references_stays(alice, workbench):
    slug = alice.projects.create_project("Thesis")["slug"]
    other = alice.projects.create_project("Other")["slug"]
    source = workbench_document(alice, workbench, slug, "paper", "paper.tex")
    alice.projects.add_file(other, str(source))

    remove(alice, slug, source)

    assert source.is_file() and alice.projects.list_files(other) == [str(source)]


def test_deleting_a_mounted_file_only_unlinks_it(alice, workbench, tmp_path):
    slug = alice.projects.create_project("Thesis")["slug"]
    mounted = tmp_path / "mount" / "paper" / "paper.tex"
    mounted.parent.mkdir(parents=True)
    mounted.write_text("x", encoding="utf-8")
    alice.projects.add_file(slug, str(mounted))

    remove(alice, slug, mounted)

    assert mounted.is_file() and alice.projects.list_files(slug) == []
