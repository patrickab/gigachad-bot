"""Deletes release derived data (library PDFs, annotation canvases, project files) once unused, and never MinerU output."""

import hashlib
import json
from uuid import uuid4

import pytest

from lib.asset_store import AssetStore
from lib.canvas_store import CanvasMutation, CanvasMutationStore
from lib.chat_cleanup import ChatCleanup
from lib.chat_store import ChatStore
from lib.document_service import DocumentScope
from lib.postgres_data_store import PostgresDataStore
from lib.project_store import ProjectStore
from lib.references import References, Released, annotation_canvas
from lib.storage_namespace import pdf_annotations

PDF = b"%PDF-1.7 paper"
EMPTY_CANVAS = {"version": 1, "frames": [], "strokes": [], "attachments": [], "texts": []}


class SandboxStub:
    """Records the chats whose sandbox state was deleted."""

    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete_chat_state(self, *, chat_id: str) -> None:
        self.deleted.append(chat_id)


class World:
    """One user's stores, wired the way request dependencies wire them."""

    def __init__(self, docs: PostgresDataStore) -> None:
        self.docs = docs
        self.assets = AssetStore(docs._scope)
        self.chats = ChatStore(data_store=docs)
        self.projects = ProjectStore(self.chats, rename=lambda *_: None)
        self.references = References(docs, self.assets)
        self.sandbox = SandboxStub()
        self.cleanup = ChatCleanup(self.references, self.sandbox)

    def library_pdf(self, name: str, content: bytes = PDF) -> str:
        """A library PDF with its MinerU markdown, one image and an annotation canvas, returning the annotation key."""
        stem = name.removesuffix(".pdf")
        self.assets.write("pdf", f"PDFs/{name}", content)
        self.assets.write("mineru_markdown", f"Mineru/{stem}.md", b"# parsed")
        self.assets.write("mineru_image", f"Mineru/images/{stem}/{stem}_0.png", b"png")
        return self.annotation(content)

    def annotation(self, content: bytes) -> str:
        key = pdf_annotations(hashlib.sha256(content).hexdigest())
        self.docs.write_bytes(key, b"")
        return key

    def chat(self, filename: str, chat_id: str, attachments: list[dict]) -> None:
        message = {"role": "user", "content": "q", "attachments": attachments}
        self.chats.save(filename, {"messages": [message], "chat_id": chat_id})

    def canvas(self, key: str, attachments: list[dict] | None = None, frames: list[dict] | None = None) -> str:
        self.docs.write_bytes(key, json.dumps({**EMPTY_CANVAS, "attachments": attachments or [], "frames": frames or []}).encode())
        return key

    def has(self, key: str) -> bool:
        return bool(self.assets.list(key)) or self.docs.is_file(key)

    def library_left(self, name: str) -> bool:
        return self.has(f"PDFs/{name}")

    def mineru_kept(self, name: str) -> bool:
        stem = name.removesuffix(".pdf")
        return self.has(f"Mineru/{stem}.md") and self.has(f"Mineru/images/{stem}/{stem}_0.png")


def pdf_attachment(name: str) -> dict:
    return {"name": name, "mime": "application/pdf", "url": "", "active": True}


@pytest.fixture
def world(data_store: PostgresDataStore) -> World:
    return World(data_store)


async def test_chat_delete_removes_its_uploads_and_unused_library_data(world: World) -> None:
    annotation = world.library_pdf("paper.pdf")
    world.assets.write("upload", "attachment/chat/c1/paper.pdf", PDF, mime="application/pdf")
    world.assets.write("upload", "attachment/chat/c1/paper.md", b"# parsed", mime="text/markdown")
    world.chat("one.json", "c1", [pdf_attachment("paper.pdf")])

    await world.cleanup.after(lambda released: world.chats.delete("one.json", released=released))

    assert world.assets.list("attachment") == []
    assert not world.library_left("paper.pdf") and world.mineru_kept("paper.pdf")
    assert not world.has(annotation)
    assert world.sandbox.deleted == ["c1"]


async def test_chat_delete_keeps_library_data_another_chat_references(world: World) -> None:
    annotation = world.library_pdf("paper.pdf")
    world.assets.write("upload", "attachment/chat/c1/paper.pdf", PDF, mime="application/pdf")
    world.chat("one.json", "c1", [pdf_attachment("paper.pdf")])
    world.chat("two.json", "c2", [{**pdf_attachment("paper.pdf"), "vaultPath": "PDFs/paper.pdf"}])

    await world.cleanup.after(lambda released: world.chats.delete("one.json", released=released))

    assert world.assets.list("attachment") == []
    assert world.library_left("paper.pdf")
    assert world.has(annotation)


async def test_cascade_delete_releases_every_branch(world: World) -> None:
    world.library_pdf("paper.pdf")
    world.chat("root.json", "root", [])
    child = world.chats.branch("root.json", 0)
    world.assets.write("upload", f"attachment/chat/{child['chat_id']}/paper.pdf", PDF, mime="application/pdf")
    world.chat(child["child_file"], child["chat_id"], [pdf_attachment("paper.pdf")])

    await world.cleanup.after(lambda released: world.chats.cascade_delete("root.json", released=released))

    assert world.assets.list("attachment") == []
    assert not world.library_left("paper.pdf") and world.mineru_kept("paper.pdf")
    assert sorted(world.sandbox.deleted) == sorted(["root", child["chat_id"]])


async def test_a_failed_delete_releases_nothing(world: World) -> None:
    world.library_pdf("paper.pdf")
    world.assets.write("upload", "attachment/chat/c1/paper.pdf", PDF, mime="application/pdf")
    world.chat("one.json", "c1", [pdf_attachment("paper.pdf")])

    def fail(released: Released) -> None:
        released.chat("c1", None, ["PDFs/paper.pdf"])
        raise RuntimeError("store failed")

    with pytest.raises(RuntimeError):
        await world.cleanup.after(fail)

    assert world.has("attachment/chat/c1/paper.pdf")
    assert world.library_left("paper.pdf")
    assert world.sandbox.deleted == []


async def test_project_delete_removes_its_documents_canvases_and_what_only_they_used(world: World) -> None:
    slug = world.projects.create_project("Thesis")["slug"]
    other = world.projects.create_project("Other")["slug"]
    world.library_pdf("paper.pdf")
    project_pdf = f"project/{slug}/document/scan.pdf"
    world.docs.write_bytes(project_pdf, b"%PDF scan")
    scan_annotation = world.annotation(b"%PDF scan")
    pasted = f"project/{slug}/document/pasted-1.png"
    world.docs.write_bytes(pasted, b"png")
    board = world.canvas(f"project/{slug}/document/board.canvas", [{"id": "a", "kind": "pdf", "path": "PDFs/paper.pdf"}],
                         [{"id": "f", "kind": "image", "path": pasted}])
    shared = f"project/{slug}/document/shared.md"
    world.docs.write_bytes(shared, b"shared")
    for key in (project_pdf, board, shared):
        world.projects.add_file(slug, key)
    world.projects.add_file(other, shared)
    world.assets.write("upload", f"attachment/project/{slug}/chat/c1/a.txt", b"a")
    world.chat(f"{slug}/tab.json", "c1", [])

    await world.cleanup.after(lambda released: world.projects.delete_project(slug, released=released))

    assert not any(world.has(key) for key in (project_pdf, scan_annotation, pasted, board))
    assert not world.library_left("paper.pdf") and world.mineru_kept("paper.pdf")
    assert world.has(shared)
    assert world.assets.list("attachment") == []
    assert world.sandbox.deleted == ["c1"]


async def test_project_delete_keeps_a_library_pdf_a_notes_canvas_shows(world: World) -> None:
    slug = world.projects.create_project("Thesis")["slug"]
    annotation = world.library_pdf("paper.pdf")
    world.projects.add_file(slug, "PDFs/paper.pdf")
    world.canvas("note/board.canvas", [{"id": "a", "kind": "pdf", "path": "PDFs/paper.pdf"}])

    await world.cleanup.after(lambda released: world.projects.delete_project(slug, released=released))

    assert world.library_left("paper.pdf")
    assert world.has(annotation)


def test_removing_a_canvas_drops_its_mutation_log_and_the_documents_only_it_showed(world: World) -> None:
    slug = world.projects.create_project("Thesis")["slug"]
    pasted = f"project/{slug}/document/pasted-1.png"
    world.docs.write_bytes(pasted, b"png")
    board = world.canvas(f"project/{slug}/document/board.canvas")
    world.projects.add_file(slug, board)
    frame = {"kind": "image", "x": 0, "y": 0, "width": 10, "path": pasted}
    CanvasMutationStore(world.docs._scope).apply(board, [CanvasMutation(uuid4(), "upsert", "frame", "f", frame)])
    with world.docs._scope.connection() as connection:
        connection.execute("INSERT INTO canvas_shares (secret_hash, user_id, canvas_key) VALUES (%s, %s, %s)",
                           (b"secret", world.docs._scope.user_id, board))

    scope = DocumentScope(slug)
    scope.unlink(world.projects, board)
    world.references.release(scope.removal(board))

    assert not world.has(board) and not world.has(pasted)
    with world.docs._scope.connection() as connection:
        assert connection.execute("SELECT count(*) FROM canvas_mutations").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM canvas_shares").fetchone()[0] == 0


def test_removing_a_note_deletes_it_with_its_unused_derived_data(world: World) -> None:
    world.library_pdf("paper.pdf")
    note = world.canvas("note/board.canvas", [{"id": "a", "kind": "pdf", "path": "PDFs/paper.pdf"}])

    world.references.release(DocumentScope("").removal(note))

    assert not world.has(note)
    assert not world.library_left("paper.pdf") and world.mineru_kept("paper.pdf")


async def test_a_referenced_host_pdf_keeps_its_annotations_and_is_never_touched(world: World, tmp_path) -> None:
    host = tmp_path / "paper.pdf"
    host.write_bytes(PDF)
    slug = world.projects.create_project("Thesis")["slug"]
    world.projects.add_file(slug, str(host))
    annotation = world.library_pdf("paper.pdf")
    world.chat("one.json", "c1", [{**pdf_attachment("paper.pdf"), "vaultPath": str(host)}])

    await world.cleanup.after(lambda released: world.chats.delete("one.json", released=released))

    assert world.has(annotation)
    assert world.library_left("paper.pdf")

    scope = DocumentScope(slug)
    scope.unlink(world.projects, str(host))
    world.references.release(scope.removal(str(host)))

    assert not world.has(annotation)
    assert not world.library_left("paper.pdf") and world.mineru_kept("paper.pdf")
    assert host.read_bytes() == PDF


def test_removing_one_upload_releases_its_library_copy_unless_still_attached(world: World) -> None:
    world.library_pdf("paper.pdf")
    world.assets.write("upload", "attachment/chat/c1/paper.pdf", PDF, mime="application/pdf")
    world.assets.write("upload", "attachment/chat/c2/paper.pdf", PDF, mime="application/pdf")

    world.references.release(Released(owned={"attachment/chat/c1/paper.pdf"}))
    assert world.library_left("paper.pdf")

    world.references.release(Released(owned={"attachment/chat/c2/paper.pdf"}))
    assert not world.library_left("paper.pdf") and world.mineru_kept("paper.pdf")


def test_dry_run_lists_existing_orphans_and_deletes_nothing(world: World) -> None:
    orphan_annotation = world.library_pdf("orphan.pdf", b"%PDF orphan")
    kept_annotation = world.library_pdf("kept.pdf")
    world.chat("one.json", "c1", [pdf_attachment("kept.pdf")])
    world.assets.write("upload", "attachment/chat/gone/a.txt", b"a")

    orphans = world.references.orphans()

    assert orphans == sorted(["PDFs/orphan.pdf", orphan_annotation])
    assert kept_annotation not in orphans
    assert [asset.logical_path for asset in world.references.orphan_uploads()] == ["attachment/chat/gone/a.txt"]
    assert world.references.size("PDFs/orphan.pdf") == len(b"%PDF orphan")
    assert world.library_left("orphan.pdf") and world.mineru_kept("orphan.pdf") and world.has(orphan_annotation)


def test_mineru_output_is_never_deleted_even_when_nothing_references_its_pdf(world: World) -> None:
    world.library_pdf("paper.pdf")
    world.assets.write("upload", "attachment/chat/c1/paper.pdf", PDF, mime="application/pdf")

    assert not any(key.startswith("Mineru/") for key in world.references.orphans())
    gone = world.references.release(Released(owned={"attachment/chat/c1/paper.pdf"}, refs={"Mineru/paper.md"}))

    assert "PDFs/paper.pdf" in gone and not any(key.startswith("Mineru/") for key in gone)
    assert world.mineru_kept("paper.pdf")


@pytest.mark.parametrize("change", ["deleted", "rewritten"])
def test_an_annotation_opened_from_the_file_tree_lives_while_a_recorded_host_file_keeps_its_bytes(
    world: World, tmp_path, change: str
) -> None:
    first, second = tmp_path / "paper.pdf", tmp_path / "copy.pdf"
    first.write_bytes(PDF)
    second.write_bytes(PDF)
    sha256 = hashlib.sha256(PDF).hexdigest()
    annotation = annotation_canvas(world.docs, sha256, str(first))
    stroke = {"page": 1, "points": [[1, 2, 0.5]], "color": "#000000", "width": 2}
    CanvasMutationStore(world.docs._scope).apply(annotation, [CanvasMutation(uuid4(), "upsert", "stroke", "s1", stroke)])
    assert annotation_canvas(world.docs, sha256, str(second)) == annotation
    world.assets.write("upload", "attachment/chat/c1/paper.pdf", PDF, mime="application/pdf")

    world.references.release(Released(owned={"attachment/chat/c1/paper.pdf"}))
    assert world.has(annotation) and annotation not in world.references.orphans()

    def change_host(host) -> None:
        if change == "deleted":
            host.unlink()
        else:
            host.write_bytes(PDF + b" edited")

    change_host(first)
    world.references.release(Released(refs={annotation}))
    assert world.has(annotation)

    change_host(second)
    assert world.references.orphans() == [annotation]
    world.references.release(Released(refs={annotation}))
    assert not world.has(annotation)


def test_chat_meta_lists_attachment_references(world: World) -> None:
    world.chat("one.json", "c1", [pdf_attachment("paper.pdf"), {"name": "n.md", "mime": "text/markdown", "vaultPath": "/x/n.md"}])

    meta = world.chats.tree_metas()["chat/one.json"]

    assert meta["refs"] == ["/x/n.md", "PDFs/paper.pdf"]


async def test_chat_meta_stored_before_refs_existed_is_derived_again(world: World) -> None:
    world.library_pdf("paper.pdf")
    world.chat("one.json", "c1", [])
    world.chat("two.json", "c2", [pdf_attachment("paper.pdf")])
    with world.docs._scope.connection() as connection:
        connection.execute("UPDATE documents SET meta = meta - 'refs' WHERE key LIKE 'chat/%'")

    await world.cleanup.after(lambda released: world.chats.delete("one.json", released=released))
    world.references.release(Released(refs={"PDFs/paper.pdf"}))

    assert world.library_left("paper.pdf")
    assert world.chats.tree_metas()["chat/two.json"]["refs"] == ["PDFs/paper.pdf"]
