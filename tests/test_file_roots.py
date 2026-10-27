from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from backend.routes import documents as documents_route
from backend.routes import file_attach as file_attach_route
from backend.routes import file_git as file_git_route
from backend.routes import file_roots as file_roots_route
from backend.routes import fileviewer as fileviewer_route
from lib.attachment_materialize import sync_library_pdf
from lib.attachments import attach_library_pdf
from lib.data_store import StorageNotFoundError
from lib.file_roots import FileRoots
from lib.host_access import list_directories
from lib.storage_namespace import library_images_prefix, library_markdown
from tests.conftest import git, mounted_roots, route_client
from tests.fakes import MemoryAssetStore


def test_list_directories_lists_only_subdirectories_case_insensitively_with_hidden(tmp_path):
    for name in ("beta", "Alpha", ".config"):
        (tmp_path / name).mkdir()
    (tmp_path / "notes.md").write_text("not a directory")

    directory, names = list_directories(str(tmp_path / "beta" / ".."))

    assert directory == tmp_path.resolve()
    assert names == [".config", "Alpha", "beta"]


def test_list_directories_refuses_files_and_missing_paths(tmp_path):
    (tmp_path / "notes.md").write_text("")
    for path in (tmp_path / "notes.md", tmp_path / "missing"):
        with pytest.raises(NotADirectoryError):
            list_directories(str(path))


def test_tree_lists_every_non_ignored_file_by_full_name(tmp_path):
    (tmp_path / ".gitignore").write_text("build/\n*.log\n")
    for name in ("main.py", "App.tsx", "Untitled.md", "paper.pdf", "debug.log", "build/out.js"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text("")
    [root] = mounted_roots(tmp_path).tree()

    assert [child["name"] for child in root["children"]] == [".gitignore", "App.tsx", "main.py", "paper.pdf", "Untitled.md"]


def test_listing_follows_an_edited_gitignore(tmp_path):
    for name in ("notes.md", "debug.log"):
        (tmp_path / name).write_text("")
    (tmp_path / ".gitignore").write_text("*.log\n")
    roots = mounted_roots(tmp_path)
    assert [file["name"] for file in roots.list_files()] == [".gitignore", "notes.md"]

    (tmp_path / ".gitignore").write_text("*.md\nnotes/\n")

    assert [file["name"] for file in roots.list_files()] == [".gitignore", "debug.log"]
    [root] = roots.tree()
    assert [child["name"] for child in root["children"]] == [".gitignore", "debug.log"]



class _Project:
    """In-memory stand-in for one project's `files` list."""

    def __init__(self) -> None:
        self.files: list[str] = []

    def require(self, slug: str) -> None:
        """Every slug names an existing project."""

    def list_files(self, slug: str) -> list[str]:
        return list(self.files)

    def add_file(self, slug: str, path: str) -> list[str]:
        if path not in self.files:
            self.files.append(path)
        return self.files

    def remove_file(self, slug: str, path: str) -> list[str]:
        self.files.remove(path)
        return self.files

    def replace_file(self, slug: str, old: str, new: str) -> list[str]:
        self.files = [new if f == old else f for f in self.files]
        return self.files


def _reference_client(roots: FileRoots, project: _Project) -> TestClient:
    return route_client(
        file_attach_route.router,
        documents_route.router,
        overrides={
            file_attach_route.get_file_roots: roots,
            file_attach_route.get_project_store: project,
            file_attach_route.get_asset_store: None,
            documents_route.get_architecture_store: SimpleNamespace(source_paths=lambda: []),
        },
    )


def test_a_reference_follows_its_file_moved_inside_the_mount_but_never_guesses(tmp_path):
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    note = tmp_path / "old" / "notes.md"
    note.write_text("v1")
    client = _reference_client(mounted_roots(tmp_path), _Project())

    note.rename(tmp_path / "new" / "notes.md")
    moved = client.post("/api/fileroots/attach", params={"path": str(note)})
    (tmp_path / "old" / "notes.md").write_text("another")
    (tmp_path / "new" / "notes.md").rename(tmp_path / "notes.md")
    ambiguous = client.post("/api/fileroots/attach", params={"path": str(tmp_path / "gone" / "notes.md")})

    assert moved.json()["path"] == str((tmp_path / "new" / "notes.md").resolve())
    assert moved.json()["content"] == "v1"
    assert ambiguous.status_code == 404


def test_listing_a_project_rewrites_a_reference_whose_file_moved(tmp_path):
    (tmp_path / "docs").mkdir()
    note = tmp_path / "notes.md"
    note.write_text("v1")
    project = _Project()
    client = _reference_client(mounted_roots(tmp_path), project)
    client.post("/api/fileroots/add-to-project", json={"slug": "test", "paths": [str(note)]})

    note.rename(tmp_path / "docs" / "notes.md")
    listed = client.get("/api/documents", params={"slug": "test"})

    moved = str((tmp_path / "docs" / "notes.md").resolve())
    assert [document["path"] for document in listed.json()["documents"]] == [moved]
    assert project.files == [moved]


def test_a_changed_pdf_drops_its_stale_extraction_and_an_unchanged_one_keeps_it():
    assets = MemoryAssetStore()
    sync_library_pdf(assets, "paper.pdf", b"%PDF v1")
    assets.write("mineru_markdown", library_markdown("paper"), b"v1 text")
    assets.write("mineru_image", f"{library_images_prefix('paper')}/paper_1.png", b"img")
    assets.write("mineru_image", f"{library_images_prefix('paper-2')}/paper-2_1.png", b"other")

    sync_library_pdf(assets, "paper.pdf", b"%PDF v1")
    kept = set(assets.assets)
    sync_library_pdf(assets, "paper.pdf", b"%PDF v2")

    assert library_markdown("paper") in kept
    assert assets.assets["PDFs/paper.pdf"].content == b"%PDF v2"
    assert library_markdown("paper") not in assets.assets
    assert f"{library_images_prefix('paper')}/paper_1.png" not in assets.assets
    assert f"{library_images_prefix('paper-2')}/paper-2_1.png" in assets.assets


def test_workspace_files_join_a_project_by_reference_and_leave_it_without_touching_disk(tmp_path):
    note = tmp_path / "AGENTS.md"
    note.write_text("v1")
    project = _Project()
    client = route_client(
        file_attach_route.router,
        documents_route.router,
        overrides={
            file_attach_route.get_file_roots: mounted_roots(tmp_path),
            file_attach_route.get_project_store: project,
            file_attach_route.get_asset_store: None,
            documents_route.get_document_store: None,
        },
    )

    added = client.post("/api/fileroots/add-to-project", json={"slug": "test", "paths": [str(note)]})

    assert added.status_code == 200
    assert project.files == [str(note.resolve())]
    assert added.json()["documents"][0]["path"] == str(note.resolve())

    removed = client.delete("/api/documents", params={"slug": "test", "path": str(note.resolve())})

    assert removed.status_code == 200
    assert project.files == []
    assert note.read_text() == "v1"


def test_workspace_files_serve_raw_bytes_and_refuse_binary_as_text(tmp_path):
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\xff")
    client = route_client(
        fileviewer_route.router,
        overrides={
            fileviewer_route.get_file_roots: mounted_roots(tmp_path),
            fileviewer_route.get_document_store: None,
            fileviewer_route.get_asset_store: None,
        },
    )

    pdf = client.get("/api/fileviewer/raw", params={"path": str(tmp_path / "paper.pdf")})
    image_as_text = client.get("/api/fileviewer/text", params={"path": str(tmp_path / "image.png")})

    assert (pdf.status_code, pdf.content) == (200, b"%PDF-1.4")
    assert image_as_text.status_code == 415

    # Unchanged host files revalidate to 304, FileResponse alone would resend the bytes.
    assert pdf.headers["cache-control"] == "private, no-cache"
    raw_url = "/api/fileviewer/raw"
    reopened = client.get(raw_url, params={"path": str(tmp_path / "paper.pdf")}, headers={"If-None-Match": pdf.headers["etag"]})
    assert (reopened.status_code, reopened.content, reopened.headers["etag"]) == (304, b"", pdf.headers["etag"])
    ranged = client.get(raw_url, params={"path": str(tmp_path / "paper.pdf")}, headers={"Range": "bytes=1-3"})
    assert (ranged.status_code, ranged.content) == (206, b"PDF")


def test_attaching_an_extracted_library_pdf_reads_only_its_markdown():
    assets = MemoryAssetStore()
    assets.write("pdf", "PDFs/paper.pdf", b"%PDF v1")
    assets.write("mineru_markdown", library_markdown("paper"), b"# Paper")

    attached = attach_library_pdf(assets, "PDFs/paper.pdf")

    assert attached.parsedMd == "# Paper"
    assert assets.read_keys == [library_markdown("paper")]
    with pytest.raises(StorageNotFoundError):
        attach_library_pdf(assets, "PDFs/missing.pdf")


def test_diff_routes_serve_mounted_repositories_and_refuse_other_directories(tmp_path):
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    (mounted / "note.md").write_text("hi\n")
    client = route_client(file_git_route.router, overrides={file_git_route.get_file_roots: mounted_roots(mounted)})

    listing = client.get("/api/fileroots/diff/files", params={"root": str(mounted)})
    outside = client.get("/api/fileroots/diff/files", params={"root": str(tmp_path)})

    assert listing.json() == {"is_git_repository": False, "files": []}
    assert outside.status_code == 403


@pytest.mark.parametrize("host_access", [True, False])
def test_diff_routes_admit_linked_worktrees_anywhere_for_host_access_only(tmp_path, host_access):
    clone = tmp_path / "clone"
    clone.mkdir()
    git(clone, "init", "-q", "-b", "main")
    git(clone, "commit", "-q", "--allow-empty", "-m", "init")
    linked = tmp_path / "elsewhere" / "task"
    git(clone, "worktree", "add", "-q", str(linked), "-b", "task")
    (linked / "src").mkdir()
    (linked / "src" / "new.md").write_text("hi\n")
    pointing_elsewhere = tmp_path / "not-a-worktree"
    pointing_elsewhere.mkdir()
    (pointing_elsewhere / ".git").write_text("something else\n")
    roots = mounted_roots(host_access=host_access)
    client = route_client(file_git_route.router, overrides={file_git_route.get_file_roots: roots})

    def status(directory) -> int:
        return client.get("/api/fileroots/diff/files", params={"root": str(directory)}).status_code

    assert status(linked) == (200 if host_access else 403)
    assert status(linked / "src") == (200 if host_access else 403)
    assert status(clone) == 403
    assert status(tmp_path / "elsewhere") == 403
    assert status(pointing_elsewhere) == 403
    assert not roots.contains(linked / ".env")
    if host_access:
        assert roots.root_of(str(linked / "src" / "new.md")) == linked


@pytest.mark.parametrize("host_access", [False, True])
def test_host_access_reaches_unmounted_home_files_but_never_credentials(tmp_path, monkeypatch, host_access):
    home, outside = tmp_path / "home", tmp_path / "outside"
    mounted, elsewhere = home / "mounted", home / "elsewhere"
    monkeypatch.setenv("HOME", str(home))
    for directory in (mounted, elsewhere / ".ssh", elsewhere / ".config" / "gh", outside):
        directory.mkdir(parents=True)
    secrets = (elsewhere / ".env", elsewhere / ".ssh" / "id_ed25519", elsewhere / ".netrc", elsewhere / ".config" / "gh" / "hosts.yml")
    for path in (mounted / ".env", elsewhere / "notes.md", outside / "shadow", *secrets):
        path.write_text("v1")
    client = route_client(
        fileviewer_route.router,
        file_roots_route.router,
        overrides={
            fileviewer_route.get_file_roots: mounted_roots(mounted, host_access=host_access),
            fileviewer_route.get_document_store: None,
        },
    )

    def read_status(path) -> int:
        return client.get("/api/fileviewer/text", params={"path": str(path)}).status_code

    saved = client.post("/api/fileroots/file", json={"path": str(elsewhere / "notes.md"), "content": "v2"})

    assert read_status(mounted / ".env") == 200
    assert read_status(elsewhere / "notes.md") == (200 if host_access else 403)
    assert (saved.status_code < 400) == host_access
    assert (elsewhere / "notes.md").read_text() == ("v2" if host_access else "v1")
    assert read_status(outside / "shadow") == 403
    for secret in secrets:
        assert read_status(secret) == 403, secret


def test_unreadable_file_is_refused_not_a_server_error(tmp_path):
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    locked = mounted / "locked.md"
    locked.write_text("x")
    locked.chmod(0)
    client = route_client(
        fileviewer_route.router,
        overrides={fileviewer_route.get_file_roots: mounted_roots(mounted), fileviewer_route.get_document_store: None},
    )
    try:
        response = client.get("/api/fileviewer/text", params={"path": str(locked)})
    finally:
        locked.chmod(0o600)
    assert (response.status_code, response.json()["detail"]) == (403, "Unknown file path")
