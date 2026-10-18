from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend.routes import documents as documents_route
from backend.routes import file_roots as file_roots_route
from backend.routes import fileviewer as fileviewer_route
from lib.file_roots import FileRoots, list_directories


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
    roots = FileRoots(repository=SimpleNamespace(load=lambda: [str(tmp_path)], save=lambda roots: None))

    [root] = roots.tree()

    assert [child["name"] for child in root["children"]] == [".gitignore", "App.tsx", "main.py", "paper.pdf", "Untitled.md"]


class _Project:
    """In-memory stand-in for one project's `files` list."""

    def __init__(self) -> None:
        self.files: list[str] = []

    def list_files(self, slug: str) -> list[str]:
        return list(self.files)

    def add_file(self, slug: str, path: str) -> list[str]:
        if path not in self.files:
            self.files.append(path)
        return self.files

    def remove_file(self, slug: str, path: str) -> list[str]:
        self.files.remove(path)
        return self.files


def test_workspace_files_join_a_project_by_reference_and_leave_it_without_touching_disk(tmp_path):
    note = tmp_path / "AGENTS.md"
    note.write_text("v1")
    project = _Project()
    roots = FileRoots(repository=SimpleNamespace(load=lambda: [str(tmp_path)], save=lambda roots: None))
    app = FastAPI()
    app.include_router(file_roots_route.router)
    app.include_router(documents_route.router)
    app.dependency_overrides[file_roots_route.get_file_roots] = lambda: roots
    app.dependency_overrides[file_roots_route.get_project_store] = lambda: project
    app.dependency_overrides[file_roots_route.get_asset_store] = lambda: None
    app.dependency_overrides[documents_route.get_document_store] = lambda: None
    client = TestClient(app)

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
    roots = FileRoots(repository=SimpleNamespace(load=lambda: [str(tmp_path)], save=lambda roots: None))
    app = FastAPI()
    app.include_router(fileviewer_route.router)
    app.dependency_overrides[fileviewer_route.get_file_roots] = lambda: roots
    app.dependency_overrides[fileviewer_route.get_document_store] = lambda: None
    app.dependency_overrides[fileviewer_route.get_asset_store] = lambda: None
    client = TestClient(app)

    pdf = client.get("/api/fileviewer/raw", params={"path": str(tmp_path / "paper.pdf")})
    image_as_text = client.get("/api/fileviewer/text", params={"path": str(tmp_path / "image.png")})

    assert (pdf.status_code, pdf.content) == (200, b"%PDF-1.4")
    assert image_as_text.status_code == 415
