"""Renaming a file of a LaTeX project through `POST /api/documents/rename` repoints the references to it."""

import pytest

from backend.routes import documents as documents_route
from backend.routes.deps import (
    get_asset_store,
    get_disk_files,
    get_document_store,
    get_file_roots,
    get_project_store,
    get_references,
    get_relocate,
)
from lib import latex_references
from lib.chat_store import ChatStore
from lib.disk_files import DiskFiles
from lib.project_store import ProjectStore
from lib.references import References
from tests.conftest import mounted_roots, route_client
from tests.fakes import MemoryAssetStore, MemoryDataStore, RecordingScope

PAPER = "\\documentclass{article}\n\\begin{document}\n\\includegraphics{imgs/a}\n\\end{document}\n"


class Docs(MemoryDataStore):
    def is_file(self, key: str) -> bool:
        return self.exists(key)


class FailingFirstRecord(RecordingScope):
    """Fails to record its first change, keeps the later ones."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def record(self, connection, **change) -> None:
        if not self.failed:
            self.failed = True
            raise RuntimeError("database down")
        super().record(connection, **change)


def client_for(roots, scope):
    docs = Docs()
    assets = MemoryAssetStore()
    overrides = {
        get_file_roots: roots,
        get_project_store: ProjectStore(ChatStore(data_store=docs), rename=lambda *_: None),
        get_document_store: docs,
        get_asset_store: assets,
        get_relocate: lambda old, new: None,
        get_references: References(docs, assets),
        get_disk_files: DiskFiles(roots, scope),
    }
    return route_client(documents_route.router, overrides=overrides)


@pytest.fixture
def folder(tmp_path):
    folder = tmp_path.resolve() / "vault" / "paper"
    (folder / "imgs").mkdir(parents=True)
    (folder / "main.tex").write_text(PAPER, encoding="utf-8")
    (folder / "imgs" / "a.png").write_bytes(b"png")
    return folder


@pytest.fixture
def scope():
    return RecordingScope()


@pytest.fixture
def client(folder, scope):
    return client_for(mounted_roots(folder.parent), scope)


def rename(client, path, name):
    return client.post("/api/documents/rename", json={"path": str(path), "slug": "", "name": name})


def recorded(scope) -> list[str]:
    return [change["key"] for change in scope.changes]


def test_renaming_an_image_rewrites_the_root_that_includes_it_and_reports_and_records_the_change(client, folder, scope):
    response = rename(client, folder / "imgs" / "a.png", "b.png")

    assert response.status_code == 200
    assert response.json()["changed"] == [str(folder / "main.tex")]
    assert (folder / "imgs" / "b.png").is_file() and not (folder / "imgs" / "a.png").exists()
    assert (folder / "main.tex").read_text(encoding="utf-8") == PAPER.replace("imgs/a", "imgs/b")
    assert scope.changes == [{"kind": "document", "key": str(folder / "main.tex"), "operation": "write"}]


def test_a_folder_reached_only_by_host_access_cascades_too(tmp_path, monkeypatch):
    home = tmp_path.resolve() / "home"
    folder = home / "paper"
    (folder / "imgs").mkdir(parents=True)
    (folder / "main.tex").write_text(PAPER, encoding="utf-8")
    (folder / "imgs" / "a.png").write_bytes(b"png")
    monkeypatch.setenv("HOME", str(home))
    scope = RecordingScope()

    response = rename(client_for(mounted_roots(host_access=True), scope), folder / "imgs" / "a.png", "b.png")

    assert response.status_code == 200
    assert response.json()["changed"] == [str(folder / "main.tex")]
    assert (folder / "main.tex").read_text(encoding="utf-8") == PAPER.replace("imgs/a", "imgs/b")
    assert recorded(scope) == [str(folder / "main.tex")]


def test_renaming_the_root_rewrites_the_root_line_of_its_chapters_and_records_each_file_once(client, folder, scope):
    (folder / "sub").mkdir()
    for name in ("one", "two"):
        (folder / "sub" / f"{name}.tex").write_text("% !TEX root = ../main.tex\nbody\n", encoding="utf-8")

    response = rename(client, folder / "main.tex", "paper.tex")

    expected = [str(folder / "sub" / "one.tex"), str(folder / "sub" / "two.tex")]
    assert response.json()["changed"] == expected
    assert recorded(scope) == expected
    assert [(folder / "sub" / name).read_text(encoding="utf-8") for name in ("one.tex", "two.tex")] == ["% !TEX root = ../paper.tex\nbody\n"] * 2


def test_a_chapter_without_a_root_line_below_the_root_cascades_into_the_root(client, folder, scope):
    (folder / "ch").mkdir()
    (folder / "ch" / "one.tex").write_text("text\n", encoding="utf-8")
    (folder / "main.tex").write_text(PAPER.replace("\\includegraphics{imgs/a}", "\\input{ch/one}"), encoding="utf-8")

    response = rename(client, folder / "ch" / "one.tex", "two.tex")

    assert response.json()["changed"] == [str(folder / "main.tex")]
    assert "\\input{ch/two}" in (folder / "main.tex").read_text(encoding="utf-8")
    assert recorded(scope) == [str(folder / "main.tex")]


def test_a_crlf_file_keeps_its_line_endings(client, folder):
    (folder / "main.tex").write_bytes(PAPER.replace("\n", "\r\n").encode("utf-8"))

    assert rename(client, folder / "imgs" / "a.png", "b.png").status_code == 200

    assert (folder / "main.tex").read_bytes() == PAPER.replace("imgs/a", "imgs/b").replace("\n", "\r\n").encode("utf-8")


def test_a_rename_nothing_refers_to_changes_nothing(client, folder, scope):
    (folder / "notes.md").write_text("x", encoding="utf-8")

    response = rename(client, folder / "notes.md", "log.md")

    assert response.status_code == 200 and response.json()["changed"] == []
    assert (folder / "main.tex").read_text(encoding="utf-8") == PAPER and scope.changes == []


def test_a_failing_record_does_not_stop_the_other_files_from_being_recorded(folder):
    (folder / "ch.tex").write_text("\\includegraphics{imgs/a}\n", encoding="utf-8")
    scope = FailingFirstRecord()

    response = rename(client_for(mounted_roots(folder.parent), scope), folder / "imgs" / "a.png", "b.png")

    assert response.status_code == 200
    assert response.json()["changed"] == [str(folder / "ch.tex"), str(folder / "main.tex")]
    assert recorded(scope) == [str(folder / "main.tex")]
    assert (folder / "ch.tex").read_text(encoding="utf-8") == "\\includegraphics{imgs/b}\n"


def test_a_failing_rewrite_leaves_the_rename_in_place(client, folder, scope, monkeypatch):
    def failing(*_):
        raise OSError("disk full")

    monkeypatch.setattr(latex_references, "rewrite_references", failing)

    response = rename(client, folder / "imgs" / "a.png", "b.png")

    assert response.status_code == 200 and response.json()["changed"] == []
    assert (folder / "imgs" / "b.png").is_file() and not (folder / "imgs" / "a.png").exists()
    assert (folder / "main.tex").read_text(encoding="utf-8") == PAPER and scope.changes == []


def test_a_folder_without_a_root_document_keeps_its_files_as_they_are(client, folder, scope):
    (folder / "main.tex").unlink()
    for name in ("one.tex", "two.tex"):
        (folder / name).write_text("\\includegraphics{imgs/a}\n", encoding="utf-8")

    assert rename(client, folder / "imgs" / "a.png", "b.png").status_code == 200

    assert (folder / "one.tex").read_text(encoding="utf-8") == "\\includegraphics{imgs/a}\n" and scope.changes == []
