import json
import os

import pytest

from backend.identity import get_request_identity
from backend.routes import workbench as workbench_route
from backend.routes.deps import get_document_store, get_file_roots, get_project_store, get_relocate
import config
from lib import latex, workbench
from lib.chat_store import ChatStore
from lib.data_store import StorageConflictError
from lib.project_store import ProjectStore
from lib.storage_namespace import WORKBENCH_SESSIONS
from tests.conftest import mounted_roots, request_identity, route_client
from tests.fakes import MemoryDataStore


class Docs(MemoryDataStore):
    def is_file(self, key: str) -> bool:
        return self.exists(key)


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = tmp_path / "workbench"
    monkeypatch.setattr(config, "WORKBENCH", root)
    return root


@pytest.fixture
def vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture
def docs():
    return Docs()


@pytest.fixture
def projects(docs):
    projects = ProjectStore(ChatStore(data_store=docs), rename=lambda *_: None)
    projects.create_project("Thesis")
    return projects


def workbench_client(projects, docs, roots, relocate=None, host_access=True):
    """A workbench route client for *projects* and *docs* that reaches *roots*."""

    def relocate_to_disk(old: str, new: str) -> None:
        """Like `PostgresKeyRelocator.relocate` for a disk path: project lists point at it, the stored row goes."""
        for project in projects.list_projects():
            projects.replace_file(project["slug"], old, new)
        docs.delete(old)

    overrides = {
        get_request_identity: request_identity(has_host_access=host_access),
        get_file_roots: roots,
        get_project_store: projects,
        get_document_store: docs,
        get_relocate: relocate or relocate_to_disk,
    }
    return route_client(workbench_route.router, overrides=overrides)


@pytest.fixture
def client(root, vault, docs, projects):
    return workbench_client(projects, docs, mounted_roots(vault, root))


@pytest.mark.parametrize(("path", "kind"), [("a.tex", "latex"), ("/x/B.TEX", "latex"), ("n.md", "markdown"), ("c.Md", "markdown"),
                                           ("a.pdf", None), ("tex", None), ("a.md.txt", None)])
def test_kind_comes_from_the_suffix_ignoring_case(path, kind):
    assert workbench.kind_of(path) == kind


def test_documents_live_in_a_folder_per_kind_project_and_stem(tmp_path):
    assert workbench.kind_directory(tmp_path, "latex") == tmp_path / "latex"
    assert workbench.document_folder(tmp_path, "markdown", "thesis", "paper") == tmp_path / "markdown" / "thesis" / "paper"


@pytest.mark.parametrize(("slug", "stem"), [("", "a"), ("a", ""), ("..", "a"), ("a", ".omp"), ("a/b", "c"), ("a", "b/c")])
def test_folder_segments_cannot_be_empty_nested_or_hidden(tmp_path, slug, stem):
    with pytest.raises(ValueError):
        workbench.document_folder(tmp_path, "latex", slug, stem)


def test_mirror_overwrites_hand_edits_of_every_kind(tmp_path):
    system = tmp_path / "latex" / ".omp" / "SYSTEM.md"
    system.parent.mkdir(parents=True)
    system.write_text("hand edit", encoding="utf-8")
    workbench.mirror_omp(tmp_path)
    assert "LaTeX" in system.read_text(encoding="utf-8")
    assert "Markdown" in (tmp_path / "markdown" / ".omp" / "SYSTEM.md").read_text(encoding="utf-8")
    for kind in workbench.KINDS:
        assert "context-file:project:AGENTS.md" in (tmp_path / kind / ".omp" / "config.yml").read_text(encoding="utf-8")


def test_mirror_leaves_unchanged_files_untouched(tmp_path):
    workbench.mirror_omp(tmp_path)
    mirrored = [tmp_path / kind / ".omp" / name for kind in workbench.KINDS for name in ("SYSTEM.md", "config.yml")]
    for path in mirrored:
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    workbench.mirror_omp(tmp_path)
    assert [path.stat().st_mtime_ns for path in mirrored] == [1_000_000_000] * len(mirrored)


def test_mirror_continues_past_a_failing_kind(tmp_path):
    (tmp_path / "latex").mkdir()
    (tmp_path / "latex" / ".omp").write_text("not a directory", encoding="utf-8")
    workbench.mirror_omp(tmp_path)
    assert "Markdown" in (tmp_path / "markdown" / ".omp" / "SYSTEM.md").read_text(encoding="utf-8")


def test_project_directories_lists_existing_ones_only(tmp_path):
    (tmp_path / "markdown" / "thesis").mkdir(parents=True)
    (tmp_path / "latex" / "thesis-2").mkdir(parents=True)
    assert workbench.project_directories(tmp_path, "thesis") == [tmp_path / "markdown" / "thesis"]


def test_moving_project_directories_puts_back_earlier_moves_when_one_is_taken(tmp_path):
    (tmp_path / "latex" / "old" / "paper").mkdir(parents=True)
    (tmp_path / "markdown" / "old").mkdir(parents=True)
    (tmp_path / "markdown" / "new").mkdir(parents=True)
    (tmp_path / "markdown" / "new" / "kept.md").write_text("kept", encoding="utf-8")
    with pytest.raises(FileExistsError):
        workbench.move_project_directories(tmp_path, "old", "new")
    assert (tmp_path / "latex" / "old" / "paper").is_dir() and not (tmp_path / "latex" / "new").exists()
    assert (tmp_path / "markdown" / "new" / "kept.md").is_file()

    (tmp_path / "markdown" / "new" / "kept.md").unlink()
    moves = workbench.move_project_directories(tmp_path, "old", "new")
    assert [target for _, target in moves] == [tmp_path / "latex" / "new", tmp_path / "markdown" / "new"]
    workbench.restore_directories(moves)
    assert (tmp_path / "latex" / "old" / "paper").is_dir() and (tmp_path / "markdown" / "old").is_dir()


def test_prepare_moves_a_stored_project_file_into_its_folder(client, root, docs, projects):
    key = "project/thesis/document/paper.tex"
    docs.write_bytes(key, b"\\documentclass{article}")
    projects.add_file("thesis", key)

    response = client.post("/api/workbench/prepare", json={"path": key})

    source = root / "latex" / "thesis" / "paper" / "paper.tex"
    assert response.json() == {"path": str(source), "cwd": str(root / "latex"), "kind": "latex"}
    assert source.read_bytes() == b"\\documentclass{article}"
    assert projects.list_files("thesis") == [str(source)]
    assert not docs.exists(key)


def test_moving_onto_an_empty_leftover_folder_replaces_it(tmp_path):
    (tmp_path / "latex" / "old" / "paper").mkdir(parents=True)
    (tmp_path / "latex" / "new" / "stale").mkdir(parents=True)
    moves = workbench.move_project_directories(tmp_path, "old", "new")
    assert moves == [(tmp_path / "latex" / "old", tmp_path / "latex" / "new")]
    assert (tmp_path / "latex" / "new" / "paper").is_dir() and not (tmp_path / "latex" / "new" / "stale").exists()


def test_prepare_refuses_notes_and_keeps_them_stored(client, root, docs):
    docs.write_bytes("note/idea.md", b"# Idea")

    assert client.post("/api/workbench/prepare", json={"path": "note/idea.md"}).status_code == 400
    assert docs.read_bytes("note/idea.md")[0] == b"# Idea"
    assert not root.exists()


def test_prepare_refuses_a_workbench_folder_outside_the_allowed_roots(root, vault, docs, projects):
    key = "project/thesis/document/paper.tex"
    docs.write_bytes(key, b"stored")
    projects.add_file("thesis", key)
    client = workbench_client(projects, docs, mounted_roots(vault))

    assert client.post("/api/workbench/prepare", json={"path": key}).status_code == 403
    assert docs.read_bytes(key)[0] == b"stored"
    assert projects.list_files("thesis") == [key]
    assert not root.exists()


def test_prepare_removes_the_folders_it_made_when_the_move_fails(root, vault, docs, projects):
    key = "project/thesis/document/plan.md"
    docs.write_bytes(key, b"stored")
    projects.add_file("thesis", key)

    def relocate(old: str, new: str) -> None:
        raise StorageConflictError(old)

    client = workbench_client(projects, docs, mounted_roots(vault, root), relocate=relocate)
    root.mkdir()
    assert client.post("/api/workbench/prepare", json={"path": key}).status_code == 409
    assert docs.read_bytes(key)[0] == b"stored"
    assert list(root.iterdir()) == []


def test_prepare_refuses_a_taken_file_name_and_keeps_the_stored_document(client, root, docs, projects):
    key = "project/thesis/document/plan.md"
    docs.write_bytes(key, b"stored")
    projects.add_file("thesis", key)
    taken = root / "markdown" / "thesis" / "plan" / "plan.md"
    taken.parent.mkdir(parents=True)
    taken.write_text("on disk", encoding="utf-8")

    assert client.post("/api/workbench/prepare", json={"path": key}).status_code == 409
    assert taken.read_text(encoding="utf-8") == "on disk"
    assert docs.read_bytes(key)[0] == b"stored"
    assert projects.list_files("thesis") == [key]


def test_prepare_leaves_an_absolute_file_in_place(client, root, vault):
    source = vault / "repo" / "README.md"
    source.parent.mkdir()
    source.write_text("# Repo", encoding="utf-8")

    response = client.post("/api/workbench/prepare", json={"path": str(source)})

    assert response.json() == {"path": str(source), "cwd": str(root / "markdown"), "kind": "markdown"}
    assert source.read_text(encoding="utf-8") == "# Repo"


def test_prepare_answers_the_resolved_absolute_path(client, root, vault):
    source = vault / "repo" / "README.md"
    source.parent.mkdir()
    source.write_text("# Repo", encoding="utf-8")
    (vault / "link.md").symlink_to(source)

    for spelling in (vault / "repo" / ".." / "repo" / "README.md", vault / "link.md"):
        response = client.post("/api/workbench/prepare", json={"path": str(spelling)})
        assert response.json()["path"] == str(source.resolve())
    assert client.post("/api/workbench/prepare", json={"path": "/etc/hostname.md"}).status_code == 403


def test_prepare_refuses_unknown_documents_and_kinds(client, vault, docs, projects):
    docs.write_bytes("project/thesis/document/a.pdf", b"%PDF")
    projects.add_file("thesis", "project/thesis/document/a.pdf")
    other = vault / "notes.txt"
    other.write_text("x", encoding="utf-8")

    def status(path: str) -> int:
        return client.post("/api/workbench/prepare", json={"path": path}).status_code

    assert status("project/thesis/document/a.pdf") == 400
    assert status(str(other)) == 400
    assert status("project/thesis/document/missing.md") == 404
    assert status("../x.md") == 400
    assert docs.exists("project/thesis/document/a.pdf")


def test_create_writes_a_new_document_into_its_folder_and_lists_it(client, root, projects):
    response = client.post("/api/workbench/create", json={"slug": "thesis", "name": "paper.tex"})

    source = root / "latex" / "thesis" / "paper" / "paper.tex"
    assert response.json() == {"path": str(source)}
    assert source.read_text(encoding="utf-8") == latex.NEW_DOCUMENT
    assert projects.list_files("thesis") == [str(source)]

    created = client.post("/api/workbench/create", json={"slug": "thesis", "name": "plan.md"}).json()["path"]
    assert (root / "markdown" / "thesis" / "plan" / "plan.md").read_text(encoding="utf-8") == ""
    assert created == str(root / "markdown" / "thesis" / "plan" / "plan.md")


@pytest.mark.parametrize(("body", "status"), [
    ({"slug": "thesis", "name": "paper.pdf"}, 400),
    ({"slug": "thesis", "name": "a/b.tex"}, 400),
    ({"slug": "thesis", "name": "pa\x00per.tex"}, 400),
    ({"slug": "thesis", "name": "pa\nper.md"}, 400),
    ({"slug": "missing", "name": "paper.tex"}, 404),
])
def test_create_refuses_bad_names_and_unknown_projects(client, root, body, status):
    assert client.post("/api/workbench/create", json=body).status_code == status
    assert not (root / "latex").exists()


def test_create_never_overwrites_an_existing_document(client, root):
    source = root / "latex" / "thesis" / "paper" / "paper.tex"
    source.parent.mkdir(parents=True)
    source.write_text("mine", encoding="utf-8")
    assert client.post("/api/workbench/create", json={"slug": "thesis", "name": "paper.tex"}).status_code == 409
    assert source.read_text(encoding="utf-8") == "mine"


def test_create_removes_the_file_when_listing_it_fails(client, root, projects, monkeypatch):
    def fail(slug: str, path: str) -> None:
        raise StorageConflictError(slug)

    monkeypatch.setattr(projects, "add_file", fail)
    with pytest.raises(StorageConflictError):
        client.post("/api/workbench/create", json={"slug": "thesis", "name": "paper.tex"})
    assert not (root / "latex").exists()


def test_session_links_round_trip_per_file(client, docs):
    path = "/data/workbench/latex/thesis/paper/paper.tex"
    assert client.get("/api/workbench/session", params={"path": path}).json() == {"session_id": None}

    assert client.put("/api/workbench/session", json={"path": path, "session_id": "s1"}).json() == {"ok": True}
    client.put("/api/workbench/session", json={"path": "/other.md", "session_id": "s2"})

    assert client.get("/api/workbench/session", params={"path": path}).json() == {"session_id": "s1"}
    assert json.loads(docs.read_bytes(WORKBENCH_SESSIONS)[0]) == {path: "s1", "/other.md": "s2"}

    client.put("/api/workbench/session", json={"path": path, "session_id": None})
    assert client.get("/api/workbench/session", params={"path": path}).json() == {"session_id": None}
    assert client.put("/api/workbench/session", json={"path": "relative.md", "session_id": "s"}).status_code == 400


@pytest.mark.parametrize(("method", "url", "request_args"), [
    ("post", "/api/workbench/prepare", {"json": {"path": "project/thesis/document/paper.tex"}}),
    ("post", "/api/workbench/create", {"json": {"slug": "thesis", "name": "paper.tex"}}),
    ("get", "/api/workbench/session", {"params": {"path": "/a.md"}}),
    ("put", "/api/workbench/session", {"json": {"path": "/a.md", "session_id": "s"}}),
])
def test_logins_without_host_access_are_refused(root, vault, docs, projects, method, url, request_args):
    key = "project/thesis/document/paper.tex"
    docs.write_bytes(key, b"stored")
    projects.add_file("thesis", key)
    client = workbench_client(projects, docs, mounted_roots(vault, root), host_access=False)

    assert getattr(client, method)(url, **request_args).status_code == 403
    assert docs.read_bytes(key)[0] == b"stored" and not docs.exists(WORKBENCH_SESSIONS)
    assert not root.exists()
