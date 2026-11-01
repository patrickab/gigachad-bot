from datetime import datetime
import json
import os

import pytest

from backend.identity import get_request_identity
from backend.routes import workbench as workbench_route
from backend.routes.deps import (
    get_asset_store,
    get_disk_files,
    get_document_store,
    get_file_roots,
    get_project_store,
    get_relocate,
)
import config
from lib import latex, workbench
from lib.chat_store import ChatStore
from lib.data_store import StorageConflictError
from lib.disk_files import DiskFiles
from lib.project_store import ProjectStore
from lib.storage_namespace import WORKBENCH_SESSIONS
from tests.conftest import mounted_roots, request_identity, route_client
from tests.fakes import MemoryAssetStore, MemoryDataStore, RecordingScope, StoredAsset


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


@pytest.fixture
def scope():
    return RecordingScope()


def workbench_client(projects, docs, roots, relocate=None, host_access=True, assets=None, scope=None):
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
        get_asset_store: assets or MemoryAssetStore(),
        get_disk_files: DiskFiles(roots, scope or RecordingScope()),
    }
    return route_client(workbench_route.router, overrides=overrides)


@pytest.fixture
def client(root, vault, docs, projects, scope):
    return workbench_client(projects, docs, mounted_roots(vault, root), scope=scope)


class MirroringAssets(MemoryAssetStore):
    """Writes mirrors to disk like `AssetStore.mirror`."""

    def mirror(self, asset, root):
        path = root.joinpath(*asset.logical_path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(asset.content)
        return path


@pytest.fixture
def documents(tmp_path, monkeypatch):
    documents = tmp_path / "Documents"
    monkeypatch.setattr(config, "DOCUMENTS", documents)
    return documents


def test_a_stored_pdf_is_read_from_its_library_copy_on_disk_and_never_moved(root, vault, docs, projects, documents):
    assets = MirroringAssets()
    assets.assets["PDFs/paper.pdf"] = StoredAsset("PDFs/paper.pdf", b"%PDF-1", "pdf", "application/pdf")
    client = workbench_client(projects, docs, mounted_roots(vault, root), assets=assets)

    prepared = client.post("/api/workbench/prepare", json={"path": "PDFs/paper.pdf"}).json()
    assert prepared == {"path": str(documents / "PDFs" / "paper.pdf"), "cwd": str(root / "pdf"), "kind": "pdf"}
    assert (documents / "PDFs" / "paper.pdf").read_bytes() == b"%PDF-1"
    assert "PDFs/paper.pdf" in assets.assets

    # An existing copy is read as it is
    assets.read_keys.clear()
    assert client.post("/api/workbench/prepare", json={"path": "PDFs/paper.pdf"}).json()["path"] == prepared["path"]
    assert assets.read_keys == []
    assert client.post("/api/workbench/prepare", json={"path": "PDFs/missing.pdf"}).status_code == 404


def test_a_mounted_pdf_is_read_in_place_and_new_pdfs_are_refused(client, vault, root):
    pdf = vault / "slides.pdf"
    pdf.write_bytes(b"%PDF-1")
    assert client.post("/api/workbench/prepare", json={"path": str(pdf)}).json() == {"path": str(pdf), "cwd": str(root / "pdf"), "kind": "pdf"}
    assert client.post("/api/workbench/create", json={"slug": "thesis", "name": "x.pdf"}).status_code == 400


@pytest.mark.parametrize(("path", "kind"), [("/x/B.TEX", "latex"), ("refs.bib", "latex"), ("n.md", "markdown"),
                                           ("a.pdf", "pdf"), ("tex", None), ("a.bibtex", None)])
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
    docs.write_bytes("project/thesis/document/a.csv", b"a,b")
    projects.add_file("thesis", "project/thesis/document/a.csv")
    other = vault / "notes.txt"
    other.write_text("x", encoding="utf-8")

    def status(path: str) -> int:
        return client.post("/api/workbench/prepare", json={"path": path}).status_code

    assert status("project/thesis/document/a.csv") == 400
    assert status(str(other)) == 400
    assert status("project/thesis/document/missing.md") == 404
    assert status("../x.md") == 400
    assert docs.exists("project/thesis/document/a.csv")


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


def test_a_file_keeps_every_session_while_new_and_picked_ones_become_active(client, docs):
    path = "/data/workbench/latex/thesis/paper/paper.tex"

    def linked() -> dict:
        return client.get("/api/workbench/session", params={"path": path}).json()

    def link(session_id: str | None, file: str = path) -> dict:
        return client.put("/api/workbench/session", json={"path": file, "session_id": session_id}).json()

    assert linked() == {"session_id": None, "session_ids": [], "moved": []}
    assert link("s1") == {"session_id": "s1", "session_ids": ["s1"], "moved": []}
    link("other", "/other.md")

    # Reset: a new session is active, the earlier one stays listed until the new one is linked
    assert link(None) == {"session_id": None, "session_ids": ["s1"], "moved": []}
    link("s2")
    # Picking an earlier session makes it active without reordering or duplicating it
    assert link("s1") == {"session_id": "s1", "session_ids": ["s1", "s2"], "moved": []}
    assert linked() == {"session_id": "s1", "session_ids": ["s1", "s2"], "moved": []}
    assert client.get("/api/workbench/session", params={"path": "/other.md"}).json() == {"session_id": "other", "session_ids": ["other"], "moved": []}
    assert client.put("/api/workbench/session", json={"path": "relative.md", "session_id": "s"}).status_code == 400


def test_a_bare_session_id_stored_before_session_lists_reads_as_one_active_session(client, docs):
    docs.write_bytes(WORKBENCH_SESSIONS, json.dumps({"/a.md": "old"}).encode(), expected=None)
    assert client.get("/api/workbench/session", params={"path": "/a.md"}).json() == {"session_id": "old", "session_ids": ["old"], "moved": []}
    client.put("/api/workbench/session", json={"path": "/a.md", "session_id": "new"})
    assert json.loads(docs.read_bytes(WORKBENCH_SESSIONS)[0]) == {"/a.md": {"active": "new", "sessions": ["old", "new"]}}


def test_sessions_of_a_moved_file_name_it_again_until_each_was_told(client, docs):
    links = {"/b.md": {"active": "s2", "sessions": ["s1", "s2"], "moved": ["s1", "s2"]}}
    docs.write_bytes(WORKBENCH_SESSIONS, json.dumps(links).encode(), expected=None)

    # Switching sessions keeps what each still has to be told
    assert client.put("/api/workbench/session", json={"path": "/b.md", "session_id": "s1"}).json()["moved"] == ["s1", "s2"]
    told = client.post("/api/workbench/session/informed", json={"path": "/b.md", "session_id": "s1"}).json()

    assert told == {"session_id": "s1", "session_ids": ["s1", "s2"], "moved": ["s2"]}
    client.post("/api/workbench/session/informed", json={"path": "/b.md", "session_id": "s2"})
    assert json.loads(docs.read_bytes(WORKBENCH_SESSIONS)[0]) == {"/b.md": {"active": "s1", "sessions": ["s1", "s2"]}}


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


def test_supporting_latex_files_prepare_in_place_but_are_not_created(client, vault, root):
    bibliography = vault / "refs.bib"
    bibliography.write_text("@book{a}", encoding="utf-8")

    prepared = client.post("/api/workbench/prepare", json={"path": str(bibliography)}).json()

    assert prepared == {"path": str(bibliography), "cwd": str(root / "latex"), "kind": "latex"}
    assert client.post("/api/workbench/create", json={"slug": "thesis", "name": "refs.bib"}).status_code == 400


def listed_latex(root, projects, stem, slug="thesis", name=None):
    """A project's LaTeX file `<root>/latex/<slug>/<stem>/<stem>.tex`, listed in project *slug* (shown as *name*)."""
    source = root / "latex" / slug / stem / f"{stem}.tex"
    source.parent.mkdir(parents=True)
    source.write_text(latex.NEW_DOCUMENT, encoding="utf-8")
    projects.add_file(slug, str(source))
    if name:
        projects.set_file_name(slug, str(source), name)
    return source


def promote(client, path, slug="thesis"):
    return client.post("/api/workbench/promote", json={"slug": slug, "path": str(path)})


def test_promoting_replaces_the_entry_by_its_folder_in_place_for_good(client, root, projects, docs):
    projects.add_file("thesis", "/vault/a.md")
    paper = listed_latex(root, projects, "paper")
    other = listed_latex(root, projects, "other")

    promoted = promote(client, paper)

    assert promoted.json() == {"folder": str(paper.parent)}
    assert projects.list_named_files("thesis") == [("/vault/a.md", "a.md"), (str(other), "other.tex"), (str(paper.parent), "paper")]
    assert paper.is_file()
    assert promote(client, paper).status_code == 404


@pytest.mark.parametrize(("shown", "promoted"), [("Paper.tex", "Paper"), ("Draft 2", "Draft 2")])
def test_promoting_drops_a_tex_suffix_from_the_display_name_only(client, root, projects, shown, promoted):
    paper = listed_latex(root, projects, "paper", name=shown)

    promote(client, paper)

    assert projects.list_named_files("thesis") == [(str(paper.parent), promoted)]


def test_promoting_refuses_what_is_not_a_listed_project_document_in_its_own_folder(client, root, vault, projects):
    projects.create_project("Other")
    unlisted = root / "latex" / "thesis" / "paper" / "paper.tex"
    unlisted.parent.mkdir(parents=True)
    unlisted.write_text("x", encoding="utf-8")
    mounted = vault / "paper.tex"
    mounted.write_text("x", encoding="utf-8")
    foreign = listed_latex(root, projects, "foreign", slug="other")
    markdown = root / "markdown" / "thesis" / "plan" / "plan.md"
    markdown.parent.mkdir(parents=True)
    markdown.write_text("x", encoding="utf-8")
    projects.add_file("thesis", str(mounted))
    projects.add_file("thesis", str(markdown))
    projects.add_file("thesis", str(foreign))

    assert promote(client, unlisted).status_code == 404
    assert promote(client, root / "latex" / "thesis" / "paper", slug="missing").status_code == 404
    for path in (mounted, markdown, foreign):
        assert promote(client, path).status_code == 400, path
    assert len(projects.list_files("thesis")) == 3


def test_promoting_a_file_whose_folder_is_already_listed_conflicts(client, root, projects):
    paper = listed_latex(root, projects, "paper")
    projects.add_file("thesis", str(paper.parent))

    assert promote(client, paper).status_code == 409
    assert str(paper) in projects.list_files("thesis")


def tex_project(root):
    """The folder `<root>/latex/thesis/paper` with a root, a chapter pointing at it and a bibliography."""
    folder = root / "latex" / "thesis" / "paper"
    (folder / "sections").mkdir(parents=True)
    (folder / "paper.tex").write_text(latex.NEW_DOCUMENT, encoding="utf-8")
    (folder / "sections" / "intro.tex").write_text("% !TEX root = ../paper.tex\nIntro\n", encoding="utf-8")
    (folder / "refs.bib").write_text("@book{a}", encoding="utf-8")
    return folder


def store_links(docs, links):
    docs.write_bytes(WORKBENCH_SESSIONS, json.dumps(links).encode(), expected=None)


def stored_links(docs):
    return json.loads(docs.read_bytes(WORKBENCH_SESSIONS)[0])


def session_of(client, path):
    return client.get("/api/workbench/session", params={"path": str(path)}).json()


def test_a_folder_adopts_the_sessions_of_its_files_merged_with_the_roots_active_one_once(client, root, docs):
    folder = tex_project(root)
    store_links(docs, {
        str(folder / "sections" / "intro.tex"): {"active": "c2", "sessions": ["c1", "c2"], "moved": ["c1"]},
        "/elsewhere.md": "keep",
        str(folder / "refs.bib"): {"active": "b1", "sessions": ["b1"]},
        str(folder / "paper.tex"): {"active": "p1", "sessions": ["c1", "p1", "p2"], "moved": ["p2", "c1"]},
        str(folder / "notes.md"): "markdown stays",
    })

    adopted = session_of(client, folder)

    assert adopted == {"session_id": "p1", "session_ids": ["c1", "c2", "b1", "p1", "p2"], "moved": ["c1", "p2"]}
    assert stored_links(docs) == {
        "/elsewhere.md": "keep",
        str(folder / "notes.md"): "markdown stays",
        str(folder): {"active": "p1", "sessions": ["c1", "c2", "b1", "p1", "p2"], "moved": ["c1", "p2"]},
    }
    assert session_of(client, folder) == adopted


def test_a_folder_without_its_root_among_the_links_takes_the_first_files_active_session(client, root, docs):
    folder = tex_project(root)
    store_links(docs, {
        str(folder / "sections" / "intro.tex"): {"active": "c1", "sessions": ["c1"]},
        str(folder / "refs.bib"): {"active": "b1", "sessions": ["b1"]},
    })

    assert session_of(client, folder) == {"session_id": "c1", "session_ids": ["c1", "b1"], "moved": []}


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


class FixedClock:
    """Stands in for `datetime` in the image route: it is always the same second."""

    @staticmethod
    def now():
        return datetime(2026, 1, 2, 3, 4, 5)


@pytest.fixture
def clock(monkeypatch):
    monkeypatch.setattr(workbench_route, "datetime", FixedClock)
    return "img-20260102-030405.png"


def paste(client, path, content=PNG):
    return client.post("/api/workbench/image", params={"path": str(path)}, files={"file": ("clip.png", content, "image/png")})


def test_a_pasted_image_goes_into_imgs_of_the_project_folder_whatever_file_it_is_pasted_in(client, root, scope, clock):
    folder = tex_project(root)

    stored = paste(client, folder / "sections" / "intro.tex")

    image = folder / "imgs" / clock
    assert stored.json() == {"path": str(image), "reference": rf"\includegraphics[width=\linewidth]{{imgs/{clock}}}", "changed": []}
    assert image.read_bytes() == PNG
    assert scope.changes == [{"kind": "document", "key": str(image), "operation": "write"}]


def test_pasting_into_a_root_without_graphicx_adds_it_and_reports_the_root(client, root, scope, clock):
    folder = tex_project(root)
    source = folder / "paper.tex"
    source.write_text("\\documentclass{article}\n\\begin{document}\n\\end{document}\n", encoding="utf-8")

    stored = paste(client, folder / "sections" / "intro.tex").json()

    image = folder / "imgs" / clock
    assert stored["changed"] == [str(source)]
    assert source.read_text(encoding="utf-8") == "\\documentclass{article}\n\\usepackage{graphicx}\n\\begin{document}\n\\end{document}\n"
    assert scope.changes == [
        {"kind": "document", "key": str(image), "operation": "write"},
        {"kind": "document", "key": str(source), "operation": "write"},
    ]


def test_a_pasted_image_of_a_markdown_file_sits_next_to_it_with_a_markdown_reference(client, root, vault, clock):
    plan = root / "markdown" / "thesis" / "plan" / "plan.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("", encoding="utf-8")

    stored = paste(client, plan)

    assert stored.json() == {"path": str(plan.parent / "imgs" / clock), "reference": f"![](imgs/{clock})", "changed": []}
    assert (plan.parent / "imgs" / clock).read_bytes() == PNG


def test_a_pasted_image_is_never_overwritten(client, root, scope, clock):
    folder = tex_project(root)
    (folder / "imgs").mkdir()
    (folder / "imgs" / clock).write_bytes(b"mine")

    assert paste(client, folder / "paper.tex").status_code == 409
    assert (folder / "imgs" / clock).read_bytes() == b"mine" and scope.changes == []


@pytest.mark.parametrize("content", [b"GIF89a" + b"\x00" * 24, b""])
def test_only_png_images_are_stored(client, root, scope, clock, content):
    folder = tex_project(root)

    assert paste(client, folder / "paper.tex", content).status_code == 400
    assert not (folder / "imgs").exists() and scope.changes == []


def test_an_image_over_the_size_limit_is_refused(client, root, clock, monkeypatch):
    folder = tex_project(root)
    monkeypatch.setattr(workbench_route, "_MAX_IMAGE_BYTES", len(PNG))

    assert paste(client, folder / "paper.tex", PNG + b"\x00").status_code == 413
    assert paste(client, folder / "paper.tex").status_code == 200


def test_images_are_pasted_into_existing_tex_and_markdown_files_inside_the_roots_only(client, root, vault, tmp_path, clock):
    folder = tex_project(root)
    outside = tmp_path / "outside.tex"
    outside.write_text("x", encoding="utf-8")
    pdf = vault / "slides.pdf"
    pdf.write_bytes(b"%PDF-1")

    assert paste(client, folder / "missing.tex").status_code == 404
    assert paste(client, outside).status_code == 403
    assert paste(client, pdf).status_code == 400
