"""Workbench chats on Pi: the prompt of each kind, the `/api/workbench/chat*` routes against a fake pi-host and the PDF text."""

import asyncio
from contextlib import suppress
import json
import threading

import httpx
import pytest

from backend.identity import get_request_identity
from backend.routes import workbench as workbench_route
from backend.routes.deps import get_asset_store, get_document_store, get_file_roots
import config
from lib import extract_queue, latex_templates, pi_host
from lib.git import run_git
from lib.storage_namespace import WORKBENCH_CHATS
from lib.workbench import prompts
from lib.workbench.links import SessionLinks
from tests.conftest import mounted_roots, request_identity, route_client
from tests.fakes import MemoryAssetStore, MemoryDataStore, MirroringAssets, StoredAsset

OMP_TOOL_WORDS = ("glob", "`cp`", "shell (", ".omp/", "sed", "perl")


class RevisionedDocs(MemoryDataStore):
    """Refuses a stale `expected` revision like `PostgresDataStore` (None writes unguarded).

    - With *meet*, reads of the links meet in pairs: two concurrent requests both read before either writes (a lone
      reader goes on after a short wait)
    - *interloper* (path, link) is stored once just before the first guarded write of the links, as by another worker
    """

    def __init__(self, *, meet: bool = False, interloper: tuple[str, dict] | None = None) -> None:
        super().__init__()
        self._meeting = threading.Barrier(2, timeout=0.5) if meet else None
        self._interloper = interloper

    def read_bytes(self, key):
        try:
            return super().read_bytes(key)
        finally:
            if key == WORKBENCH_CHATS and self._meeting is not None:
                with suppress(threading.BrokenBarrierError):
                    self._meeting.wait()

    def write_bytes(self, key, content, *, expected=None, meta=None):
        if key == WORKBENCH_CHATS and expected is not None and self._interloper is not None:
            path, link = self._interloper
            self._interloper = None
            stored = json.loads(super().read_bytes(key)[0])
            super().write_bytes(key, json.dumps({**stored, path: link}).encode())
        return super().write_bytes(key, content, expected=expected, meta=meta)


class FakePiHost:
    """pi-host's `POST /sessions` and `GET /sessions`: new sessions get ids `s1`, `s2`, ..., resuming an unknown id answers 404."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.sessions: dict[str, dict] = {}
        # Session id -> the failure pi-host answers when it is resumed.
        self.failures: dict[str, tuple[int, dict]] = {}

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/sessions":
            return httpx.Response(200, json={"sessions": list(self.sessions.values())})
        assert (request.method, request.url.path) == ("POST", "/sessions")
        body = json.loads(request.content)
        self.bodies.append(body)
        resume = body["resume"]
        if isinstance(resume, dict):
            if resume["sessionId"] in self.failures:
                status, failure = self.failures[resume["sessionId"]]
                return httpx.Response(status, json=failure)
            known = self.sessions.get(resume["sessionId"])
            if known is None:
                return httpx.Response(404, json={"error": "not_found"})
            return httpx.Response(200, json={"session": known})
        session_id = f"s{len(self.sessions) + 1}"
        self.sessions[session_id] = _session(session_id, body["cwd"], body["profile"])
        return httpx.Response(200, json={"session": self.sessions[session_id]})


def _session(session_id: str, cwd: str, profile: str) -> dict:
    return {
        "sessionId": session_id, "sessionPath": f"/sessions/{session_id}.jsonl", "cwd": cwd, "machine": "local",
        "createdAt": "2026-11-19T10:00:00.000Z", "modifiedAt": "2026-11-19T10:00:00.000Z", "firstMessage": "",
        "messageCount": 0, "tokens": 0, "live": None, "pinned": False, "archived": False, "profile": profile,
    }


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
def documents(tmp_path, monkeypatch):
    documents = tmp_path / "Documents"
    monkeypatch.setattr(config, "DOCUMENTS", documents)
    return documents


@pytest.fixture
def host(tmp_path, monkeypatch):
    token = tmp_path / "pi-host.token"
    token.write_text("secret\n")
    monkeypatch.setenv("PI_HOST_TOKEN_FILE", str(token))
    host = FakePiHost()
    monkeypatch.setattr(pi_host, "_transport", httpx.MockTransport(host.handle))
    return host


@pytest.fixture
def docs():
    return MemoryDataStore()


@pytest.fixture
def assets():
    return MirroringAssets()


@pytest.fixture
def client(root, vault, docs, assets, documents):
    overrides = {
        get_request_identity: request_identity(),
        get_file_roots: mounted_roots(vault, root),
        get_document_store: docs,
        get_asset_store: assets,
    }
    return route_client(workbench_route.router, overrides=overrides)


def _markdown(root):
    file = root / "markdown" / "thesis" / "notes" / "notes.md"
    file.parent.mkdir(parents=True)
    file.write_text("# Notes\n", encoding="utf-8")
    return file


def _stored(docs, key):
    return json.loads(docs.read_bytes(key)[0])


# Prompts


@pytest.mark.parametrize("kind", prompts.CHAT_KINDS)
def test_every_kind_has_a_pi_prompt_without_omp_tools(kind):
    prompt = prompts.render(kind)
    # The checkout's path is substituted in and may contain a word like `sed` ("unused").
    text = prompt.replace(str(latex_templates.TEMPLATE_SETS), "")
    assert not any(word in text for word in OMP_TOOL_WORDS)
    assert "{{" not in prompt


def test_the_latex_prompt_names_the_template_sets_by_absolute_path(template_sets):
    prompt = prompts.render("latex")
    assert f"`{latex_templates.TEMPLATE_SETS}/X/STYLE.md`" in prompt
    assert f"`{latex_templates.TEMPLATE_SETS}/letter/skeleton.tex`" in prompt
    assert latex_templates.TEMPLATE_SETS.is_absolute()


def test_a_prompt_that_is_a_file_path_is_refused(tmp_path, monkeypatch):
    (tmp_path / "workbench").mkdir()
    (tmp_path / "workbench" / "markdown.md").write_text(str(tmp_path), encoding="utf-8")
    monkeypatch.setattr(config, "TEMPLATES", tmp_path)
    with pytest.raises(ValueError, match="file path"):
        prompts.render("markdown")


# Routes


def test_a_markdown_chat_opens_a_document_session_in_its_kind_directory_and_links_it(client, host, root, docs):
    file = _markdown(root)

    opened = client.post("/api/workbench/chat", json={"path": str(file)}).json()

    assert opened["session"]["sessionId"] == "s1" and opened["session"]["profile"] == "document"
    assert opened["link"] == {"session_id": "s1", "session_ids": ["s1"], "moved": []}
    assert (opened["kind"], opened["cwd"], opened["pdf_text"]) == ("markdown", str(root / "markdown"), None)
    assert host.bodies == [
        {"cwd": str(root / "markdown"), "profile": "document", "resume": None, "systemPrompt": prompts.render("markdown")},
    ]
    assert _stored(docs, WORKBENCH_CHATS) == {str(file): {"active": "s1", "sessions": ["s1"]}}


def test_reopening_resumes_the_active_session_with_the_prompt_again(client, host, root):
    file = _markdown(root)
    client.post("/api/workbench/chat", json={"path": str(file)})

    reopened = client.post("/api/workbench/chat", json={"path": str(file)}).json()

    assert reopened["session"]["sessionId"] == "s1"
    assert host.bodies[1]["resume"] == {"sessionId": "s1"}
    assert host.bodies[1]["systemPrompt"] == prompts.render("markdown")


def test_new_and_switch_keep_every_session_listed(client, host, root, docs):
    file = _markdown(root)
    client.post("/api/workbench/chat", json={"path": str(file)})

    assert client.post("/api/workbench/chat", json={"path": str(file), "new": True}).json()["link"] == {
        "session_id": "s2", "session_ids": ["s1", "s2"], "moved": [],
    }
    switched = client.post("/api/workbench/chat", json={"path": str(file), "session_id": "s1"}).json()
    assert switched["link"] == {"session_id": "s1", "session_ids": ["s1", "s2"], "moved": []}

    listed = client.get("/api/workbench/chats", params={"path": str(file)}).json()
    assert listed["link"]["session_id"] == "s1"
    assert [session["sessionId"] for session in listed["sessions"]] == ["s1", "s2"]


def test_an_active_session_pi_host_lost_is_replaced_but_a_named_one_answers_404(client, host, root, docs):
    file = _markdown(root)
    docs.write_bytes(WORKBENCH_CHATS, json.dumps({str(file): {"active": "gone", "sessions": ["gone"]}}).encode(), expected=None)

    named = client.post("/api/workbench/chat", json={"path": str(file), "session_id": "gone"})
    assert (named.status_code, named.json()) == (404, {"detail": "not_found"})
    opened = client.post("/api/workbench/chat", json={"path": str(file)}).json()

    assert opened["link"] == {"session_id": "s1", "session_ids": ["s1"], "moved": []}


def test_model_and_thinking_choose_a_new_sessions_model_but_a_resumed_one_keeps_its_own(client, host, root):
    file = _markdown(root)
    choice = {"model": {"provider": "anthropic", "id": "claude-haiku-4-5"}, "thinking": "low"}

    client.post("/api/workbench/chat", json={"path": str(file), **choice})
    client.post("/api/workbench/chat", json={"path": str(file), **choice})
    client.post("/api/workbench/chat", json={"path": str(file), "new": True, **choice})

    assert host.bodies[0]["model"] == {"provider": "anthropic", "id": "claude-haiku-4-5"} and host.bodies[0]["thinking"] == "low"
    assert "model" not in host.bodies[1] and "thinking" not in host.bodies[1]
    assert host.bodies[2]["model"]["id"] == "claude-haiku-4-5"


def test_an_active_session_pi_host_cannot_start_stays_listed_beside_a_new_one(client, host, root):
    file = _markdown(root)
    client.post("/api/workbench/chat", json={"path": str(file)})
    # pi-host resumes in the session's stored folder: a moved project folder makes the spawn fail.
    host.failures["s1"] = (502, {"error": "spawn_failed", "message": "ENOENT: no such file or directory, posix_spawn"})

    failed = client.post("/api/workbench/chat", json={"path": str(file), "session_id": "s1"})
    assert (failed.status_code, failed.json()) == (502, {"detail": "spawn_failed"})
    replaced = client.post("/api/workbench/chat", json={"path": str(file)}).json()

    assert replaced["link"] == {"session_id": "s2", "session_ids": ["s1", "s2"], "moved": []}
    host.failures["s2"] = (502, {"error": "other"})
    other = client.post("/api/workbench/chat", json={"path": str(file)})
    assert (other.status_code, other.json()) == (502, {"detail": "other"})


def test_a_latex_file_chats_in_its_project_folder_even_at_a_repository_top(client, host, vault):
    repository = vault / "paper"
    repository.mkdir()
    run_git(repository, "init", "-q")
    tex = repository / "main.tex"
    tex.write_text("\\documentclass{article}\n\\begin{document}\n\\end{document}\n", encoding="utf-8")

    for path in (tex, repository):
        opened = client.post("/api/workbench/chat", json={"path": str(path), "new": True}).json()
        assert (opened["kind"], opened["cwd"]) == ("latex", str(repository))
    assert all(body["cwd"] == str(repository) for body in host.bodies)
    assert not (repository / ".omp").exists()


def test_a_canvas_and_a_notebook_chat_where_they_did(client, host, root, vault, docs):
    key = "project/thesis/document/board.canvas"
    docs.write_bytes(key, b"{}")
    notebook = vault / "lab" / "run.ipynb"
    notebook.parent.mkdir()
    notebook.write_text("{}", encoding="utf-8")

    canvas = client.post("/api/workbench/chat", json={"path": key}).json()
    lab = client.post("/api/workbench/chat", json={"path": str(notebook)}).json()

    assert (canvas["kind"], canvas["cwd"]) == ("canvas", str(root / "canvas"))
    assert (root / "canvas").is_dir()
    assert (lab["kind"], lab["cwd"]) == ("notebook", str(vault))
    assert [body["systemPrompt"] for body in host.bodies] == [
        prompts.render("canvas"), prompts.render("notebook"),
    ]


@pytest.mark.parametrize(("path", "status"), [
    ("relative.md", 400), ("/outside/file.md", 403), ("project/thesis/document/gone.canvas", 404),
])
def test_invalid_targets_are_refused_before_pi_host(client, host, path, status):
    assert client.post("/api/workbench/chat", json={"path": path}).status_code == status
    assert host.bodies == []


def test_a_folder_without_latex_and_a_file_of_another_kind_are_refused(client, host, vault):
    (vault / "data.csv").write_text("a,b\n", encoding="utf-8")
    assert client.post("/api/workbench/chat", json={"path": str(vault)}).status_code == 400
    assert client.post("/api/workbench/chat", json={"path": str(vault / "data.csv")}).status_code == 400
    assert host.bodies == []


def test_a_pdf_chat_names_its_extracted_text_or_says_it_is_pending(client, host, vault, assets, documents, monkeypatch):
    queued = []
    monkeypatch.setattr(extract_queue, "enqueue", lambda name, content, store: queued.append(name))
    pdf = vault / "paper.pdf"
    pdf.write_bytes(b"%PDF-1")

    pending = client.post("/api/workbench/chat", json={"path": str(pdf)}).json()
    assert pending["pdf_text"] == {"state": "pending", "path": None}
    assert queued == ["paper.pdf"]

    assets.assets["Mineru/paper.md"] = StoredAsset("Mineru/paper.md", b"# Paper\n", "mineru_markdown", "text/markdown")
    ready = client.post("/api/workbench/chat", json={"path": str(pdf)}).json()

    text = documents / "Mineru" / "paper.md"
    assert ready["pdf_text"] == {"state": "ready", "path": str(text)}
    assert text.read_text(encoding="utf-8") == "# Paper\n"
    assert ready["kind"] == "pdf"


def test_a_session_told_the_new_path_stops_owing_it(client, host, root, docs):
    file = _markdown(root)
    owed = {str(file): {"active": "s1", "sessions": ["s1"], "moved": ["s1"]}}
    docs.write_bytes(WORKBENCH_CHATS, json.dumps(owed).encode(), expected=None)

    told = client.post("/api/workbench/chat/informed", json={"path": str(file), "session_id": "s1"}).json()

    assert told == {"session_id": "s1", "session_ids": ["s1"], "moved": []}


def test_logins_without_host_access_cannot_open_chats(root, vault, docs, host):
    client = route_client(workbench_route.router, overrides={
        get_request_identity: request_identity(has_host_access=False), get_file_roots: mounted_roots(vault, root),
        get_document_store: docs, get_asset_store: MemoryAssetStore(),
    })
    assert client.post("/api/workbench/chat", json={"path": str(_markdown(root))}).status_code == 403
    assert host.bodies == []


# Concurrent opens


def _async_client(root, vault, docs):
    overrides = {
        get_request_identity: request_identity(), get_file_roots: mounted_roots(vault, root),
        get_document_store: docs, get_asset_store: MirroringAssets(),
    }
    app = route_client(workbench_route.router, overrides=overrides).app
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _seeded(docs, links):
    docs.write_bytes(WORKBENCH_CHATS, json.dumps(links).encode())


@pytest.mark.parametrize("seeded", [False, True], ids=["first-links", "stored-links"])
async def test_concurrent_first_opens_of_two_files_keep_both_links(root, vault, documents, host, seeded):
    docs = RevisionedDocs(meet=True)
    other = {"/elsewhere.md": {"active": "s0", "sessions": ["s0"]}} if seeded else {}
    if seeded:
        _seeded(docs, other)
    first, second = _markdown(root), root / "markdown" / "thesis" / "notes" / "draft.md"
    second.write_text("# Draft\n", encoding="utf-8")

    async with _async_client(root, vault, docs) as client:
        opened = await asyncio.gather(*(client.post("/api/workbench/chat", json={"path": str(path)}) for path in (first, second)))

    assert [response.status_code for response in opened] == [200, 200]
    paths = (str(first), str(second))
    sessions = {path: response.json()["session"]["sessionId"] for path, response in zip(paths, opened, strict=True)}
    assert _stored(docs, WORKBENCH_CHATS) == {
        **other, **{path: {"active": session, "sessions": [session]} for path, session in sessions.items()},
    }


@pytest.mark.parametrize("seeded", [False, True], ids=["first-links", "stored-links"])
async def test_concurrent_first_opens_of_one_file_share_one_session(root, vault, documents, host, seeded):
    docs = RevisionedDocs(meet=True)
    other = {"/elsewhere.md": {"active": "s0", "sessions": ["s0"]}} if seeded else {}
    if seeded:
        _seeded(docs, other)
    file = _markdown(root)

    async with _async_client(root, vault, docs) as client:
        opened = await asyncio.gather(*(client.post("/api/workbench/chat", json={"path": str(file)}) for _ in range(2)))

    assert [response.status_code for response in opened] == [200, 200]
    assert [response.json()["session"]["sessionId"] for response in opened] == ["s1", "s1"]
    assert list(host.sessions) == ["s1"]
    assert _stored(docs, WORKBENCH_CHATS) == {**other, str(file): {"active": "s1", "sessions": ["s1"]}}


def test_a_link_another_worker_stored_meanwhile_is_kept():
    docs = RevisionedDocs(interloper=("/c.md", {"active": "s3", "sessions": ["s3"]}))
    _seeded(docs, {"/b.md": {"active": "s2", "sessions": ["s2"]}})

    SessionLinks(docs).activate("/a.md", "s1")

    assert _stored(docs, WORKBENCH_CHATS) == {
        "/b.md": {"active": "s2", "sessions": ["s2"]},
        "/c.md": {"active": "s3", "sessions": ["s3"]},
        "/a.md": {"active": "s1", "sessions": ["s1"]},
    }
