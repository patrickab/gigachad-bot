"""LearnHarness sessions: the adapter's contract with LearnHarness, the host-gated learn routes and the project grouping."""

from collections.abc import Callable
import json

from fastapi.testclient import TestClient
import httpx
import pytest

from backend.identity import get_request_identity
from backend.routes import learn as learn_route
from backend.routes import projects as projects_route
from lib import learnharness
from lib.chat_store import ChatStore
from lib.data_store import StorageConflictError
from lib.project_store import ProjectStore
from lib.upstream import UpstreamContractError, UpstreamRequestFailed, UpstreamUnavailable
from tests.conftest import request_identity, route_client
from tests.fakes import MemoryDataStore

# Trimmed from a live LearnHarness v0.4.0 `GET /api/chats`.
CHATS = {
    "chats": [
        {
            "id": "chat_mvtsqwg2_5e0e6463",
            "title": "Pendulum period vs length",
            "createdAt": "2026-10-28T20:01:02.000Z",
            "updatedAt": "2026-10-28T20:09:12.000Z",
            "kind": "teach",
            "messageCount": 14,
            "phase": "teach",
            "pendingQuestion": True,
            "pendingProblem": False,
            "busy": True,
            "compacting": False,
        },
        {"id": "chat_mvtt377m_61b554a3", "title": "Eigenvalues", "kind": "practice", "busy": False},
    ]
}

# A route's `(status, body)` or `(status, body, headers)`, or a callable answering it when the request arrives.
Answer = tuple[int, object] | tuple[int, object, dict[str, str]]
Route = Answer | Callable[[], Answer]


class LearnHarness:
    """Answers `METHOD /path` from `routes`, records every request, anything else is a connection failure."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, routes: dict[str, Route]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []
        monkeypatch.setattr(learnharness.client, "transport", httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f"{request.method} {request.url.path}"
        if key not in self.routes:
            raise httpx.ConnectError("refused", request=request)
        route = self.routes[key]
        status, body, *headers = route() if callable(route) else route
        content = body if isinstance(body, bytes) else json.dumps(body).encode()
        return httpx.Response(status, content=content, headers=headers[0] if headers else None)

    def bodies(self, key: str) -> list[object]:
        return [json.loads(r.content) for r in self.requests if f"{r.method} {r.url.path}" == key]


# ── Adapter ─────────────────────────────────────────────────────────────────────────────────────────────────────────────


async def test_list_sessions_reads_chat_list_items(monkeypatch):
    LearnHarness(monkeypatch, {"GET /api/chats": (200, CHATS)})

    sessions = [session.model_dump() for session in await learnharness.list_sessions()]

    assert sessions == [
        {
            "id": "chat_mvtsqwg2_5e0e6463",
            "title": "Pendulum period vs length",
            "kind": "teach",
            "busy": True,
            "compacting": False,
            "pending_question": True,
            "pending_problem": False,
            "updated_at": "2026-10-28T20:09:12.000Z",
        },
        {
            "id": "chat_mvtt377m_61b554a3",
            "title": "Eigenvalues",
            "kind": "practice",
            "busy": False,
            "compacting": False,
            "pending_question": False,
            "pending_problem": False,
            "updated_at": None,
        },
    ]


async def test_a_dropped_field_names_the_field(monkeypatch):
    broken = {"chats": [{key: value for key, value in CHATS["chats"][0].items() if key != "title"}]}
    LearnHarness(monkeypatch, {"GET /api/chats": (200, broken)})

    with pytest.raises(UpstreamContractError, match=r"LearnHarness /api/chats: field `chats\.0\.title`"):
        await learnharness.list_sessions()


async def test_create_session_sends_the_title(monkeypatch):
    harness = LearnHarness(monkeypatch, {"POST /api/chats": (200, {"id": "chat_new_1", "title": "Learn Fourier series"})})

    session = await learnharness.create_session("Learn Fourier series")

    assert (session.id, session.title, session.kind, session.busy) == ("chat_new_1", "Learn Fourier series", None, False)
    assert harness.bodies("POST /api/chats") == [{"title": "Learn Fourier series"}]


async def test_submit_text_turn_returns_once_the_turn_is_acknowledged(monkeypatch):
    harness = LearnHarness(monkeypatch, {"POST /api/chat": (200, {"ok": True})})

    await learnharness.submit_text_turn("chat_new_1", "I want to learn Fourier series.")

    assert harness.bodies("POST /api/chat") == [
        {"chatId": "chat_new_1", "input": {"type": "text", "text": "I want to learn Fourier series."}}
    ]


async def test_a_turn_answered_without_the_acknowledgement_names_the_field(monkeypatch):
    LearnHarness(monkeypatch, {"POST /api/chat": (200, {"nope": 1})})

    with pytest.raises(UpstreamContractError, match=r"/api/chat: field `ok`"):
        await learnharness.submit_text_turn("chat_new_1", "goal")


async def test_a_refused_turn_keeps_learnharness_status_and_message(monkeypatch):
    refusal = {"error": "No model configured — set up your models first", "code": "models_not_configured", "missing": ["strong"]}
    LearnHarness(monkeypatch, {"POST /api/chat": (503, refusal)})

    with pytest.raises(UpstreamRequestFailed, match="No model configured") as caught:
        await learnharness.submit_text_turn("chat_new_1", "goal")
    assert caught.value.status == 503


async def test_an_unreachable_learnharness_is_unavailable(monkeypatch):
    LearnHarness(monkeypatch, {})

    with pytest.raises(UpstreamUnavailable):
        await learnharness.submit_text_turn("chat_new_1", "goal")
    with pytest.raises(UpstreamUnavailable):
        await learnharness.list_sessions()


async def test_delete_session_preserves_knowledge_and_counts_unknown_as_deleted(monkeypatch):
    harness = LearnHarness(
        monkeypatch,
        {
            "DELETE /api/chats/chat_a": (200, {"ok": True, "mode": "preserve-graph"}),
            "DELETE /api/chats/chat_gone": (404, {"error": "No such session", "code": "not_found"}),
        },
    )

    await learnharness.delete_session("chat_a")
    await learnharness.delete_session("chat_gone")

    assert [request.url.params["mode"] for request in harness.requests] == ["preserve-graph", "preserve-graph"]


async def test_delete_session_can_remove_the_graph_traces(monkeypatch):
    harness = LearnHarness(monkeypatch, {"DELETE /api/chats/chat_a": (200, {"ok": True, "mode": "delete-nodes"})})

    await learnharness.delete_session("chat_a", "delete-nodes")

    assert [request.url.params["mode"] for request in harness.requests] == ["delete-nodes"]


MARKDOWN = "# Pendulum period vs length\n\n**You:** why does length matter?\n"
MARKDOWN_HEADERS = {
    "content-type": "text/markdown; charset=utf-8",
    "content-disposition": 'attachment; filename="pendulum-period-vs-length.md"',
}


async def test_export_markdown_answers_the_document_and_its_file_name(monkeypatch):
    LearnHarness(monkeypatch, {"GET /api/chats/chat_a/markdown": (200, MARKDOWN.encode(), MARKDOWN_HEADERS)})

    export = await learnharness.export_markdown("chat_a")

    assert export == learnharness.SessionExport(filename="pendulum-period-vs-length.md", markdown=MARKDOWN)


async def test_export_markdown_keeps_refusals_and_names_a_changed_shape(monkeypatch):
    LearnHarness(
        monkeypatch,
        {
            "GET /api/chats/chat_gone/markdown": (404, {"error": "Unknown chat"}),
            "GET /api/chats/chat_page/markdown": (200, b"<!DOCTYPE html>", {"content-type": "text/html"}),
        },
    )

    with pytest.raises(UpstreamRequestFailed, match="Unknown chat"):
        await learnharness.export_markdown("chat_gone")
    with pytest.raises(UpstreamContractError, match="no Markdown attachment"):
        await learnharness.export_markdown("chat_page")
    with pytest.raises(UpstreamUnavailable):
        await learnharness.export_markdown("chat_unrouted")


async def test_delete_session_raises_learnharness_refusal(monkeypatch):
    LearnHarness(monkeypatch, {"DELETE /api/chats/chat_a": (409, {"error": "A turn is running", "code": "turn_running"})})

    with pytest.raises(UpstreamRequestFailed, match="A turn is running"):
        await learnharness.delete_session("chat_a")


async def test_a_404_of_a_moved_route_never_counts_as_deleted(monkeypatch):
    LearnHarness(monkeypatch, {"DELETE /api/chats/chat_a": (404, b"<!DOCTYPE html><html><body>404: This page could not be found")})

    with pytest.raises(UpstreamContractError, match="404 without"):
        await learnharness.delete_session("chat_a")


async def test_rename_session_puts_the_title_and_answers_the_stored_one(monkeypatch):
    harness = LearnHarness(monkeypatch, {"PUT /api/chats/chat_a": (200, {"title": "Fourier series"})})

    assert await learnharness.rename_session("chat_a", "Fourier series") == "Fourier series"
    assert harness.bodies("PUT /api/chats/chat_a") == [{"title": "Fourier series"}]


async def test_renaming_an_unknown_session_keeps_learnharness_refusal(monkeypatch):
    LearnHarness(monkeypatch, {"PUT /api/chats/chat_gone": (404, {"error": "Not found"})})

    with pytest.raises(UpstreamRequestFailed, match="Not found") as caught:
        await learnharness.rename_session("chat_gone", "Fourier series")
    assert caught.value.status == 404


# ── Routes ──────────────────────────────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def store() -> ProjectStore:
    projects = ProjectStore(ChatStore(data_store=MemoryDataStore()), rename=lambda *_: None)
    projects.create_project("Physics")
    projects.create_project("Maths")
    return projects


def _learn_client(store: ProjectStore, *, has_host_access: bool = True) -> TestClient:
    return route_client(
        learn_route.router,
        overrides={get_request_identity: request_identity(has_host_access), learn_route.get_project_store: store},
    )


def test_logins_without_host_access_never_reach_learnharness(monkeypatch, store):
    harness = LearnHarness(monkeypatch, {"GET /api/chats": (200, CHATS)})
    client = _learn_client(store, has_host_access=False)

    assert client.get("/api/learn/sessions").status_code == 403
    assert client.patch("/api/learn/sessions/chat_a", json={"title": "Fourier"}).status_code == 403
    assert client.delete("/api/learn/sessions/chat_a").status_code == 403
    assert client.get("/api/learn/sessions/chat_a/markdown").status_code == 403
    assert client.put("/api/learn/sessions/chat_a/project", json={"project_slug": "physics"}).status_code == 403
    assert harness.requests == []
    assert store.learn_session_projects() == {}


def test_sessions_carry_their_project(monkeypatch, store):
    LearnHarness(monkeypatch, {"GET /api/chats": (200, CHATS)})
    store.assign_learn_session("chat_mvtsqwg2_5e0e6463", "physics")

    response = _learn_client(store).get("/api/learn/sessions")

    assert response.status_code == 200
    assert [(s["id"], s["title"], s["kind"], s["busy"], s["project_slug"]) for s in response.json()] == [
        ("chat_mvtsqwg2_5e0e6463", "Pendulum period vs length", "teach", True, "physics"),
        ("chat_mvtt377m_61b554a3", "Eigenvalues", "practice", False, None),
    ]
    row = response.json()[0]
    assert (row["pending_question"], row["pending_problem"], row["compacting"], row["updated_at"]) == (
        True,
        False,
        False,
        "2026-10-28T20:09:12.000Z",
    )


def test_sessions_learnharness_no_longer_lists_leave_their_project(monkeypatch, store):
    LearnHarness(monkeypatch, {"GET /api/chats": (200, CHATS)})
    store.assign_learn_session("chat_mvtsqwg2_5e0e6463", "physics")
    store.assign_learn_session("chat_deleted_in_learnharness", "physics")
    store.assign_learn_session("chat_also_deleted", "maths")

    assert _learn_client(store).get("/api/learn/sessions").status_code == 200

    assert store.learn_session_projects() == {"chat_mvtsqwg2_5e0e6463": "physics"}


def test_a_failed_session_list_prunes_nothing(monkeypatch, store):
    LearnHarness(monkeypatch, {})
    store.assign_learn_session("chat_a", "physics")

    assert _learn_client(store).get("/api/learn/sessions").status_code == 503
    assert store.learn_session_projects() == {"chat_a": "physics"}


def test_unreachable_learnharness_is_503_and_unknown_shape_is_502_naming_the_field(monkeypatch, store):
    LearnHarness(monkeypatch, {})
    assert _learn_client(store).get("/api/learn/sessions").status_code == 503

    LearnHarness(monkeypatch, {"GET /api/chats": (200, {"sessions": []})})
    response = _learn_client(store).get("/api/learn/sessions")
    assert response.status_code == 502
    assert "`chats`" in response.json()["detail"]


def test_assigning_moves_a_session_between_projects_and_out(store):
    client = _learn_client(store)

    assert client.put("/api/learn/sessions/chat_a/project", json={"project_slug": "physics"}).status_code == 204
    assert client.put("/api/learn/sessions/chat_a/project", json={"project_slug": "maths"}).status_code == 204
    assert store.learn_sessions("physics") == []
    assert store.learn_session_projects() == {"chat_a": "maths"}

    assert client.put("/api/learn/sessions/chat_a/project", json={"project_slug": None}).status_code == 204
    assert store.learn_session_projects() == {}


def test_assigning_to_an_unknown_project_or_a_foreign_id_is_refused(store):
    client = _learn_client(store)

    assert client.put("/api/learn/sessions/chat_a/project", json={"project_slug": "nope"}).status_code == 404
    assert client.put("/api/learn/sessions/chat.a/project", json={"project_slug": "physics"}).status_code == 422
    assert store.learn_session_projects() == {}


def test_renaming_trims_the_title_and_answers_the_stored_one(monkeypatch, store):
    harness = LearnHarness(monkeypatch, {"PUT /api/chats/chat_a": (200, {"title": "Fourier series"})})
    client = _learn_client(store)

    response = client.patch("/api/learn/sessions/chat_a", json={"title": "  Fourier series "})

    assert response.status_code == 200
    assert response.json() == {"title": "Fourier series"}
    assert harness.bodies("PUT /api/chats/chat_a") == [{"title": "Fourier series"}]


def test_a_blank_title_or_a_foreign_id_never_reaches_learnharness(monkeypatch, store):
    harness = LearnHarness(monkeypatch, {})
    client = _learn_client(store)

    assert client.patch("/api/learn/sessions/chat_a", json={"title": "  "}).status_code == 422
    assert client.patch("/api/learn/sessions/chat.a", json={"title": "Fourier"}).status_code == 422
    assert client.delete("/api/learn/sessions/chat.a").status_code == 422
    assert harness.requests == []


def test_renaming_an_unknown_session_is_404(monkeypatch, store):
    LearnHarness(monkeypatch, {"PUT /api/chats/chat_gone": (404, {"error": "Not found"})})

    response = _learn_client(store).patch("/api/learn/sessions/chat_gone", json={"title": "Fourier"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Not found"


def test_deleting_a_session_preserves_knowledge_and_leaves_its_project(monkeypatch, store):
    harness = LearnHarness(monkeypatch, {"DELETE /api/chats/chat_a": (200, {"ok": True})})
    store.assign_learn_session("chat_a", "physics")
    store.assign_learn_session("chat_b", "physics")

    assert _learn_client(store).delete("/api/learn/sessions/chat_a").status_code == 204

    assert [request.url.params["mode"] for request in harness.requests] == ["preserve-graph"]
    assert store.learn_session_projects() == {"chat_b": "physics"}


def test_deleting_a_session_with_its_graph_traces(monkeypatch, store):
    harness = LearnHarness(monkeypatch, {"DELETE /api/chats/chat_a": (200, {"ok": True})})
    client = _learn_client(store)

    assert client.delete("/api/learn/sessions/chat_a", params={"mode": "delete-nodes"}).status_code == 204
    assert client.delete("/api/learn/sessions/chat_a", params={"mode": "everything"}).status_code == 422

    assert [request.url.params["mode"] for request in harness.requests] == ["delete-nodes"]


def test_exporting_a_session_answers_learnharness_markdown_and_file_name(monkeypatch, store):
    LearnHarness(
        monkeypatch,
        {
            "GET /api/chats/chat_a/markdown": (200, MARKDOWN.encode(), MARKDOWN_HEADERS),
            "GET /api/chats/chat_gone/markdown": (404, {"error": "Unknown chat"}),
        },
    )
    client = _learn_client(store)

    response = client.get("/api/learn/sessions/chat_a/markdown")

    assert response.status_code == 200
    assert response.json() == {"filename": "pendulum-period-vs-length.md", "markdown": MARKDOWN}
    missing = client.get("/api/learn/sessions/chat_gone/markdown")
    assert (missing.status_code, missing.json()["detail"]) == (404, "Unknown chat")


def test_a_refused_session_delete_keeps_its_project(monkeypatch, store):
    LearnHarness(monkeypatch, {"DELETE /api/chats/chat_a": (409, {"error": "A turn is running"})})
    store.assign_learn_session("chat_a", "physics")

    response = _learn_client(store).delete("/api/learn/sessions/chat_a")

    assert response.status_code == 409
    assert response.json()["detail"] == "A turn is running"
    assert store.learn_session_projects() == {"chat_a": "physics"}


# ── Project delete ──────────────────────────────────────────────────────────────────────────────────────────────────────


class _Cleanup:
    """Runs the store call without releasing anything."""

    async def after(self, run):
        return run(None)


def _projects_client(store: ProjectStore) -> TestClient:
    return route_client(
        projects_route.router,
        overrides={projects_route.get_project_store: store, projects_route.get_chat_cleanup: _Cleanup()},
    )


def test_deleting_a_project_deletes_its_learn_sessions_first(monkeypatch, store):
    harness = LearnHarness(
        monkeypatch,
        {"DELETE /api/chats/chat_a": (200, {"ok": True}), "DELETE /api/chats/chat_gone": (404, {"error": "No such session"})},
    )
    store.assign_learn_session("chat_a", "physics")
    store.assign_learn_session("chat_gone", "physics")
    store.assign_learn_session("chat_other", "maths")

    assert _projects_client(store).delete("/api/projects/physics").status_code == 200

    assert [request.url.path for request in harness.requests] == ["/api/chats/chat_a", "/api/chats/chat_gone"]
    assert [project["slug"] for project in store.list_projects()] == ["maths"]
    assert store.learn_session_projects() == {"chat_other": "maths"}


@pytest.mark.parametrize(("routes", "status"), [({}, 503), ({"DELETE /api/chats/chat_a": (409, {"error": "A turn is running"})}, 409)])
def test_a_failed_session_delete_keeps_the_project(monkeypatch, store, routes, status):
    LearnHarness(monkeypatch, routes)
    store.assign_learn_session("chat_a", "physics")

    response = _projects_client(store).delete("/api/projects/physics")

    assert response.status_code == status
    assert [project["slug"] for project in store.list_projects()] == ["physics", "maths"]
    assert store.learn_sessions("physics") == ["chat_a"]


def test_deleting_a_project_without_learn_sessions_never_calls_learnharness(monkeypatch, store):
    harness = LearnHarness(monkeypatch, {})

    assert _projects_client(store).delete("/api/projects/physics").status_code == 200
    assert harness.requests == []


def test_a_refusal_midway_keeps_only_the_sessions_that_still_exist(monkeypatch, store):
    LearnHarness(
        monkeypatch,
        {"DELETE /api/chats/chat_a": (200, {"ok": True}), "DELETE /api/chats/chat_b": (409, {"error": "A turn is running"})},
    )
    store.assign_learn_session("chat_a", "physics")
    store.assign_learn_session("chat_b", "physics")

    assert _projects_client(store).delete("/api/projects/physics").status_code == 409
    assert store.learn_sessions("physics") == ["chat_b"]


def test_a_session_assigned_during_the_delete_goes_with_the_project(monkeypatch, store):
    def deleted_while_the_tool_assigns() -> tuple[int, object]:
        store.assign_learn_session("chat_late", "physics")
        return 200, {"ok": True}

    harness = LearnHarness(
        monkeypatch,
        {"DELETE /api/chats/chat_a": deleted_while_the_tool_assigns, "DELETE /api/chats/chat_late": (200, {"ok": True})},
    )
    store.assign_learn_session("chat_a", "physics")

    assert _projects_client(store).delete("/api/projects/physics").status_code == 200
    assert [request.url.path for request in harness.requests] == ["/api/chats/chat_a", "/api/chats/chat_late"]
    assert store.learn_session_projects() == {}


def test_a_moved_delete_route_keeps_the_project(monkeypatch, store):
    LearnHarness(monkeypatch, {"DELETE /api/chats/chat_a": (404, b"<!DOCTYPE html><html>404: This page could not be found")})
    store.assign_learn_session("chat_a", "physics")

    response = _projects_client(store).delete("/api/projects/physics")

    assert response.status_code == 502
    assert [project["slug"] for project in store.list_projects()] == ["physics", "maths"]
    assert store.learn_sessions("physics") == ["chat_a"]


# ── Mapping writes ──────────────────────────────────────────────────────────────────────────────────────────────────────


class _RacedDataStore(MemoryDataStore):
    """Enforces revisions. `race` runs once, right before the next guarded write: another writer of the same file."""

    def __init__(self) -> None:
        super().__init__()
        self.race: Callable[[], None] | None = None

    def write_bytes(self, key, content, *, expected=None, meta=None):
        if expected is not None and (race := self.race) is not None:
            self.race = None
            race()
        if expected is not None and key in self._documents and self.read_bytes(key)[1] != expected:
            raise StorageConflictError(key)
        return super().write_bytes(key, content, expected=expected, meta=meta)


def test_a_concurrent_project_write_and_the_assignment_both_land():
    data = _RacedDataStore()
    projects = ProjectStore(ChatStore(data_store=data), rename=lambda *_: None)
    projects.create_project("Physics")
    data.race = lambda: projects.add_card("physics", "Read chapter 3")

    projects.assign_learn_session("chat_a", "physics")

    assert projects.learn_sessions("physics") == ["chat_a"]
    assert [card["title"] for card in projects.get_project("physics")["kanban"]] == ["Read chapter 3"]


def test_a_project_file_that_keeps_changing_is_a_conflict(monkeypatch):
    data = _RacedDataStore()
    projects = ProjectStore(ChatStore(data_store=data), rename=lambda *_: None)
    projects.create_project("Physics")

    def always_raced(key, content, *, expected=None, meta=None):
        if expected is not None:
            raise StorageConflictError(key)
        return MemoryDataStore.write_bytes(data, key, content, meta=meta)

    monkeypatch.setattr(data, "write_bytes", always_raced)
    with pytest.raises(StorageConflictError):
        projects.assign_learn_session("chat_a", "physics")
