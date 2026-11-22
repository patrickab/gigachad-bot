"""The factory routes pass core actions through a fake pi-host, stop a preview before `tasks.remove`, and the old
factory tables move into the core once.

- pi-host is an `httpx.MockTransport` speaking `POST /factory/call` and `GET /factory/actions`; it keeps tasks,
  settings and imported records like the core, and answers a refusal as the core does (`{error: code, message}`)
- The table move needs the test database (`POSTGRES_TEST_DATABASE_URL`), without one those tests skip
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import httpx
import pytest

from backend.identity import get_request_identity
from backend.routes.factory import router
from lib import factory_migration, factory_preview, pi_host
from tests.conftest import request_identity, route_client

TASK_ID = "6f1c2a3b-4d5e-4f60-8a7b-9c0d1e2f3a4b"
ACTIONS = {"tasks.get": {"summary": "One task", "params": {"type": "object", "required": ["id"], "properties": {}}, "mutates": False}}
STATUS = {"invalid": 400, "not_found": 404, "busy": 409, "refused": 409, "failed": 500, "factory_plugin_missing": 503}
SCHEMA_DIRECTORY = Path(__file__).resolve().parents[1] / "db" / "schema"
SCHEMA_STEPS = [SCHEMA_DIRECTORY / name for name in ("0017_factory_tasks.sql", "0020_factory_rows_migrated.sql")]


def _view(task_id: str = TASK_ID, state: str | None = "done", running: bool = False) -> dict[str, Any]:
    return {"id": task_id, "repo": "/repo", "worktree": f"/worktrees/{task_id}", "branch": "task", "state": state, "running": running}


@dataclass
class FakePiHost:
    """pi-host's factory routes over a tiny core. `calls` keeps `(action, params)` of every `/factory/call`."""

    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    imported: dict[str, dict[str, Any]] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    # action -> (code, message) or (code, message, detail) the core answers it with.
    refusals: dict[str, tuple[str, str] | tuple[str, str, dict[str, Any]]] = field(default_factory=dict)
    down: bool = False

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("refused", request=request)
        match (request.method, request.url.path):
            case ("GET", "/factory/actions"):
                return httpx.Response(200, json={"actions": ACTIONS})
            case ("POST", "/factory/call"):
                body = json.loads(request.content)
                self.calls.append((body["action"], body["params"]))
                if refusal := self.refusals.get(body["action"]):
                    code, message, *detail = refusal
                    error = {"error": code, "message": message} | ({"detail": detail[0]} if detail else {})
                    return httpx.Response(STATUS[code], json=error)
                return self._call(body["action"], body["params"])
        return httpx.Response(404, json={"error": "no_route"})

    def _call(self, action: str, params: dict[str, Any]) -> httpx.Response:
        match action:
            case "tasks.get":
                task = self._task(params["id"])
                return httpx.Response(200, json={"result": task}) if task else self._unknown(params["id"])
            case "tasks.remove":
                task = self._task(params["id"])
                if task is None:
                    return self._unknown(params["id"])
                del self.tasks[task["id"]]
                return httpx.Response(200, json={"result": {"removed": task["id"], "dropped": None}})
            case "settings.get":
                stored = self.settings.get(params["repo"], {})
                return httpx.Response(200, json={"result": {"stored": stored, "resolved": {}, "defaults": {}}})
            case "settings.set":
                self.settings[params["repo"]] = params["settings"]
                return httpx.Response(200, json={"result": {"stored": params["settings"], "resolved": {}, "defaults": {}}})
            case "tasks.import":
                record = params["record"]
                self.imported.setdefault(record["id"], record)
                return httpx.Response(200, json={"result": _view(record["id"])})
        return httpx.Response(200, json={"result": {"action": action, "params": params}})

    def _task(self, prefix: str) -> dict[str, Any] | None:
        return next((task for task_id, task in self.tasks.items() if task_id.startswith(prefix)), None)

    @staticmethod
    def _unknown(prefix: str) -> httpx.Response:
        return httpx.Response(404, json={"error": "not_found", "message": f'no task has an id starting with "{prefix}"'})


@pytest.fixture
def host(tmp_path, monkeypatch) -> FakePiHost:
    token = tmp_path / "pi-host.token"
    token.write_text("secret-token\n")
    monkeypatch.setenv("PI_HOST_TOKEN_FILE", str(token))
    fake = FakePiHost()
    monkeypatch.setattr(pi_host, "_transport", httpx.MockTransport(fake.handle))
    return fake


@pytest.fixture
def stopped_previews(monkeypatch) -> list[str]:
    """The ids of the tasks whose preview was stopped, in place of `lib.factory_preview`'s processes."""
    stopped: list[str] = []
    monkeypatch.setattr(factory_preview, "stop_preview", stopped.append)
    return stopped


@pytest.fixture
def client(host, stopped_previews) -> TestClient:
    return route_client(router, overrides={get_request_identity: request_identity()})


def test_a_call_runs_the_core_action_and_answers_its_result_as_it_is(client, host):
    params = {"repo": "/repo", "title": "Rename", "request": "Call it Library", "overrides": {"maxRounds": 1}}

    response = client.post("/api/factory/call", json={"action": "tasks.start", "params": params})

    assert response.status_code == 200, response.text
    assert response.json() == {"action": "tasks.start", "params": params}
    assert host.calls == [("tasks.start", params)]


def test_a_call_without_params_sends_an_empty_object(client, host):
    assert client.post("/api/factory/call", json={"action": "repos.list"}).status_code == 200
    assert host.calls == [("repos.list", {})]


@pytest.mark.parametrize("code", ["invalid", "not_found", "busy", "refused", "failed", "factory_plugin_missing"])
def test_a_core_refusal_answers_its_status_with_code_and_message(client, host, code):
    host.refusals["tasks.approve"] = (code, "task 6f1c2a3b is done, approve needs waiting_approve")

    response = client.post("/api/factory/call", json={"action": "tasks.approve", "params": {"id": "6f1c"}})

    assert response.status_code == STATUS[code]
    assert response.json() == {"detail": {"code": code, "message": "task 6f1c2a3b is done, approve needs waiting_approve"}}


def test_pi_host_down_is_503(client, host):
    host.down = True
    assert client.post("/api/factory/call", json={"action": "repos.list", "params": {}}).status_code == 503
    assert client.get("/api/factory/actions").status_code == 503


def test_actions_are_the_cores(client):
    response = client.get("/api/factory/actions")

    assert (response.status_code, response.json()) == (200, {"actions": ACTIONS})


def test_the_routes_need_host_access(host):
    outsider = route_client(router, overrides={get_request_identity: request_identity(has_host_access=False)})

    assert outsider.post("/api/factory/call", json={"action": "repos.list", "params": {}}).status_code == 403
    assert outsider.get("/api/factory/actions").status_code == 403
    assert host.calls == []


def test_remove_stops_the_removed_tasks_preview_by_its_full_id(client, host, stopped_previews):
    host.tasks[TASK_ID] = _view()

    response = client.post("/api/factory/call", json={"action": "tasks.remove", "params": {"id": "6f1c", "force": True}})

    assert response.status_code == 200, response.text
    assert response.json() == {"removed": TASK_ID, "dropped": None}
    assert stopped_previews == [TASK_ID]
    assert host.calls == [("tasks.remove", {"id": "6f1c", "force": True})]


def test_a_refused_remove_keeps_the_preview_and_passes_the_cores_detail(client, host, stopped_previews):
    host.tasks[TASK_ID] = _view()
    refusal = "task 6f1c2a3b has 2 unmerged commits, remove it with force"
    host.refusals["tasks.remove"] = ("refused", refusal, {"force": True})

    response = client.post("/api/factory/call", json={"action": "tasks.remove", "params": {"id": TASK_ID}})

    assert response.status_code == 409
    assert response.json() == {
        "detail": {"code": "refused", "message": refusal, "detail": {"force": True}}
    }
    assert stopped_previews == []


def test_remove_of_an_unknown_task_is_the_cores_404(client, host, stopped_previews):
    response = client.post("/api/factory/call", json={"action": "tasks.remove", "params": {"id": "abcd"}})

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "not_found"
    assert stopped_previews == []


def test_a_preview_that_does_not_stop_after_the_removal_is_502(client, host, monkeypatch):
    host.tasks[TASK_ID] = _view()

    def stop_preview(_task_id: str) -> None:
        raise factory_preview.PreviewError("The preview database could not be dropped")

    monkeypatch.setattr(factory_preview, "stop_preview", stop_preview)

    response = client.post("/api/factory/call", json={"action": "tasks.remove", "params": {"id": TASK_ID}})

    assert response.status_code == 502
    assert response.json() == {
        "detail": {
            "code": "failed",
            "message": f"Task {TASK_ID} was removed, its preview did not stop: The preview database could not be dropped",
        }
    }
    assert TASK_ID not in host.tasks


async def test_factory_call_errors_are_the_cores_codes(host):
    host.refusals["tasks.get"] = ("busy", "task 6f1c2a3b is busy")
    with pytest.raises(pi_host.FactoryCallError) as refused:
        await pi_host.factory_call("tasks.get", {"id": "6f1c"})
    assert (refused.value.code, refused.value.message, refused.value.detail) == ("busy", "task 6f1c2a3b is busy", None)

    host.refusals["tasks.get"] = ("refused", "task 6f1c2a3b has unmerged work", {"force": True})
    with pytest.raises(pi_host.FactoryCallError) as forceable:
        await pi_host.factory_call("tasks.get", {"id": "6f1c"})
    assert forceable.value.detail == {"force": True}

    host.refusals["tasks.get"] = ("factory_plugin_missing", f"Install it from {pi_host.FACTORY_PLUGIN_SOURCE}")
    with pytest.raises(pi_host.FactoryPluginMissing, match=r"patrickab/\.os"):
        await pi_host.factory_call("tasks.get", {"id": "6f1c"})

    assert await pi_host.factory_actions() == ACTIONS


OLD_DEFAULTS = {"merge": "review", "maxRounds": 3}


def test_old_settings_become_the_cores_stored_settings_over_the_old_defaults():
    stored = {"merge": "auto", "max_rounds": 5, "review": False, "preview": True, "models": {"coder": "a/b"}, "branch": None}
    core = {"merge": "auto", "maxRounds": 5, "review": False, "preview": True, "models": {"coder": "a/b"}}
    legacy = {"source_branch": "feature", "target_branch": "release", "worktree": False}

    assert factory_migration.core_settings(stored) == core
    assert factory_migration.core_settings(legacy) == OLD_DEFAULTS | {"branch": "release"}
    assert factory_migration.core_settings({"branch": "main", "target_branch": "release"}) == OLD_DEFAULTS | {"branch": "main"}
    # An old repo registered with `{}` was review-gated with 3 rounds; the core's own defaults would auto-merge.
    assert factory_migration.core_settings({}) == OLD_DEFAULTS


# The move of the old tables (Postgres).


@pytest.fixture
def old_tables(postgres_pool):
    """The factory tables as `0017` and `0020` made them, empty; afterwards so again."""

    def reset() -> None:
        with postgres_pool.connection() as connection, connection.transaction():
            for table in ("factory_tasks", "factory_repo_settings"):
                connection.execute(f"DROP TABLE IF EXISTS {table}_migrated, {table} CASCADE")
            for step in SCHEMA_STEPS:
                connection.execute(step.read_text())

    reset()
    yield postgres_pool
    reset()


def _insert(pool, settings: dict[str, dict[str, Any]], tasks: list[tuple]) -> None:
    with pool.connection() as connection, connection.transaction():
        for repo, stored in settings.items():
            connection.execute("INSERT INTO factory_repo_settings (repo, settings) VALUES (%s, %s)", (repo, json.dumps(stored)))
        for row in tasks:
            connection.execute(
                "INSERT INTO factory_tasks (id, repo, session_id, session_path, worktree, branch, title, created_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                row,
            )


def _unmarked(pool) -> set[str]:
    """The repos and task ids not marked as moved."""
    with pool.connection() as connection:
        repos = connection.execute("SELECT repo FROM factory_repo_settings WHERE migrated_at IS NULL").fetchall()
        tasks = connection.execute("SELECT id FROM factory_tasks WHERE migrated_at IS NULL").fetchall()
    return {name for (name,) in repos + tasks}


CREATED = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)
HEX_ID = "6f1c2a3b4d5e4f608a7b9c0d1e2f3a4b"
SESSION_FILE = "/pi/factory/sessions/--repo--/2026_s-1.jsonl"
MOVED = (HEX_ID, "/repo", "s-1", SESSION_FILE, "/worktrees/rename", "rename-6f1c2a3b", "Rename", CREATED)
OLD = ("0123456789abcdef0123456789abcdef", "/repo", "s-0", "/pi/sessions/--repo--/s-0.jsonl", None, None, "In the checkout", CREATED)
SECOND_HEX_ID = "7a2b3c4d5e6f40718293a4b5c6d7e8f9"
SECOND_FILE = "/pi/factory/sessions/--repo--/2026_s-2.jsonl"
SECOND = (SECOND_HEX_ID, "/repo", "s-2", SECOND_FILE, "/worktrees/docs", "docs-7a2b3c4d", "Docs", CREATED)


async def test_the_rows_move_into_the_core_once_each(old_tables, host, stopped_previews, caplog):
    _insert(old_tables, {"/repo": {"merge": "auto", "max_rounds": 4, "preview": True, "models": {}, "branch": None}}, [MOVED, OLD])

    with caplog.at_level(logging.WARNING, logger=factory_migration.__name__):
        assert await factory_migration.migrate_factory_rows(old_tables) is True

    assert host.settings == {"/repo": {"merge": "auto", "maxRounds": 4, "preview": True, "models": {}}}
    assert host.imported == {
        TASK_ID: {
            "id": TASK_ID, "repo": "/repo", "worktree": "/worktrees/rename", "branch": "rename-6f1c2a3b", "title": "Rename",
            "sessionId": "s-1", "sessionDir": "/pi/factory/sessions/--repo--", "createdAt": CREATED.isoformat(),
        }
    }
    assert stopped_previews == [HEX_ID]
    assert _unmarked(old_tables) == set()
    assert [record.getMessage() for record in caplog.records] == [
        f"The factory task {OLD[0]} (In the checkout) has no worktree of its own, not moved into the factory core"
    ]

    host.calls.clear()
    assert await factory_migration.migrate_factory_rows(old_tables) is True
    assert host.calls == []


async def test_a_repo_registered_with_empty_settings_keeps_its_review_gate(old_tables, host, stopped_previews):
    _insert(old_tables, {"/repo": {}}, [])

    assert await factory_migration.migrate_factory_rows(old_tables) is True

    assert host.settings == {"/repo": OLD_DEFAULTS}


async def test_settings_the_core_already_stores_stay_and_the_skip_is_logged(old_tables, host, stopped_previews, caplog):
    host.settings["/repo"] = {"merge": "review"}
    _insert(old_tables, {"/repo": {"merge": "auto"}}, [])

    with caplog.at_level(logging.WARNING, logger=factory_migration.__name__):
        assert await factory_migration.migrate_factory_rows(old_tables) is True

    assert host.settings == {"/repo": {"merge": "review"}}
    assert [record.getMessage() for record in caplog.records] == [
        "The factory core already stores settings for /repo, the old ones stay unmoved"
    ]
    assert _unmarked(old_tables) == set()


async def test_a_failing_import_keeps_only_its_row_and_a_moved_task_is_never_imported_again(old_tables, host, stopped_previews):
    _insert(old_tables, {"/repo": {}}, [MOVED, SECOND])

    def fail_second(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        if body.get("action") == "tasks.import" and body["params"]["record"]["sessionId"] == "s-2":
            return httpx.Response(500, json={"error": "failed", "message": "the disk is full"})
        return host.handle(request)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pi_host, "_transport", httpx.MockTransport(fail_second))
        assert await factory_migration.migrate_factory_rows(old_tables) is False
    assert list(host.imported) == [TASK_ID]
    assert _unmarked(old_tables) == {SECOND_HEX_ID}

    # The user removes the moved task in the core before the next start: it does not come back.
    host.imported.clear()
    host.calls.clear()
    assert await factory_migration.migrate_factory_rows(old_tables) is True
    assert [params["record"]["sessionId"] for action, params in host.calls if action == "tasks.import"] == ["s-2"]
    assert _unmarked(old_tables) == set()


async def test_a_repo_the_core_cannot_read_is_skipped_and_pi_host_down_keeps_the_rows(old_tables, host, stopped_previews):
    _insert(old_tables, {"/gone": {"merge": "auto"}}, [])
    host.down = True

    assert await factory_migration.migrate_factory_rows(old_tables) is False
    assert _unmarked(old_tables) == {"/gone"}

    host.down = False
    host.refusals["settings.get"] = ("not_found", "/gone is not inside a git repository")
    assert await factory_migration.migrate_factory_rows(old_tables) is True
    assert host.settings == {}
    assert _unmarked(old_tables) == set()


def test_tables_an_earlier_build_renamed_come_back_with_every_row_marked(old_tables):
    with old_tables.connection() as connection, connection.transaction():
        for table in ("factory_tasks", "factory_repo_settings"):
            connection.execute(f"DROP TABLE {table} CASCADE")
        connection.execute(SCHEMA_STEPS[0].read_text())
    _insert(old_tables, {"/repo": {}}, [MOVED])
    with old_tables.connection() as connection, connection.transaction():
        for table in ("factory_tasks", "factory_repo_settings"):
            connection.execute(f"ALTER TABLE {table} RENAME TO {table}_migrated")
        connection.execute(SCHEMA_STEPS[1].read_text())

    assert _unmarked(old_tables) == set()
