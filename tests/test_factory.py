"""The factory routes pass core actions through a fake pi-host and stop a preview before `tasks.remove`.

- pi-host is an `httpx.MockTransport` speaking `POST /factory/call` and `GET /factory/actions`; it keeps tasks like
  the core, and answers a refusal as the core does (`{error: code, message}`)
"""

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import httpx
import pytest

from backend.identity import get_request_identity
from backend.routes.factory import router
from lib import factory_preview, pi_host
from tests.conftest import request_identity, route_client

TASK_ID = "6f1c2a3b-4d5e-4f60-8a7b-9c0d1e2f3a4b"
ACTIONS = {"tasks.get": {"summary": "One task", "params": {"type": "object", "required": ["id"], "properties": {}}, "mutates": False}}
STATUS = {"invalid": 400, "not_found": 404, "busy": 409, "refused": 409, "failed": 500, "factory_plugin_missing": 503}


def _view(task_id: str = TASK_ID, state: str | None = "done", running: bool = False) -> dict[str, Any]:
    return {"id": task_id, "repo": "/repo", "worktree": f"/worktrees/{task_id}", "branch": "task", "state": state, "running": running}


@dataclass
class FakePiHost:
    """pi-host's factory routes over a tiny core. `calls` keeps `(action, params)` of every `/factory/call`."""

    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
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
    params = {"repo": "/repo", "title": "Rename", "request": "Call it Library", "overrides": {"maxRounds": 1}, "tag": "plugin/m1/w1"}

    response = client.post("/api/factory/call", json={"action": "tasks.start", "params": params})

    assert response.status_code == 200, response.text
    assert response.json() == {"action": "tasks.start", "params": params}
    assert host.calls == [("tasks.start", params)]


def test_a_start_without_a_tag_is_tagged_with_the_requesting_login(client, host):
    params = {"repo": "/repo", "title": "Rename", "request": "Call it Library"}

    response = client.post("/api/factory/call", json={"action": "tasks.start", "params": params})

    assert response.status_code == 200, response.text
    assert host.calls == [("tasks.start", params | {"tag": "me@example.test"})]


def test_a_login_that_is_no_valid_tag_is_tagged_with_its_refused_characters_replaced(host, stopped_previews):
    identity = request_identity()
    client = route_client(router, overrides={get_request_identity: replace(identity, login="o'neil!/x y@example.test")})
    params = {"repo": "/repo", "title": "Rename", "request": "Call it Library"}

    response = client.post("/api/factory/call", json={"action": "tasks.start", "params": params})

    assert response.status_code == 200, response.text
    assert host.calls == [("tasks.start", params | {"tag": "o-neil--x-y@example.test"})]


def test_only_a_start_gets_a_tag(client, host):
    host.tasks[TASK_ID] = _view()

    assert client.post("/api/factory/call", json={"action": "tasks.get", "params": {"id": "6f1c"}}).status_code == 200
    assert client.post("/api/factory/call", json={"action": "tasks.list", "params": {"repo": "/repo"}}).status_code == 200

    assert host.calls == [("tasks.get", {"id": "6f1c"}), ("tasks.list", {"repo": "/repo"})]


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
    assert host.calls[-1] == ("tasks.remove", {"id": "6f1c", "force": True})


def _git(repo: Path, *args: str) -> str:
    import subprocess

    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def _repo_with_merged_task(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("base\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    (repo / "feature.txt").write_text("feature\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "task")
    return repo, _git(repo, "rev-parse", "HEAD")


def _merged_host(host: FakePiHost, repo: Path, commit: str) -> None:
    host.tasks[TASK_ID] = _view() | {"repo": str(repo), "merged": commit}


def test_removing_a_merged_task_reverts_its_commit(client, host, stopped_previews, tmp_path):
    repo, commit = _repo_with_merged_task(tmp_path)
    _merged_host(host, repo, commit)

    response = client.post("/api/factory/call", json={"action": "tasks.remove", "params": {"id": TASK_ID}})

    assert response.status_code == 200, response.text
    assert not (repo / "feature.txt").exists()
    assert (repo / "a.txt").exists()


def test_a_failed_revert_keeps_the_task(client, host, stopped_previews, tmp_path):
    repo, commit = _repo_with_merged_task(tmp_path)
    (repo / "feature.txt").write_text("changed later\n")
    _git(repo, "commit", "-qam", "later")
    _merged_host(host, repo, commit)

    response = client.post("/api/factory/call", json={"action": "tasks.remove", "params": {"id": TASK_ID}})

    assert response.status_code == 409
    assert TASK_ID in host.tasks
    assert (repo / "feature.txt").read_text() == "changed later\n"
    assert _git(repo, "status", "--porcelain") == ""


def test_a_task_that_merged_nothing_is_removed_without_a_revert(client, host, stopped_previews, tmp_path):
    repo, _commit = _repo_with_merged_task(tmp_path)
    host.tasks[TASK_ID] = _view() | {"repo": str(repo), "merged": None, "reason": "no_changes"}

    assert client.post("/api/factory/call", json={"action": "tasks.remove", "params": {"id": TASK_ID}}).status_code == 200
    assert (repo / "feature.txt").exists()


def test_a_task_whose_commit_is_not_on_the_checkout_is_kept(client, host, stopped_previews, tmp_path):
    """A task of a squash wave: its commit stayed on the wave's branch, only the squash commit reached the target."""
    repo, commit = _repo_with_merged_task(tmp_path)
    _git(repo, "reset", "-q", "--hard", "HEAD^")
    (repo / "feature.txt").write_text("feature\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "wave (squash)")
    _merged_host(host, repo, commit)

    response = client.post("/api/factory/call", json={"action": "tasks.remove", "params": {"id": TASK_ID}})

    assert (response.status_code, response.json()["detail"]["code"]) == (409, "refused")
    assert TASK_ID in host.tasks
    assert (repo / "feature.txt").exists()


def test_a_refused_remove_keeps_the_preview_and_passes_the_cores_detail(client, host, stopped_previews):
    host.tasks[TASK_ID] = _view()
    refusal = "task 6f1c2a3b has 2 unmerged commits, remove it with force"
    host.refusals["tasks.remove"] = ("refused", refusal, {"force": True})

    response = client.post("/api/factory/call", json={"action": "tasks.remove", "params": {"id": TASK_ID}})

    assert response.status_code == 409
    assert response.json() == {"detail": {"code": "refused", "message": refusal, "detail": {"force": True}}}
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
    with pytest.raises(pi_host.FactoryPluginMissing, match=r"patrickab/agent-factory"):
        await pi_host.factory_call("tasks.get", {"id": "6f1c"})

    assert await pi_host.factory_actions() == ACTIONS
