"""The factory routes start, show and remove tasks on a fake pi-host, a throwaway repo and a stub `agent-worktree`.

- pi-host is an `httpx.MockTransport` that keeps sessions and their `factory-` entries, and takes `/factory start`
  like the extension: a `factory-task` and a `factory-state` entry
- The rows live in the test database (`POSTGRES_TEST_DATABASE_URL`), without one the route tests skip
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient
import httpx
import pytest

from backend.identity import get_request_identity
from backend.routes.factory import get_factory_store, router
from lib import factory, factory_preview, pi_host
from tests.conftest import git, request_identity, route_client
from tests.test_factory_workspace import STUB


@dataclass
class FakePiHost:
    """pi-host as the factory uses it. `requests` keeps `(method, path, body or query)` of every call."""

    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    entries: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    requests: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    # Ids of the sessions created with `factoryRepo`: pi-host's factory root, listed only with `factory=1`.
    factory_root: set[str] = field(default_factory=set)
    # False: `/factory start` is answered `handled` and writes nothing, like the extension refusing it.
    accepts_start: bool = True
    down: bool = False

    def record(self, session_id: str, custom_type: str, data: dict[str, Any]) -> None:
        """Append a custom entry as the extension would, a `factory-state` also becomes the session's `factory`."""
        entries = self.entries.setdefault(session_id, [])
        timestamp = f"2026-11-19T10:00:{len(entries):02d}.000Z"
        entries.append({"type": "custom", "id": f"e{len(entries)}", "customType": custom_type, "data": data, "timestamp": timestamp})
        if custom_type == "factory-state":
            self.sessions[session_id]["factory"] = {**data, "at": timestamp}

    def set_state(self, session_id: str, state: str) -> None:
        self.record(session_id, "factory-state", {"state": state, "round": 1, "reason": ""})

    def sent(self, method: str, suffix: str) -> list[dict[str, Any]]:
        """Bodies (or queries) of the requests with *method* whose path ends in *suffix*."""
        return [body for sent_method, path, body in self.requests if sent_method == method and path.endswith(suffix)]

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("refused", request=request)
        body = json.loads(request.content) if request.content else dict(request.url.params)
        self.requests.append((request.method, request.url.path, body))
        match (request.method, *request.url.path.strip("/").split("/")):
            case ("GET", "sessions"):
                archived = body["archived"] == "1"
                with_factory_root = body.get("factory") == "1" and not archived
                sessions = [
                    session
                    for session_id, session in self.sessions.items()
                    if body.get("cwd") in (None, session["cwd"])
                    and session["archived"] == archived
                    and (with_factory_root or session_id not in self.factory_root)
                ]
                return httpx.Response(200, json={"sessions": sessions})
            case ("GET", "factory", "draft"):
                return httpx.Response(200, json={"draft": "gates: []\nmerge_gates: []\n", "source": ""})
            case ("POST", "sessions"):
                return self._open(body)
            case ("POST", "sessions", session_id, "prompt"):
                return self._prompt(session_id, body["message"])
            case ("GET", "sessions", session_id, "entries"):
                if session_id not in self.entries:
                    return httpx.Response(404, json={"error": "not_found"})
                entries = [entry for entry in self.entries[session_id] if entry["customType"].startswith(body["customType"])]
                return httpx.Response(200, json={"entries": entries, "leafId": self.entries[session_id][-1]["id"]})
            case ("DELETE", "sessions", session_id):
                self.entries.pop(session_id, None)
                known = self.sessions.pop(session_id, None)
                return httpx.Response(204) if known else httpx.Response(404, json={"error": "not_found"})
        return httpx.Response(404, json={"error": "no_route"})

    def _open(self, body: dict[str, Any]) -> httpx.Response:
        if body["resume"] is None:
            session_id = str(uuid4())
            if "factoryRepo" in body:
                self.factory_root.add(session_id)
            root = "/factory/sessions" if "factoryRepo" in body else "/sessions"
            self.sessions[session_id] = {
                "sessionId": session_id,
                "sessionPath": f"{root}/{session_id}.jsonl",
                "cwd": body["cwd"],
                "machine": "local",
                "createdAt": "2026-11-19T10:00:00.000Z",
                "modifiedAt": "2026-11-19T10:00:00.000Z",
                "firstMessage": "",
                "messageCount": 0,
                "tokens": 0,
                "pinned": False,
                "archived": False,
                "profile": body["profile"],
            }
        else:
            session_id = body["resume"]["sessionId"]
            if session_id not in self.sessions:
                return httpx.Response(404, json={"error": "not_found"})
        self.sessions[session_id]["live"] = {"running": False, "waiting": False, "attached": 0}
        return httpx.Response(200, json={"session": self.sessions[session_id]})

    def _prompt(self, session_id: str, message: str) -> httpx.Response:
        session = self.sessions.get(session_id)
        if session is None or session.get("live") is None:
            return httpx.Response(409, json={"error": "not_live"})
        if message.startswith("/factory start ") and self.accepts_start:
            task = json.loads(message.removeprefix("/factory start "))
            cwd = Path(session["cwd"])
            branch = git(cwd, "symbolic-ref", "--short", "HEAD")
            position = {"branch": branch, "mainBranch": "main", "baseSha": git(cwd, "rev-parse", "HEAD")}
            self.record(session_id, "factory-task", {**task, **position})
            self.set_state(session_id, "running")
        return httpx.Response(200, json={"disposition": "handled"})


@pytest.fixture
def host(tmp_path, monkeypatch) -> FakePiHost:
    token = tmp_path / "pi-host.token"
    token.write_text("secret-token\n")
    monkeypatch.setenv("PI_HOST_TOKEN_FILE", str(token))
    fake = FakePiHost()
    monkeypatch.setattr(pi_host, "_transport", httpx.MockTransport(fake.handle))
    # A start that writes no entry is given up on at once.
    monkeypatch.setattr(factory, "START_WAIT_SECONDS", 0.0)
    return fake


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A git repository on `main`, its factory worktrees going to `tmp_path/worktrees` through the stub `agent-worktree`."""
    directory = tmp_path / "bin"
    directory.mkdir()
    script = directory / "agent-worktree"
    script.write_text(STUB)
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{directory}:{os.environ['PATH']}")
    monkeypatch.setenv("GIGACHAD_FACTORY_WORKTREES", str(tmp_path / "worktrees"))
    checkout = (tmp_path / "repo").resolve()
    git(tmp_path, "init", "-q", "-b", "main", str(checkout))
    (checkout / "readme.md").write_text("hi\n")
    git(checkout, "add", "readme.md")
    git(checkout, "commit", "-q", "-m", "readme")
    return checkout


@pytest.fixture
def store(postgres_pool) -> factory.FactoryStore:
    with postgres_pool.connection() as connection:
        connection.execute("TRUNCATE factory_tasks, factory_repo_settings")
    return factory.FactoryStore(postgres_pool)


@pytest.fixture
def stopped_previews(monkeypatch) -> list[str]:
    """The ids of the tasks whose preview was stopped, in place of `lib.factory_preview`'s processes."""
    stopped: list[str] = []
    monkeypatch.setattr(factory_preview, "stop_preview", stopped.append, raising=False)
    return stopped


@pytest.fixture
def client(store, host, stopped_previews) -> TestClient:
    return route_client(router, overrides={get_request_identity: request_identity(), get_factory_store: store})


@pytest.fixture
def registered(client, repo) -> str:
    """The path of the repo, registered."""
    return client.post("/api/factory/repos", json={"path": str(repo)}).json()["path"]


def _start(client: TestClient, repo: str, **overrides: Any) -> dict[str, Any]:
    body = {"repo": repo, "title": "Rename the sidebar!", "request": "Call it Library", "overrides": overrides}
    response = client.post("/api/factory/tasks", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _branches(repo: Path) -> list[str]:
    return git(repo, "branch", "--format=%(refname:short)").splitlines()


def test_settings_chain_is_defaults_then_repo_then_task():
    assert factory.resolve_settings({}) == factory.RepoSettings(
        merge="review", max_rounds=3, review=False, preview=False, models={}, branch=None
    )

    stored = {"merge": "auto", "max_rounds": 5, "models": {"coder": "anthropic/a", "reviewer": "anthropic/b"}}
    overrides = factory.SettingsOverrides(max_rounds=1, branch="release", models={"reviewer": "openai/c"})
    settings = factory.resolve_settings(stored, overrides)

    assert (settings.merge, settings.max_rounds, settings.review, settings.branch) == ("auto", 1, False, "release")
    assert settings.models == {"coder": "anthropic/a", "reviewer": "openai/c"}


def test_settings_saved_with_a_source_and_a_target_branch_keep_the_one_tasks_merged_into():
    assert factory.resolve_settings({"source_branch": "feature", "target_branch": "release"}).branch == "release"
    assert factory.resolve_settings({"source_branch": "feature", "target_branch": None}).branch == "feature"
    assert factory.resolve_settings({"source_branch": "feature", "branch": "main", "worktree": False}).branch == "main"
    assert factory.resolve_settings({"source_branch": None, "target_branch": None}) == factory.RepoSettings()


def test_the_branch_resolves_to_a_local_one_unset_the_default_branch_then_main_then_master(tmp_path):
    repo = (tmp_path / "repo").resolve()
    git(tmp_path, "init", "-q", "-b", "trunk", str(repo))
    git(repo, "commit", "-q", "--allow-empty", "-m", "first")
    with pytest.raises(factory.UnknownBranch, match="the repo has no default branch, main or master, set a branch"):
        factory.resolve_branch(repo, None)
    assert factory.resolve_branch(repo, "trunk") == "trunk"
    with pytest.raises(factory.UnknownBranch, match="the branch release is not a local branch"):
        factory.resolve_branch(repo, "release")

    git(repo, "branch", "master")
    assert factory.resolve_branch(repo, None) == "master"
    git(repo, "branch", "main")
    assert factory.resolve_branch(repo, None) == "main"
    # The repo's default branch (origin's HEAD) wins over both names.
    git(repo, "update-ref", "refs/remotes/origin/trunk", "trunk")
    git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/trunk")
    assert factory.resolve_branch(repo, None) == "trunk"


def test_the_start_command_carries_the_settings_in_the_extensions_shape():
    settings = factory.RepoSettings(merge="auto", max_rounds=2, review=False, models={"coder": "anthropic/a"})
    command = factory.start_command("Title", "Do it", settings, "release")

    assert command.startswith("/factory start ")
    assert json.loads(command.removeprefix("/factory start ")) == {
        "request": "Do it",
        "title": "Title",
        "settings": {
            "merge": "auto", "maxRounds": 2, "review": False, "models": {"coder": "anthropic/a"}, "branch": "release",
        },
    }


def test_a_worktree_is_named_after_the_title_and_the_short_id():
    assert factory.workspace_name("Rename the Sidebar!", "0123456789abcdef") == "rename-the-sidebar-01234567"
    assert factory.workspace_name("???", "0123456789abcdef") == "task-01234567"


def test_a_workspace_diff_holds_changed_and_untracked_files_and_leaves_the_index(repo):
    (repo / "readme.md").write_text("hello\n")
    (repo / "notes dir").mkdir()
    (repo / "notes dir" / "new.txt").write_text("fresh\n")

    diff = factory.workspace_diff(repo, git(repo, "rev-parse", "HEAD"))

    assert "diff --git a/readme.md b/readme.md" in diff
    assert "-hi\n+hello\n" in diff
    assert "--- /dev/null\n+++ b/notes dir/new.txt" in diff
    assert "+fresh\n" in diff
    assert "\x1b[" not in diff
    assert git(repo, "diff", "--cached", "--name-only") == ""
    assert git(repo, "ls-files", "--others", "--exclude-standard") == "notes dir/new.txt"


def test_repos_are_registered_once_by_the_path_of_a_primary_checkout(client, repo, tmp_path):
    info = {"name": "repo", "path": str(repo)}

    assert client.get("/api/factory/repos").json() == []
    assert client.post("/api/factory/repos", json={"path": str(repo)}).json() == info
    assert client.post("/api/factory/repos", json={"path": f"{repo}/"}).json() == info
    assert client.get("/api/factory/repos").json() == [info]

    linked = tmp_path / "linked"
    git(repo, "worktree", "add", "-q", "-b", "side", str(linked))
    (repo / "sub").mkdir()
    for path in (str(linked), str(repo / "sub"), str(tmp_path), str(tmp_path / "absent"), "relative"):
        refused = client.post("/api/factory/repos", json={"path": path})
        assert refused.status_code == 400, path
    assert client.get("/api/factory/repos").json() == [info]


def test_settings_of_a_repo_are_the_defaults_until_saved(client, registered):
    defaults = {
        "merge": "review", "max_rounds": 3, "review": False, "preview": False, "models": {}, "branch": None,
    }
    saved = {**defaults, "merge": "auto", "max_rounds": 7, "models": {"reviewer": "anthropic/b"}}

    assert client.get("/api/factory/settings", params={"repo": registered}).json() == defaults
    assert client.put("/api/factory/settings", params={"repo": registered}, json=saved).json() == saved
    assert client.get("/api/factory/settings", params={"repo": registered}).json() == saved

    assert client.put("/api/factory/settings", params={"repo": registered}, json={**saved, "max_rounds": 21}).status_code == 422
    assert client.put("/api/factory/settings", params={"repo": registered}, json={**saved, "branch": "a..b"}).status_code == 422
    with_branch = client.put("/api/factory/settings", params={"repo": registered}, json={**saved, "branch": "release"})
    assert with_branch.json()["branch"] == "release"
    assert client.get("/api/factory/settings", params={"repo": "/absent"}).status_code == 404
    assert client.put("/api/factory/settings", params={"repo": "/absent"}, json=saved).status_code == 404


def test_the_branches_of_a_repo_are_its_local_branches_alphabetically(client, registered, repo):
    for name in ("release", "Feature/b", "feature/a"):
        git(repo, "branch", name)

    assert client.get("/api/factory/branches", params={"repo": registered}).json() == ["feature/a", "Feature/b", "main", "release"]
    assert client.get("/api/factory/branches", params={"repo": "/absent"}).status_code == 404


def test_a_task_starts_in_a_worktree_with_a_coding_session_and_the_start_command(client, host, registered, repo, tmp_path):
    client.put("/api/factory/settings", params={"repo": registered}, json={"merge": "auto", "models": {"coder": "anthropic/a"}})

    task = _start(client, registered, max_rounds=1)

    worktree = tmp_path / "worktrees" / f"rename-the-sidebar-{task['id'][:8]}"
    assert worktree.is_dir()
    assert task["worktree"] == str(worktree)
    assert task["branch"] == worktree.name
    assert (task["repo"], task["title"]) == (registered, "Rename the sidebar!")
    assert task["factory"]["state"] == "running"
    assert task["live"] == {"running": False, "waiting": False, "attached": 0}
    assert host.sent("POST", "/sessions") == [
        {"cwd": str(worktree), "profile": "coding", "resume": None, "factoryRepo": registered}
    ]
    session = host.sessions[task["session_id"]]
    assert task["session_path"] == session["sessionPath"] == f"/factory/sessions/{task['session_id']}.jsonl"
    (prompt,) = host.sent("POST", "/prompt")
    assert json.loads(prompt["message"].removeprefix("/factory start ")) == {
        "request": "Call it Library",
        "title": "Rename the sidebar!",
        "settings": {
            "merge": "auto", "maxRounds": 1, "review": False, "models": {"coder": "anthropic/a"}, "branch": "main",
        },
    }


def test_a_task_starts_at_its_branchs_tip_whatever_the_primary_has_checked_out(client, host, registered, repo):
    git(repo, "checkout", "-q", "-b", "release")
    (repo / "readme.md").write_text("release\n")
    git(repo, "commit", "-q", "-am", "release")
    release_tip = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "-b", "side")
    (repo / "readme.md").write_text("dirty\n")

    task = _start(client, registered, branch="release")

    assert git(Path(task["worktree"]), "rev-parse", "HEAD") == release_tip
    assert (Path(task["worktree"]) / "readme.md").read_text() == "release\n"
    (prompt,) = host.sent("POST", "/prompt")
    assert json.loads(prompt["message"].removeprefix("/factory start "))["settings"]["branch"] == "release"


def test_a_task_on_an_unknown_branch_is_refused_before_anything_exists(client, host, registered, repo, tmp_path):
    body = {"repo": registered, "title": "T", "request": "R", "overrides": {"branch": "absent"}}
    refused = client.post("/api/factory/tasks", json=body)

    assert refused.status_code == 400
    assert "absent is not a local branch" in refused.text
    assert host.sent("POST", "/sessions") == []
    assert not (tmp_path / "worktrees").exists()
    assert client.post("/api/factory/tasks", json={**body, "overrides": {"branch": "a..b"}}).status_code == 422


def test_tasks_are_listed_with_their_state_from_one_pi_host_listing(client, host, registered):
    first = _start(client, registered)
    second = _start(client, registered)
    host.set_state(first["session_id"], "waiting_approve")
    host.sessions[first["session_id"]]["live"] = None
    host.requests.clear()

    tasks = client.get("/api/factory/tasks", params={"repo": registered}).json()

    assert [task["id"] for task in tasks] == [second["id"], first["id"]]
    assert tasks[1]["factory"] == {"state": "waiting_approve", "round": 1, "reason": "", "at": "2026-11-19T10:00:02.000Z"}
    assert tasks[1]["live"] is None
    assert host.requests == [("GET", "/sessions", {"archived": "0", "factory": "1"})]

    # A session pi-host lost takes the archive listing too, and shows no state.
    host.sessions.pop(second["session_id"])
    host.requests.clear()
    tasks = client.get("/api/factory/tasks", params={"repo": registered}).json()
    assert (tasks[0]["factory"], tasks[0]["live"]) == (None, None)
    assert host.requests == [
        ("GET", "/sessions", {"archived": "0", "factory": "1"}),
        ("GET", "/sessions", {"archived": "1"}),
    ]

    one = client.get(f"/api/factory/tasks/{first['id']}").json()
    assert one == tasks[1]
    assert client.get("/api/factory/tasks", params={"repo": "/absent"}).status_code == 404
    assert client.get("/api/factory/tasks/absent000").status_code == 404


def test_tasks_find_their_sessions_in_the_factory_root_and_older_ones_in_pis_normal_root(client, host, registered):
    older = _start(client, registered)
    newer = _start(client, registered)
    # A task started before the factory root existed: its session file stays in Pi's normal root.
    host.factory_root.discard(older["session_id"])
    host.set_state(older["session_id"], "waiting_approve")

    tasks = {task["id"]: task for task in client.get("/api/factory/tasks", params={"repo": registered}).json()}

    assert tasks[older["id"]]["factory"]["state"] == "waiting_approve"
    assert tasks[newer["id"]]["factory"]["state"] == "running"
    assert client.get(f"/api/factory/tasks/{newer['id']}").json()["factory"]["state"] == "running"
    assert client.get(f"/api/factory/tasks/{older['id']}").json()["factory"]["state"] == "waiting_approve"


def test_a_start_the_session_does_not_take_fails_and_leaves_nothing(client, host, registered, repo, tmp_path):
    host.accepts_start = False

    response = client.post("/api/factory/tasks", json={"repo": registered, "title": "Rename", "request": "Do it"})

    assert response.status_code == 502
    assert "no factory-task entry" in response.json()["detail"]
    assert host.sessions == {}
    assert len(host.sent("DELETE", "")) == 1
    assert _branches(repo) == ["main"]
    assert list((tmp_path / "worktrees").iterdir()) == []
    assert client.get("/api/factory/tasks", params={"repo": registered}).json() == []


def test_a_start_without_pi_host_answers_503_and_removes_the_worktree(client, host, registered, repo, tmp_path):
    host.down = True

    response = client.post("/api/factory/tasks", json={"repo": registered, "title": "Rename", "request": "Do it"})

    assert response.status_code == 503
    assert _branches(repo) == ["main"]
    assert list((tmp_path / "worktrees").iterdir()) == []


def test_a_start_fails_for_an_unknown_repo_and_a_failing_workspace(client, host, registered, repo):
    unknown = client.post("/api/factory/tasks", json={"repo": "/absent", "title": "Rename", "request": "Do it"})
    assert unknown.status_code == 404

    (repo / "factory.yaml").write_text("hooks:\n  after_create: [[sh, -c, 'echo no network >&2; exit 3']]\n")
    failed = client.post("/api/factory/tasks", json={"repo": registered, "title": "Rename", "request": "Do it"})

    assert failed.status_code == 502
    assert "failed with exit code 3" in failed.json()["detail"]
    assert "no network" in failed.json()["detail"]
    assert host.requests == []
    assert _branches(repo) == ["main"]


def test_the_diff_of_a_task_is_measured_from_the_branch_it_merges_into(client, host, registered, repo):
    task = _start(client, registered)
    worktree = Path(task["worktree"])
    (worktree / "readme.md").write_text("hello\n")
    git(worktree, "commit", "-q", "-am", "round 1")
    (worktree / "readme.md").write_text("hello there\n")
    (worktree / "new.txt").write_text("fresh\n")
    host.requests.clear()

    response = client.get(f"/api/factory/tasks/{task['id']}/diff")

    assert response.status_code == 200
    assert response.json()["base"] == git(repo, "rev-parse", "main")
    diff = response.json()["diff"]
    assert "diff --git a/readme.md b/readme.md" in diff
    assert "-hi\n+hello there\n" in diff
    assert "--- /dev/null\n+++ b/new.txt" in diff
    assert host.sent("GET", "/entries") == [{"customType": "factory-"}]
    assert git(worktree, "diff", "--cached", "--name-only") == ""
    assert git(worktree, "ls-files", "--others", "--exclude-standard") == "new.txt"


def test_commands_open_the_session_and_prompt_it(client, host, registered):
    task = _start(client, registered)
    session_id = task["session_id"]
    host.sessions[session_id]["live"] = None
    host.requests.clear()

    for command in ("approve", "retry", "cancel"):
        answer = client.post(f"/api/factory/tasks/{task['id']}/{command}")
        assert answer.status_code == 200
        assert (answer.json()["id"], answer.json()["factory"]["state"]) == (task["id"], "running")

    opened = {"cwd": task["worktree"], "profile": "coding", "resume": {"sessionId": session_id}}
    assert host.sent("POST", "/sessions") == [opened, opened, opened]
    assert host.sent("POST", f"/sessions/{session_id}/prompt") == [
        {"message": "/factory approve"},
        {"message": "/factory retry"},
        {"message": "/factory cancel"},
    ]
    assert client.post(f"/api/factory/tasks/{task['id']}/start").status_code == 422
    assert client.post("/api/factory/tasks/absent000/approve").status_code == 404


def test_opening_the_session_makes_it_live_without_a_prompt(client, host, registered):
    task = _start(client, registered)
    host.requests.clear()

    answer = client.post(f"/api/factory/tasks/{task['id']}/session")

    assert answer.status_code == 200
    assert answer.json()["id"] == task["id"]
    assert host.sent("POST", "/sessions") == [
        {"cwd": task["worktree"], "profile": "coding", "resume": {"sessionId": task["session_id"]}}
    ]
    assert host.sent("POST", "/prompt") == []
    assert client.post("/api/factory/tasks/absent000/session").status_code == 404


def test_a_message_to_a_merging_task_is_refused(client, host, registered):
    task = _start(client, registered)
    host.set_state(task["session_id"], "merging")
    host.requests.clear()

    refused = client.post(f"/api/factory/tasks/{task['id']}/messages", json={"text": "One more thing"})

    assert refused.status_code == 409
    assert host.sent("POST", "/prompt") == []


def test_a_message_is_queued_behind_a_running_turn(client, host, registered):
    task = _start(client, registered)
    host.requests.clear()

    answer = client.post(f"/api/factory/tasks/{task['id']}/messages", json={"text": "Also rename the icon"})

    assert answer.status_code == 200
    assert answer.json()["id"] == task["id"]
    assert host.sent("POST", "/prompt") == [{"message": "Also rename the icon", "streamingBehavior": "followUp"}]
    assert client.post(f"/api/factory/tasks/{task['id']}/messages", json={"text": ""}).status_code == 422


def test_an_archived_task_session_keeps_its_state(client, host, registered):
    task = _start(client, registered)
    # Only an older task's session, in Pi's normal root, can be archived: the factory root never is.
    host.factory_root.discard(task["session_id"])
    host.sessions[task["session_id"]]["archived"] = True

    listed = client.get("/api/factory/tasks", params={"repo": registered}).json()
    assert listed[0]["factory"]["state"] == "running"
    assert client.get(f"/api/factory/tasks/{task['id']}").json()["factory"]["state"] == "running"
    assert client.delete(f"/api/factory/tasks/{task['id']}").status_code == 409


def test_a_session_pi_host_lost_passes_its_refusal_on(client, host, registered):
    task = _start(client, registered)
    host.sessions.pop(task["session_id"])

    refused = client.post(f"/api/factory/tasks/{task['id']}/approve")

    assert (refused.status_code, refused.json()) == (404, {"detail": "not_found"})


def test_delete_refuses_a_task_at_work_and_else_removes_everything(
    client, host, registered, repo, scratch_pi_agent_dir, stopped_previews, monkeypatch
):
    task = _start(client, registered)
    session_id = task["session_id"]
    reviews = scratch_pi_agent_dir / "factory-reviews" / session_id
    reviews.mkdir(parents=True)
    (reviews / "review-1.jsonl").write_text("{}\n")

    for state in ("running", "merging"):
        host.set_state(session_id, state)
        refused = client.delete(f"/api/factory/tasks/{task['id']}")
        assert refused.status_code == 409
        assert state in refused.json()["detail"]
    assert Path(task["worktree"]).is_dir()
    assert session_id in host.sessions
    assert client.get(f"/api/factory/tasks/{task['id']}").status_code == 200
    assert stopped_previews == []

    # The preview serves the worktree: it stops while the worktree is still there.
    worktree_at_stop: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        factory_preview, "stop_preview", lambda task_id: worktree_at_stop.append((task_id, Path(task["worktree"]).is_dir()))
    )
    host.set_state(session_id, "blocked")
    assert client.delete(f"/api/factory/tasks/{task['id']}").status_code == 204

    assert worktree_at_stop == [(task["id"], True)]
    assert session_id not in host.sessions
    assert not reviews.exists()
    assert not Path(task["worktree"]).exists()
    assert _branches(repo) == ["main"]
    assert client.get(f"/api/factory/tasks/{task['id']}").status_code == 404
    assert client.delete(f"/api/factory/tasks/{task['id']}").status_code == 404


def test_delete_removes_an_interrupted_task(client, host, registered):
    task = _start(client, registered)
    for state in ("running", "merging"):
        host.set_state(task["session_id"], state)
    host.sessions[task["session_id"]]["live"] = None

    assert client.delete(f"/api/factory/tasks/{task['id']}").status_code == 204

    assert not Path(task["worktree"]).exists()
    assert task["session_id"] not in host.sessions


def test_delete_answers_502_and_keeps_everything_when_the_preview_does_not_stop(client, host, registered, monkeypatch):
    task = _start(client, registered)
    host.set_state(task["session_id"], "blocked")

    def stop_preview(task_id: str) -> None:
        raise factory_preview.PreviewError("The preview database could not be dropped")

    monkeypatch.setattr(factory_preview, "stop_preview", stop_preview)

    refused = client.delete(f"/api/factory/tasks/{task['id']}")

    assert (refused.status_code, refused.json()) == (502, {"detail": "The preview database could not be dropped"})
    assert Path(task["worktree"]).is_dir()
    assert task["session_id"] in host.sessions
    assert host.sent("DELETE", task["session_id"]) == []
    assert client.get(f"/api/factory/tasks/{task['id']}").status_code == 200


def test_logins_without_host_access_are_refused(store, host):
    client = route_client(router, overrides={get_request_identity: request_identity(False), get_factory_store: store})

    assert client.get("/api/factory/repos").status_code == 403
    assert client.post("/api/factory/tasks", json={"repo": "/r", "title": "t", "request": "r"}).status_code == 403


def test_config_is_a_draft_until_written_once_and_never_overwritten(client, host, registered, repo):
    draft = client.get("/api/factory/config", params={"repo": registered}).json()
    assert draft["exists"] is False
    assert draft["draft"] == "gates: []\nmerge_gates: []\n"
    assert host.sent("GET", "/factory/draft") == [{"cwd": registered}]

    content = "gates:\n  - [make, lint]\n"
    saved = client.put("/api/factory/config", params={"repo": registered}, json={"content": content})
    assert saved.json() == {"exists": True, "draft": None}
    assert (repo / "factory.yaml").read_text() == content
    assert client.get("/api/factory/config", params={"repo": registered}).json() == {"exists": True, "draft": None}

    again = client.put("/api/factory/config", params={"repo": registered}, json={"content": "gates: []\n"})
    assert again.status_code == 409
    assert (repo / "factory.yaml").read_text() == content


@pytest.mark.parametrize("content", ["gates: [", "- a\n", "gates: [bash]\n", "merge_gates:\n  - [1]\n"])
def test_config_that_is_no_factory_yaml_is_refused(client, registered, repo, content):
    refused = client.put("/api/factory/config", params={"repo": registered}, json={"content": content})
    assert refused.status_code == 400
    assert not (repo / "factory.yaml").exists()


def test_config_of_an_unknown_repo_is_404(client, tmp_path):
    assert client.get("/api/factory/config", params={"repo": str(tmp_path)}).status_code == 404
    assert client.put("/api/factory/config", params={"repo": str(tmp_path)}, json={"content": ""}).status_code == 404
