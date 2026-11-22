"""A factory task's preview starts from its workspace's `factory.yaml`, reads `ready` or `failed`, and stops clean.

- The workspace is a throwaway git checkout, the preview process `python -m http.server` or a shell line
- State goes to a scratch `XDG_STATE_HOME`, ports come from a few free ones instead of 8100-8199, and whatever a test
  leaves running is stopped after it
- The database and route tests need the test database (`POSTGRES_TEST_DATABASE_URL`), without one they skip
"""

from datetime import UTC, datetime
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import time
from typing import Any
import urllib.request
from uuid import uuid4

from fastapi.testclient import TestClient
import psycopg
import pytest
import yaml

from backend.identity import get_request_identity
from backend.routes.factory import get_factory_store
from backend.routes.factory_preview import router
from lib import factory, factory_preview
from lib.factory_preview import Preview, PreviewError, PreviewNotConfigured, get_preview, start_preview, stop_preview
from tests.conftest import git, is_running, request_identity, requires_proc, route_client, still_running

pytestmark = requires_proc

SERVER = [sys.executable, "-m", "http.server", "{port.web}", "--bind", "127.0.0.1"]
WEB = {"name": "web", "argv": SERVER, "ready_url": "http://127.0.0.1:{port.web}/"}
SERVED = {"processes": [WEB], "url": "http://127.0.0.1:{port.web}/app?x=1"}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(autouse=True)
def previews(tmp_path, monkeypatch) -> Path:
    """The scratch state root. Previews take ports from four free ones, and none outlives the test."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    first = _free_port()
    monkeypatch.setattr(factory_preview, "PREVIEW_PORTS", range(first, first + 4))
    root = tmp_path / "state" / "gigachad-bot" / "factory-previews"
    yield root
    for directory in root.iterdir() if root.exists() else []:
        if directory.is_dir():
            stop_preview(directory.name)


def _task(tmp_path: Path, preview: dict[str, Any] | None, task_id: str = "task0001") -> factory.TaskRow:
    """A task working in a checkout of its own whose `factory.yaml` has *preview*."""
    checkout = (tmp_path / f"repo-{task_id}").resolve()
    git(tmp_path, "init", "-q", "-b", "main", str(checkout))
    (checkout / "factory.yaml").write_text(yaml.safe_dump({"preview": preview} if preview else {"gates": []}))
    return factory.TaskRow(
        id=task_id, repo=str(checkout), session_id=f"session-{task_id}", session_path=f"/sessions/{task_id}.jsonl",
        worktree=None, branch=None, title="Preview me", created_at=datetime.now(UTC),
    )


def _shell(line: str, **section: Any) -> dict[str, Any]:
    """A preview whose one process is the shell *line*. Nothing answers its `ready_url`."""
    process = {"name": "web", "argv": ["sh", "-c", line], "ready_url": "http://127.0.0.1:{port.web}/"}
    return {"processes": [process], "url": "http://127.0.0.1:{port.web}/", **section}


def _wait_for(task_id: str, state: str, seconds: float = 15.0) -> Preview:
    deadline = time.monotonic() + seconds
    while (preview := get_preview(task_id)) is None or preview.state != state:
        assert time.monotonic() < deadline, f"the preview did not become {state}: {preview}"
        time.sleep(0.05)
    return preview


def _state(task_id: str) -> factory_preview.PreviewState:
    state = factory_preview._read_state(task_id)
    assert state is not None
    return state


def _log(task_id: str) -> str:
    """The output of the preview's process, waited for until it has a line."""
    log = Path(_state(task_id).processes[0].log)
    deadline = time.monotonic() + 5
    while not log.read_text().endswith("\n") and time.monotonic() < deadline:
        time.sleep(0.05)
    return log.read_text()


def test_a_preview_starts_in_the_workspace_becomes_ready_behind_the_gate_and_stops_clean(tmp_path, previews):
    task = _task(tmp_path, SERVED)

    started = start_preview(task)

    state = _state(task.id)
    port, pid = state.ports["web"], state.processes[0].pid
    gate_url = f"{factory_preview.preview_gate_url()}/app?x=1"
    assert started == Preview(state="starting", url=gate_url, error=None)
    assert port in factory_preview.PREVIEW_PORTS
    assert os.getsid(pid) == pid and os.getpgid(pid) == pid
    assert _wait_for(task.id, "ready") == Preview(state="ready", url=gate_url, error=None)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/factory.yaml", timeout=5) as response:
        assert yaml.safe_load(response.read())["preview"]["url"] == SERVED["url"]

    assert start_preview(task).state == "ready"
    assert _state(task.id).processes[0].pid == pid

    stop_preview(task.id)

    assert get_preview(task.id) is None
    assert still_running([pid], 5) == []
    assert factory_preview._is_bindable(port)
    assert not (previews / task.id).exists()


def test_the_processes_get_the_configs_environment_and_none_of_the_backends(tmp_path, monkeypatch):
    monkeypatch.setenv("GIGACHAD_DATABASE_URL", "postgresql://real/app")
    line = 'echo "url=[$GIGACHAD_DATABASE_URL] data=$DATA home=$HOME repo={repo_path}"; exec sleep 30'
    task = _task(tmp_path, _shell(line, env={"DATA": "{data_dir}"}))

    start_preview(task)

    data_dir = Path(_state(task.id).data_dir)
    assert _log(task.id) == f"url=[] data={data_dir} home={os.environ['HOME']} repo={task.repo}\n"
    assert data_dir.is_dir()
    assert get_preview(task.id).state == "starting"


def test_a_process_that_exits_at_once_reads_failed_with_its_log_tail_and_starts_again(tmp_path):
    task = _task(tmp_path, _shell("echo no such module >&2; exit 3"))

    start_preview(task)

    failed = _wait_for(task.id, "failed")
    assert failed.url is None
    assert failed.error == "The preview process web is not running:\nno such module"
    first = _state(task.id).processes[0]

    assert start_preview(task).state == "starting"
    assert (_state(task.id).processes[0].pid, _state(task.id).processes[0].start_ticks) != (first.pid, first.start_ticks)


def test_a_preview_that_never_answers_fails_after_its_timeout(tmp_path):
    task = _task(tmp_path, _shell("echo compiling; exec sleep 30", ready_timeout_s=1))

    assert start_preview(task).state == "starting"

    failed = _wait_for(task.id, "failed")
    assert failed.error == "The preview was not ready after 1 s: web:\ncompiling"
    pid = _state(task.id).processes[0].pid
    assert is_running(pid)

    stop_preview(task.id)

    assert still_running([pid], 5) == []


def test_a_preview_whose_processes_were_killed_reads_failed(tmp_path):
    task = _task(tmp_path, SERVED)
    start_preview(task)
    _wait_for(task.id, "ready")

    os.killpg(_state(task.id).processes[0].pid, signal.SIGKILL)

    assert _wait_for(task.id, "failed").error.startswith("The preview process web is not running")
    stop_preview(task.id)
    assert get_preview(task.id) is None


@pytest.mark.parametrize("stale", [{"boot_id": "another-boot"}, {"start_ticks": 1}])
def test_state_from_another_boot_or_a_reused_pid_reads_failed_and_stop_leaves_the_process_alone(tmp_path, previews, stale):
    task = _task(tmp_path, SERVED)
    start_preview(task)
    _wait_for(task.id, "ready")
    state = _state(task.id)
    pid = state.processes[0].pid
    try:
        state.boot_id = stale.get("boot_id", state.boot_id)
        state.processes[0].start_ticks = stale.get("start_ticks", state.processes[0].start_ticks)
        factory_preview._write_state(state)

        assert get_preview(task.id).state == "failed"

        stop_preview(task.id)

        assert get_preview(task.id) is None
        assert not (previews / task.id).exists()
        assert is_running(pid)
    finally:
        os.killpg(pid, signal.SIGKILL)
        factory_preview._reap()
    assert still_running([pid], 5) == []


def test_a_start_fails_when_the_ports_run_out_and_leaves_nothing(tmp_path, previews, monkeypatch):
    first = _free_port()
    monkeypatch.setattr(factory_preview, "PREVIEW_PORTS", range(first, first + 1))
    holder, late = _task(tmp_path, SERVED, "holder01"), _task(tmp_path, SERVED, "late0001")
    start_preview(holder)
    assert _state(holder.id).ports == {"web": first}

    with pytest.raises(PreviewError, match=f"No free preview port for web in {first}-{first}"):
        start_preview(late)

    assert get_preview(late.id) is None
    assert not (previews / late.id).exists()

    stop_preview(holder.id)
    assert start_preview(late).state == "starting"
    assert _state(late.id).ports == {"web": first}


def test_an_unreadable_state_of_another_task_does_not_block_a_start(tmp_path, previews):
    (previews / "broken01").mkdir(parents=True)
    (previews / "broken01" / "state.json").write_text('{"written by": "an older build"}')

    assert start_preview(_task(tmp_path, SERVED)).state == "starting"
    # `stop_preview` still refuses a state it cannot read: the fixture's teardown must not meet it.
    shutil.rmtree(previews / "broken01")


def test_a_port_something_else_listens_on_is_skipped(tmp_path):
    first = factory_preview.PREVIEW_PORTS.start
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", first))
        listener.listen()

        start_preview(_task(tmp_path, SERVED))

    assert _state("task0001").ports["web"] > first


def test_a_start_that_cannot_run_leaves_nothing(tmp_path, previews):
    with pytest.raises(PreviewNotConfigured, match="has no preview section"):
        start_preview(_task(tmp_path, None, "plain001"))
    with pytest.raises(PreviewError, match=r"\{database_url\}"):
        start_preview(_task(tmp_path, _shell("true", env={"URL": "{database_url}"}), "nodb0001"))
    with pytest.raises(PreviewError, match="is invalid"):
        start_preview(_task(tmp_path, {"processes": [], "url": "http://127.0.0.1:1/"}, "empty001"))
    missing = _task(tmp_path, {**SERVED, "processes": [{**WEB, "argv": ["no-such-command-here"]}]}, "nocmd001")
    with pytest.raises(PreviewError, match=r"The preview did not start: .*no-such-command-here"):
        start_preview(missing)

    assert [entry.name for entry in previews.iterdir()] == ["lock"]


def test_stopping_a_task_without_a_preview_is_fine():
    stop_preview("nothing01")

    assert get_preview("nothing01") is None


def test_a_preview_with_a_database_gets_an_empty_one_named_after_the_task_and_stop_drops_it(tmp_path, monkeypatch, database_url):
    monkeypatch.setenv("GIGACHAD_DATABASE_URL", database_url)
    # Preview databases are server-wide: a suite running beside this one must not get the same name.
    task_id = f"withdb{uuid4().hex[:8]}"
    name = f"gigachad_preview_{task_id}"
    task = _task(tmp_path, _shell('echo "$DATABASE"; exec sleep 30', database=True, env={"DATABASE": "{database_url}"}), task_id)

    def exists() -> bool:
        with psycopg.connect(database_url) as connection:
            return connection.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone() is not None

    start_preview(task)

    assert _state(task.id).database == name
    with psycopg.connect(_log(task.id).strip()) as connection:
        assert connection.execute("SELECT current_database()").fetchone() == (name,)
        assert connection.execute("SELECT count(*) FROM pg_tables WHERE schemaname = 'public'").fetchone() == (0,)

    stop_preview(task.id)

    assert not exists()


@pytest.fixture
def store(postgres_pool) -> factory.FactoryStore:
    with postgres_pool.connection() as connection:
        connection.execute("TRUNCATE factory_tasks, factory_repo_settings")
    return factory.FactoryStore(postgres_pool)


def _stored(store: factory.FactoryStore, task: factory.TaskRow) -> factory.TaskRow:
    store.register_repo(task.repo)
    return store.insert_task(
        task_id=task.id, repo=task.repo, session_id=task.session_id, session_path=task.session_path, worktree=None,
        branch=None, title=task.title,
    )


def _client(store: factory.FactoryStore, has_host_access: bool = True) -> TestClient:
    return route_client(router, overrides={get_request_identity: request_identity(has_host_access), get_factory_store: store})


def test_the_routes_start_show_and_stop_a_tasks_preview(tmp_path, store):
    task = _stored(store, _task(tmp_path, SERVED))
    client = _client(store)
    path = f"/api/factory/tasks/{task.id}/preview"
    gate_url = f"{factory_preview.preview_gate_url()}/app?x=1"

    absent = client.get(path)
    assert (absent.status_code, absent.json()) == (200, None)

    started = client.post(f"{path}/start")
    assert (started.status_code, started.json()) == (200, {"state": "starting", "url": gate_url, "error": None})
    _wait_for(task.id, "ready")
    assert client.get(path).json() == {"state": "ready", "url": gate_url, "error": None}

    stopped = client.post(f"{path}/stop")
    assert (stopped.status_code, stopped.json()) == (200, {"state": "stopped", "url": None, "error": None})
    assert client.get(path).json() is None
    assert client.post(f"{path}/stop").json()["state"] == "stopped"


def test_the_routes_refuse_unknown_tasks_missing_previews_and_logins_without_host_access(tmp_path, store):
    plain = _stored(store, _task(tmp_path, None, "plain001"))
    client = _client(store)

    assert client.get("/api/factory/tasks/unknown01/preview").status_code == 404
    assert client.post("/api/factory/tasks/unknown01/preview/start").status_code == 404
    assert client.post("/api/factory/tasks/unknown01/preview/stop").status_code == 404
    unconfigured = client.post(f"/api/factory/tasks/{plain.id}/preview/start")
    assert unconfigured.status_code == 409
    assert "has no preview section" in unconfigured.json()["detail"]

    outsider = _client(store, has_host_access=False)
    assert outsider.get(f"/api/factory/tasks/{plain.id}/preview").status_code == 403
    assert outsider.post(f"/api/factory/tasks/{plain.id}/preview/start").status_code == 403
    assert outsider.post(f"/api/factory/tasks/{plain.id}/preview/stop").status_code == 403
