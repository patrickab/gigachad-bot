"""Host processes never outlive the backend that started them: real processes, real kills.

The Jupyter Server and its kernels are covered in `test_jupyter.py`.
"""

import json
import os
import shutil
import signal
import subprocess
import sys

import pytest

from lib.process_owner import OWNER_VARIABLE, owned_environment, reap_orphans
from tests.conftest import SRC, is_running, requires_proc, still_running

pytestmark = requires_proc

_NO_MARK = {name: value for name, value in os.environ.items() if name != OWNER_VARIABLE}


@pytest.fixture
def sleepers():
    """`sleep(env)` starts a `sleep` child of the test with *env*, every one is killed at teardown."""
    started: list[subprocess.Popen] = []

    def sleep(env: dict[str, str]) -> int:
        started.append(subprocess.Popen(["sleep", "600"], env=env))  # noqa: S607
        return started[-1].pid

    yield sleep
    for process in started:
        process.kill()
        process.wait()


def test_what_a_killed_backend_left_behind_is_reaped_with_its_descendants_and_nothing_else(backend_process, sleepers):
    backend, started = backend_process("""
        import json, os, subprocess, time
        from lib.process_owner import owned_environment
        marked = subprocess.Popen(["sh", "-c", "sleep 600 & echo $!; wait"], env=owned_environment(os.environ), stdout=subprocess.PIPE)
        unmarked = subprocess.Popen(["sleep", "600"])
        print(json.dumps({"child": marked.pid, "grandchild": int(marked.stdout.readline()), "unmarked": unmarked.pid}), flush=True)
        time.sleep(600)
    """)
    owned = sleepers(owned_environment(_NO_MARK))
    backend.kill()
    backend.wait()

    reaped = reap_orphans()

    assert {started["child"], started["grandchild"]} <= set(reaped)
    assert still_running([started["child"], started["grandchild"]], 5) == []
    assert started["unmarked"] not in reaped
    assert owned not in reaped
    assert is_running(started["unmarked"])
    assert is_running(owned)
    os.kill(started["unmarked"], signal.SIGKILL)


def test_a_mark_naming_a_pid_that_now_runs_another_process_counts_as_ended(sleepers):
    reused = sleepers({**_NO_MARK, OWNER_VARIABLE: f"{os.getpid()}:1"})  # this pid, another start time
    unreadable = sleepers({**_NO_MARK, OWNER_VARIABLE: "not a process"})

    reaped = reap_orphans()

    assert reused in reaped
    assert unreadable not in reaped
    assert is_running(unreadable)


def test_the_reaper_spares_itself_and_its_ancestors_whatever_their_mark():
    gone = {**_NO_MARK, OWNER_VARIABLE: "999999999:1"}  # no pid that high: an owner that ended
    script = "import json, os; from lib import process_owner as p; print(json.dumps([os.getpid(), os.getppid(), p.reap_orphans()]))"

    shell = subprocess.run(  # noqa: S603
        ["sh", "-c", f'{sys.executable} -c "$0"; true', script], cwd=SRC, env=gone, capture_output=True, text=True, check=True
    )

    reaper, parent, reaped = json.loads(shell.stdout)
    assert reaper not in reaped
    assert parent not in reaped


def test_stop_owned_ends_what_is_left_of_the_backends_own_children_only(backend_process):
    _, started = backend_process("""
        import json, os, subprocess, time
        from lib.process_owner import owned_environment, stop_owned
        marked = subprocess.Popen(["sleep", "600"], env=owned_environment(os.environ))
        unmarked = subprocess.Popen(["sleep", "600"])
        stopped = stop_owned()
        marked.wait()
        print(json.dumps({"marked": marked.pid, "unmarked": unmarked.pid, "stopped": stopped}), flush=True)
        time.sleep(600)
    """)

    assert started["stopped"] == [started["marked"]]
    assert is_running(started["unmarked"])
    os.kill(started["unmarked"], signal.SIGKILL)


@pytest.mark.skipif(shutil.which("nvim") is None, reason="nvim is not installed")
def test_neovim_ends_when_the_backend_is_killed(backend_process, tmp_path, monkeypatch):
    for variable in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
        monkeypatch.setenv(variable, str(tmp_path / variable.lower()))  # an empty config, never the developer's
    (tmp_path / "xdg_config_home" / "nvim").mkdir(parents=True)
    backend, started = backend_process(f"""
        import asyncio, json
        from pathlib import Path
        from lib.neovim_session import NeovimSession

        async def main():
            session = NeovimSession.spawn("a.md", "# a\\n", rows=24, cols=80, cwd=Path({str(tmp_path)!r}))

            async def save(path, text):
                return text

            await session.connect(save, lambda text: None)
            terminal = session._process.pid
            tasks = Path(f"/proc/{{terminal}}/task").iterdir()
            children = [int(pid) for task in tasks for pid in (task / "children").read_text().split()]
            print(json.dumps({{"pids": [terminal, *children], "workdir": str(session._workdir)}}), flush=True)
            await asyncio.sleep(600)

        asyncio.run(main())
    """)

    backend.kill()
    backend.wait()

    assert still_running(started["pids"], 5) == []
    shutil.rmtree(started["workdir"])  # the private copies a killed backend could not delete
