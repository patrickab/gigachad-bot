"""On-demand previews of a factory task's workspace, and where the browser reaches them.

- Consumes the `preview` section of the task worktree's `factory.yaml` (the factory core owns the rest of the file),
  `GIGACHAD_DATABASE_URL` (the server a preview's throwaway database is made on) and `GIGACHAD_PREVIEW_URL`, where
  the browser reaches the preview gate (default 127.0.0.1:30185)
- Provides `start_preview`, `get_preview` and `stop_preview`: the processes of one task on loopback ports
  `PREVIEW_PORTS`, each in its own process group and session, with an empty database and a data directory of their own.
  They get PATH, HOME, USER and LANG plus the config's `env`, nothing else of the backend's environment
- Placeholders in `argv`, `ready_url`, `env` and `url`: `{port.<process name>}`, `{repo_path}` (the primary checkout),
  `{database_url}` (with `database: true`) and `{data_dir}`
- State is one directory per task under `$XDG_STATE_HOME/gigachad-bot/factory-previews`: `state.json`, a log per
  process and `data/`. Every backend worker reads it and a preview outlives a backend reload, nothing stops it but
  `stop_preview`. A lock file beside the directories makes starts and stops take turns across workers
- A recorded process counts only while `/proc` shows it with the start time and boot it was recorded with, so state
  left by a reboot or a reused pid reads as `failed` and its process group is never signalled (Linux only)
- `through_gate` moves a preview's loopback URL onto the gate (`backend/preview_gate.py`): the browser cannot reach
  the port itself
- `start_preview` and `stop_preview` block on subprocesses and the database, `get_preview` on HTTP requests: call
  them with `asyncio.to_thread`
"""

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
import fcntl
import http.client
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time
from typing import Annotated, Literal
import urllib.error
from urllib.parse import urlsplit
import urllib.request

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
import yaml

from lib.db_schema import database_url
from lib.upstream import upstream_url

CONFIG_NAME = "factory.yaml"
PREVIEW_PORTS = range(8100, 8200)
DEFAULT_PREVIEW_URL = "http://127.0.0.1:30185"
LOOPBACK = "127.0.0.1"
INHERITED_ENVIRONMENT = ("PATH", "HOME", "USER", "LANG")
READY_REQUEST_TIMEOUT_SECONDS = 2.0
# How long a process group gets to end after SIGTERM, then after SIGKILL.
TERMINATE_WAIT_SECONDS = 5.0
KILL_WAIT_SECONDS = 2.0
LOG_TAIL_BYTES = 1500
# A PostgreSQL that does not answer must not hold the preview lock, and every start, stop and task delete behind it.
DATABASE_CONNECT_TIMEOUT_SECONDS = 10
_PLACEHOLDER = re.compile(r"\{(port\.[^{}\s]+|repo_path|database_url|data_dir)\}")
# Readiness requests go straight to loopback, whatever proxy the backend's environment names.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))
# The preview processes this worker started, kept only to reap them when they end: their state is on disk.
_children: dict[int, subprocess.Popen] = {}


class PreviewError(RuntimeError):
    """A preview could not be started or stopped, or `factory.yaml` is invalid. The message says why."""


class PreviewNotConfigured(LookupError):
    """The workspace's `factory.yaml` has no `preview` section."""


@dataclass(frozen=True)
class PreviewTask:
    """The factory task a preview runs for, from the core's `tasks.get`: `id` keys its state, `worktree` is where its
    processes run, `repo` (the primary checkout) fills `{repo_path}`."""

    id: str
    repo: str
    worktree: str


class ConfigModel(BaseModel):
    """A part of `factory.yaml`. Keys this module does not know are ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True)


Argv = Annotated[list[str], Field(min_length=1)]


class Preview(BaseModel):
    """A task's preview as the browser sees it. `url` is on the preview gate, `error` ends with a log tail."""

    state: Literal["starting", "ready", "failed", "stopped"]
    url: str | None = None
    error: str | None = None


class PreviewProcess(ConfigModel):
    """One process of a preview. Its `name` picks its port (`{port.<name>}`), `ready_url` answers once it serves."""

    name: Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")]
    argv: Argv
    ready_url: str


class PreviewConfig(ConfigModel):
    """The `preview` section of `factory.yaml`."""

    processes: Annotated[list[PreviewProcess], Field(min_length=1)]
    env: dict[str, str] = Field(default_factory=dict)
    database: bool = False
    url: str
    ready_timeout_s: Annotated[int, Field(gt=0)] = 180

    @model_validator(mode="after")
    def _unique_process_names(self) -> "PreviewConfig":
        names = [process.name for process in self.processes]
        if len(set(names)) != len(names):
            raise ValueError("preview process names must be unique")
        return self


class ProcessState(BaseModel):
    """A started process. `pid` is also its process group and session, `start_ticks` tells it from a later process
    with the same pid."""

    name: str
    pid: int
    start_ticks: int
    ready_url: str
    log: str


class PreviewState(BaseModel):
    """What `state.json` keeps of a preview. `ports` holds every process name, `processes` those started so far."""

    task_id: str
    boot_id: str
    url: str
    started_at: float
    ready_timeout_s: int
    ports: dict[str, int]
    database: str | None
    data_dir: str
    processes: list[ProcessState] = []


def preview_gate_url() -> str:
    """Base URL the browser reaches the preview gate at."""
    return upstream_url("GIGACHAD_PREVIEW_URL", DEFAULT_PREVIEW_URL)


def through_gate(url: str) -> str:
    """Move a preview's loopback URL onto the preview gate, keeping path and query. Any other URL stays as it is."""
    parts = urlsplit(url)
    if parts.hostname != LOOPBACK or parts.port not in PREVIEW_PORTS:
        return url
    return f"{preview_gate_url()}{url.removeprefix(f'{parts.scheme}://{parts.netloc}')}"


def start_preview(task: PreviewTask) -> Preview:
    """Start the preview of *task* in its workspace and answer `starting` at once, `get_preview` tells when it serves.

    A preview that already starts or serves is answered as it is, a failed one is torn down and started again.
    `PreviewNotConfigured` without a `preview` section, `PreviewError` when a port, the database or a process fails.
    """
    config = _preview_config(Path(task.worktree))
    with _locked():
        _reap()
        existing = _read_state(task.id)
        if existing is not None:
            preview = _status(existing)
            if preview.state != "failed":
                return preview
            _tear_down(existing)
        directory = _task_directory(task.id)
        data_dir = directory / "data"
        ports = _allocate_ports([process.name for process in config.processes])
        values = {"repo_path": task.repo, "data_dir": str(data_dir)}
        values.update({f"port.{name}": str(port) for name, port in ports.items()})
        database = f"gigachad_preview_{task.id}" if config.database else None
        if database is not None:
            values["database_url"] = make_conninfo(database_url(), dbname=database)
        environment = {name: os.environ[name] for name in INHERITED_ENVIRONMENT if name in os.environ}
        environment.update({name: _filled(value, values) for name, value in config.env.items()})
        commands = [
            (process.name, [_filled(argument, values) for argument in process.argv], _filled(process.ready_url, values))
            for process in config.processes
        ]
        state = PreviewState(
            task_id=task.id, boot_id=_boot_id(), url=_filled(config.url, values), started_at=time.time(),
            ready_timeout_s=config.ready_timeout_s, ports=ports, database=database, data_dir=str(data_dir),
        )
        if database is not None:
            _create_database(database)
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            _write_state(state)
            for name, argv, ready_url in commands:
                state.processes.append(_spawn(name, argv, ready_url, Path(task.worktree), environment, directory / f"{name}.log"))
                _write_state(state)
        except OSError as exc:
            _tear_down(state)
            raise PreviewError(f"The preview did not start: {exc}") from exc
        return Preview(state="starting", url=through_gate(state.url))


def get_preview(task_id: str) -> Preview | None:
    """The preview of *task_id*, None when none was started or it was stopped.

    `ready` once every `ready_url` answers, `failed` when a process no longer runs or `ready_timeout_s` passed
    without all of them answering, else `starting`. A failed preview keeps its logs and ports until it is stopped
    or started again.
    """
    _reap()
    state = _read_state(task_id)
    return None if state is None else _status(state)


def stop_preview(task_id: str) -> None:
    """End the preview of *task_id*: its process groups (SIGTERM, then SIGKILL), its database, its directory.

    Nothing to stop is fine. `PreviewError` when the database cannot be dropped: the state stays for another try.
    """
    with _locked():
        _reap()
        state = _read_state(task_id)
        if state is not None:
            _tear_down(state)


def _preview_config(workspace: Path) -> PreviewConfig:
    """The `preview` section of the workspace's `factory.yaml`, placeholders as written."""
    file = workspace / CONFIG_NAME
    try:
        document = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        document = {}
    except yaml.YAMLError as exc:
        raise PreviewError(f"{file} is invalid: {exc}") from exc
    if not isinstance(document, dict):
        raise PreviewError(f"{file} is invalid: it is not a mapping")
    section = document.get("preview")
    if not section:
        raise PreviewNotConfigured(f"{file} has no preview section")
    try:
        return PreviewConfig.model_validate(section)
    except ValidationError as exc:
        raise PreviewError(f"The preview section of {file} is invalid: {exc}") from exc


def _state_root() -> Path:
    state_home = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state_home) / "gigachad-bot" / "factory-previews"


def _task_directory(task_id: str) -> Path:
    return _state_root() / task_id


@contextmanager
def _locked() -> Iterator[None]:
    """Hold the previews' lock: one start or stop at a time over all workers, so no two take the same port."""
    root = _state_root()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _read_state(task_id: str) -> PreviewState | None:
    try:
        return PreviewState.model_validate_json((_task_directory(task_id) / "state.json").read_text())
    except FileNotFoundError:
        return None


def _readable_state(task_id: str) -> PreviewState | None:
    """The task's state, None when it cannot be read: one bad file must not block every other preview's start."""
    try:
        return _read_state(task_id)
    except (OSError, ValueError):  # pydantic's ValidationError is a ValueError
        return None


def _write_state(state: PreviewState) -> None:
    """Replace the task's `state.json` in one step: a reader never sees half a file."""
    file = _task_directory(state.task_id) / "state.json"
    scratch = file.with_suffix(".tmp")
    scratch.write_text(state.model_dump_json())
    scratch.replace(file)


def _filled(template: str, values: dict[str, str]) -> str:
    def value(match: re.Match[str]) -> str:
        if match[1] not in values:
            raise PreviewError(f"The preview uses {{{match[1]}}}, which has no value: a process name or `database: true` is missing")
        return values[match[1]]

    return _PLACEHOLDER.sub(value, template)


def _allocate_ports(names: list[str]) -> dict[str, int]:
    """A free port of `PREVIEW_PORTS` per name: one no recorded preview holds and nothing listens on."""
    root = _state_root()
    states = [_readable_state(entry.name) for entry in root.iterdir() if entry.is_dir()]
    taken = {port for state in states if state is not None for port in state.ports.values()}
    ports: dict[str, int] = {}
    for name in names:
        port = next((port for port in PREVIEW_PORTS if port not in taken and _is_bindable(port)), None)
        if port is None:
            raise PreviewError(f"No free preview port for {name} in {PREVIEW_PORTS.start}-{PREVIEW_PORTS.stop - 1}")
        taken.add(port)
        ports[name] = port
    return ports


def _is_bindable(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((LOOPBACK, port))
        except OSError:
            return False
    return True


def _create_database(name: str) -> None:
    """An empty database *name* beside the backend's own, replacing one a crashed start left."""
    try:
        with psycopg.connect(database_url(), autocommit=True, connect_timeout=DATABASE_CONNECT_TIMEOUT_SECONDS) as connection:
            connection.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    except psycopg.Error as exc:
        raise PreviewError(f"The preview database {name} could not be created: {exc}") from exc


def _drop_database(name: str) -> None:
    try:
        with psycopg.connect(database_url(), autocommit=True, connect_timeout=DATABASE_CONNECT_TIMEOUT_SECONDS) as connection:
            connection.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))
    except psycopg.Error as exc:
        raise PreviewError(f"The preview database {name} could not be dropped: {exc}") from exc


def _spawn(name: str, argv: list[str], ready_url: str, workspace: Path, environment: dict[str, str], log: Path) -> ProcessState:
    """Start *argv* in *workspace* as the leader of a new session, its output going to *log*."""
    with log.open("wb") as output:
        process = subprocess.Popen(
            argv, cwd=workspace, env=environment, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    # Not reaped yet, so `/proc` still has it even if it ended already.
    _, _, start_ticks = _process_stat(process.pid)
    _children[process.pid] = process
    return ProcessState(name=name, pid=process.pid, start_ticks=start_ticks, ready_url=ready_url, log=str(log))


def _reap() -> None:
    """Collect the ended processes this worker started, so they do not stay behind as zombies."""
    for pid, process in list(_children.items()):
        if process.poll() is not None:
            del _children[pid]


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _process_stat(pid: int) -> tuple[str, int, int] | None:
    """State letter, process group and start time (clock ticks since boot) of *pid*, None when there is no such process."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # The fields after the command name, which may itself hold spaces and brackets: state is field 3 of `proc(5)`.
    fields = stat[stat.rindex(")") + 2 :].split()
    return fields[0], int(fields[2]), int(fields[19])


def _runs(state: PreviewState, process: ProcessState) -> bool:
    """Whether the recorded *process* itself still runs: same boot, same start time, not a zombie."""
    stat = _process_stat(process.pid)
    return state.boot_id == _boot_id() and stat is not None and stat[0] != "Z" and stat[2] == process.start_ticks


def _own_groups(state: PreviewState) -> list[int]:
    """The process groups of *state* that may be signalled.

    None from another boot, and none whose pid now belongs to a later process. A leader that is gone leaves its
    group to what it started (`npm exec` and the server under it): the kernel keeps that id from new processes.
    """
    if state.boot_id != _boot_id():
        return []
    groups = []
    for process in state.processes:
        stat = _process_stat(process.pid)
        if stat is None or stat[2] == process.start_ticks:
            groups.append(process.pid)
    return groups


def _running_groups() -> set[int]:
    """The process groups with a process that still runs, zombies not counted."""
    stats = [_process_stat(int(entry)) for entry in os.listdir("/proc") if entry.isdigit()]
    return {stat[1] for stat in stats if stat is not None and stat[0] != "Z"}


def _signal_groups(groups: list[int], signal_number: signal.Signals) -> None:
    for group in groups:
        with suppress(ProcessLookupError):
            os.killpg(group, signal_number)


def _wait_until_gone(groups: list[int], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while True:
        _reap()
        if _running_groups().isdisjoint(groups):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _tear_down(state: PreviewState) -> None:
    """End the processes of *state*, drop its database, then remove its directory and with it the state."""
    groups = _own_groups(state)
    _signal_groups(groups, signal.SIGTERM)
    if not _wait_until_gone(groups, TERMINATE_WAIT_SECONDS):
        _signal_groups(groups, signal.SIGKILL)
        _wait_until_gone(groups, KILL_WAIT_SECONDS)
    if state.database is not None:
        _drop_database(state.database)
    shutil.rmtree(_task_directory(state.task_id), ignore_errors=True)


def _responds(url: str) -> bool:
    try:
        with _DIRECT.open(url, timeout=READY_REQUEST_TIMEOUT_SECONDS) as response:
            return response.status < 500
    except urllib.error.HTTPError as exc:
        return exc.code < 500
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        return False


def _log_tail(log: str) -> str:
    """The end of the log file *log* as a suffix for an error message, empty without output."""
    try:
        with Path(log).open("rb") as file:
            file.seek(0, os.SEEK_END)
            file.seek(max(0, file.tell() - LOG_TAIL_BYTES))
            tail = file.read().decode(errors="replace").strip()
    except OSError:
        return ""
    return f":\n{tail}" if tail else ""


def _status(state: PreviewState) -> Preview:
    for process in state.processes:
        if not _runs(state, process):
            return Preview(state="failed", error=f"The preview process {process.name} is not running{_log_tail(process.log)}")
    started = {process.name for process in state.processes}
    # A name without a process: another worker is between two spawns, or died there.
    pending = [name for name in state.ports if name not in started]
    silent = [process for process in state.processes if not _responds(process.ready_url)]
    pending += [process.name for process in silent]
    if not pending:
        return Preview(state="ready", url=through_gate(state.url))
    if time.time() - state.started_at > state.ready_timeout_s:
        log_tail = _log_tail(silent[0].log) if silent else ""
        error = f"The preview was not ready after {state.ready_timeout_s} s: {', '.join(pending)}{log_tail}"
        return Preview(state="failed", error=error)
    return Preview(state="starting", url=through_gate(state.url))
