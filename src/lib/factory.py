"""The agent factory: tasks a Pi coding session works through in a workspace of a repo, driven by the factory extension.

- Consumes `lib.pi_host` (the task's session: opened, prompted with `/factory <command>`, its `factory-` entries, its
  removal), `lib.factory_workspace` (the task's worktree), `lib.factory_preview` (a deleted task's preview is
  stopped), git for the diff, and the tables `factory_repo_settings` and `factory_tasks` (`db/schema/0017_factory_tasks.sql`)
- Provides the repos with their settings and local branches, and a task's life: start, list, read, diff, command, message, delete
- A repo is the absolute path of a primary git checkout, its name the folder's, for display only
- The extension (a Pi user extension) runs the rounds, gates, review and merge inside the session. A task's state is
  the session's last `factory-state` entry, which pi-host reports as `SessionInfo.factory`: no state is stored here
- Settings chain: the defaults of `RepoSettings`, the repo's stored settings, a task's overrides. `branch` resolves at
  a task's start (`resolve_branch`): its worktree starts at that branch's tip, is rebased onto it and merges into it
- pi-host failures are `lib.pi_host`'s own, workspace failures `WorkspaceError`, git failures `GitError`
"""

import asyncio
import contextlib
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import shutil
from typing import Annotated, Any, Literal
from uuid import uuid4

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pydantic import AfterValidator, BaseModel, ConfigDict, Field
import yaml

from lib import factory_preview, pi_host
from lib.factory_workspace import Workspace, WorkspaceError, create_workspace, remove_workspace
from lib.git import GitError, git_ok, git_text, run_git

log = logging.getLogger(__name__)

TaskCommand = Literal["approve", "retry", "cancel"]
Role = Literal["coder", "reviewer"]

# Prefix of every entry the extension writes into a task's session.
ENTRY_PREFIX = "factory-"
# The repo file the extension reads its hooks and gates from.
CONFIG_FILE = "factory.yaml"
# A task in one of these states has its session at work while a Pi child runs it: it is not deleted.
BUSY_STATES = ("running", "merging")
# How long a start may take to show its `factory-task` entry: Pi writes the file with the first coder prompt.
START_WAIT_SECONDS = 4.0
START_POLL_SECONDS = 0.2
DIFF_TIMEOUT_SECONDS = 30.0
_DIFF_OPTIONS = ("--no-color", "--no-ext-diff", "--src-prefix=a/", "--dst-prefix=b/")
_TASK_COLUMNS = "id, repo, session_id, session_path, worktree, branch, title, created_at"


class UnknownRepo(LookupError):
    """No repo is registered at that path."""


class UnknownTask(LookupError):
    """No task has that id, or its session no longer says what the task is."""


class NotACheckout(ValueError):
    """The path is not the top of a primary git checkout."""


class TaskBusy(RuntimeError):
    """The task's session is at work (running or merging)."""


class UnknownBranch(ValueError):
    """The task's branch is not a local branch of the repo, or the repo has no default branch, `main` or `master`."""


class ConfigExists(RuntimeError):
    """The repo already has a `factory.yaml`, which is never overwritten."""


class InvalidConfig(ValueError):
    """The text is no `factory.yaml` the factory can read."""


class StartFailed(RuntimeError):
    """The session did not take the task: the extension refused the start, or is not loaded."""


# A conservative subset of `git check-ref-format --branch`: no leading `-`, `.` or `/` in a component, no `..`, `//`, `@{`.
_BRANCH_PATTERN = re.compile(r"(?!-)(?!.*\.\.)(?!.*//)(?!.*@\{)(?!.*/\.)(?!\.)[A-Za-z0-9._/@-]+(?<![./])(?<!\.lock)")


def _branch_name(name: str) -> str:
    if not _BRANCH_PATTERN.fullmatch(name) or name == "@":
        raise ValueError(f"{name!r} is not a valid branch name")
    return name


BranchName = Annotated[str, AfterValidator(_branch_name)]


class FactoryModel(BaseModel):
    """Factory JSON is snake_case like the rest of gigachad's API. Unknown fields are ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True)


class RepoSettings(FactoryModel):
    """How a repo's tasks run. `models` maps a role to a model selector (`provider/id`), a missing role uses Pi's own.

    `branch` is the branch a task's worktree starts from, is rebased onto and merges into: None for the repo's default
    branch (origin's `HEAD`), else `main`, else `master`.
    """

    merge: Literal["auto", "review"] = "review"
    max_rounds: int = Field(default=3, ge=1, le=20)
    review: bool = False
    preview: bool = False
    models: dict[Role, str] = {}
    branch: BranchName | None = None


class SettingsOverrides(FactoryModel):
    """The settings one task changes, None leaves the repo's."""

    merge: Literal["auto", "review"] | None = None
    max_rounds: int | None = Field(default=None, ge=1, le=20)
    review: bool | None = None
    preview: bool | None = None
    models: dict[Role, str] | None = None
    branch: BranchName | None = None


class RepoInfo(FactoryModel):
    """A registered repo: `path` identifies it, `name` is its folder."""

    name: str
    path: str


class NewRepo(FactoryModel):
    path: str


class NewTask(FactoryModel):
    repo: str
    title: str = Field(min_length=1)
    request: str = Field(min_length=1)
    overrides: SettingsOverrides = SettingsOverrides()


class RepoConfig(FactoryModel):
    """`exists`: the repo has a `factory.yaml`. `draft`: the generated one, None when it exists."""

    exists: bool
    draft: str | None


class NewConfig(FactoryModel):
    content: str


class NewMessage(FactoryModel):
    text: str = Field(min_length=1)


class TaskRow(FactoryModel):
    """A task as stored: `worktree` its folder, `branch` its own branch. Both are None for a task of an older version
    that worked in the repo's checkout."""

    id: str
    repo: str
    session_id: str
    session_path: str
    worktree: str | None
    branch: str | None
    title: str
    created_at: datetime

    @property
    def cwd(self) -> str:
        """The folder the task's session runs in."""
        return self.worktree or self.repo

    @property
    def workspace(self) -> Workspace:
        return Workspace(path=Path(self.cwd), branch=self.branch, worktree=self.worktree is not None)


class Task(TaskRow):
    """A task with what pi-host says about its session: `factory` its state (None before one is written or with the
    session gone), `live` its Pi child (None when stopped), `usage` the tokens and dollars of its coder and reviews."""

    factory: pi_host.FactoryState | None
    live: pi_host.LiveStatus | None
    usage: pi_host.SessionUsage | None = None


class TaskDiff(FactoryModel):
    """The task's changes as one unified diff against the commit `base`."""

    base: str
    diff: str


class FactoryStore:
    """The factory's rows: `factory_repo_settings` and `factory_tasks`. Every method blocks on the database."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def repos(self) -> list[str]:
        """Registered repo paths, by path."""
        with self._pool.connection() as connection:
            rows = connection.execute("SELECT repo FROM factory_repo_settings ORDER BY repo").fetchall()
        return [repo for (repo,) in rows]

    def register_repo(self, repo: str) -> None:
        """Register *repo* without settings of its own, a registered one stays as it is."""
        with self._pool.connection() as connection:
            connection.execute("INSERT INTO factory_repo_settings (repo) VALUES (%s) ON CONFLICT (repo) DO NOTHING", (repo,))

    def repo_settings(self, repo: str) -> dict[str, Any]:
        """The settings stored for *repo*, empty before a save. `UnknownRepo` when it is not registered."""
        with self._pool.connection() as connection:
            row = connection.execute("SELECT settings FROM factory_repo_settings WHERE repo = %s", (repo,)).fetchone()
        if row is None:
            raise UnknownRepo(f"Unknown repo {repo}")
        return row[0]

    def save_repo_settings(self, repo: str, settings: RepoSettings) -> None:
        """Replace the settings of *repo*. `UnknownRepo` when it is not registered."""
        with self._pool.connection() as connection:
            saved = connection.execute(
                "UPDATE factory_repo_settings SET settings = %s WHERE repo = %s", (Jsonb(settings.model_dump()), repo)
            ).rowcount
        if not saved:
            raise UnknownRepo(f"Unknown repo {repo}")

    def tasks(self, repo: str) -> list[TaskRow]:
        """The tasks of *repo*, newest first."""
        with self._pool.connection() as connection:
            rows = connection.execute(
                f"SELECT {_TASK_COLUMNS} FROM factory_tasks WHERE repo = %s ORDER BY created_at DESC, id", (repo,)
            ).fetchall()
        return [_task_row(row) for row in rows]

    def task(self, task_id: str) -> TaskRow:
        """The task *task_id*, `UnknownTask` without one."""
        with self._pool.connection() as connection:
            row = connection.execute(f"SELECT {_TASK_COLUMNS} FROM factory_tasks WHERE id = %s", (task_id,)).fetchone()
        if row is None:
            raise UnknownTask(f"Unknown task {task_id}")
        return _task_row(row)

    def insert_task(
        self, *, task_id: str, repo: str, session_id: str, session_path: str, worktree: str | None, branch: str | None, title: str
    ) -> TaskRow:
        with self._pool.connection() as connection:
            row = connection.execute(
                f"""
                INSERT INTO factory_tasks (id, repo, session_id, session_path, worktree, branch, title)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING {_TASK_COLUMNS}
                """,
                (task_id, repo, session_id, session_path, worktree, branch, title),
            ).fetchone()
        return _task_row(row)

    def delete_task(self, task_id: str) -> None:
        with self._pool.connection() as connection:
            connection.execute("DELETE FROM factory_tasks WHERE id = %s", (task_id,))


def _task_row(row: tuple) -> TaskRow:
    return TaskRow(**dict(zip(_TASK_COLUMNS.split(", "), row, strict=True)))


def resolve_settings(stored: dict[str, Any], overrides: SettingsOverrides | None = None) -> RepoSettings:
    """The settings chain: defaults, the repo's *stored* settings, a task's *overrides*. `models` merge role by role.

    Settings saved while a task had a source and a target branch name the one it merged into as `branch`.
    """
    if "branch" not in stored and (legacy := stored.get("target_branch") or stored.get("source_branch")):
        stored = {**stored, "branch": legacy}
    merged = RepoSettings().model_dump()
    for layer in (stored, overrides.model_dump(exclude_none=True) if overrides else {}):
        merged = {**merged, **layer, "models": {**merged["models"], **layer.get("models", {})}}
    return RepoSettings.model_validate(merged)


def repo_info(repo: str) -> RepoInfo:
    return RepoInfo(name=Path(repo).name, path=repo)


def checkout_path(path: str) -> str:
    """The resolved *path* when it is the top of a primary git checkout, `NotACheckout` otherwise.

    A linked worktree is refused: tasks get worktrees of the primary checkout.
    """
    folder = Path(path).expanduser()
    if not folder.is_absolute() or not folder.is_dir():
        raise NotACheckout(f"{path} is not an absolute folder")
    folder = folder.resolve()
    try:
        common_dir = git_text(folder, "rev-parse", "--path-format=absolute", "--git-common-dir").strip()
    except GitError as exc:
        raise NotACheckout(f"{path} is not a git checkout") from exc
    if Path(common_dir) != folder / ".git":
        raise NotACheckout(f"{path} is not the top of a primary git checkout")
    return str(folder)


def _local_branch(repo: Path, branch: str) -> bool:
    return git_ok(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}")


def _default_branch(repo: Path) -> str | None:
    """The branch origin's `HEAD` names, the repo's default branch; None without one."""
    try:
        return git_text(repo, "symbolic-ref", "--short", "refs/remotes/origin/HEAD").strip().removeprefix("origin/") or None
    except GitError:
        return None


def resolve_branch(repo: Path, branch: str | None) -> str:
    """The branch a task starts from and merges into: *branch*, unset the repo's default branch (origin's `HEAD`), else
    `main`, else `master`. It must be a local branch of *repo*, `UnknownBranch` otherwise."""
    if branch is None:
        branch = next((name for name in (_default_branch(repo), "main", "master") if name and _local_branch(repo, name)), None)
        if branch is None:
            raise UnknownBranch("the repo has no default branch, main or master, set a branch")
    elif not _local_branch(repo, branch):
        raise UnknownBranch(f"the branch {branch} is not a local branch of {repo}")
    return branch


def workspace_name(title: str, task_id: str) -> str:
    """Branch and folder name of a task's worktree: the slug of its title and its short id."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40].strip("-") or "task"
    return f"{slug}-{task_id[:8]}"


def start_command(title: str, request: str, settings: RepoSettings, branch: str) -> str:
    """`/factory start <json>` as the plugin reads it, settings in its camelCase shape, `branch` the resolved one."""
    task = {
        "request": request,
        "title": title,
        "settings": {
            "merge": settings.merge,
            "maxRounds": settings.max_rounds,
            "review": settings.review,
            "models": settings.models,
            "branch": branch,
        },
    }
    return f"/factory start {json.dumps(task, ensure_ascii=False)}"


def reviews_directory(session_id: str) -> Path:
    """Where the extension keeps the reviewer sessions of the task in session *session_id* (Pi's agent dir)."""
    agent = os.environ.get("PI_CODING_AGENT_DIR")
    return (Path(agent).expanduser() if agent else Path.home() / ".pi" / "agent") / "factory-reviews" / session_id


def _entry_data(entries: list[dict[str, Any]], custom_type: str) -> list[dict[str, Any]]:
    return [entry.get("data") or {} for entry in entries if entry.get("customType") == custom_type]


def diff_base(workspace: Path, entries: list[dict[str, Any]]) -> str:
    """The commit a task's changes are measured from, given its session's `factory-` *entries*: the merge base of its
    branch and the branch it merges into (the plugin's `taskBase`)."""
    started = _entry_data(entries, "factory-task")
    if not started:
        raise UnknownTask("The task's session has no factory-task entry")
    return git_text(workspace, "merge-base", "HEAD", started[-1]["mainBranch"]).strip()


def workspace_diff(workspace: Path, base: str) -> str:
    """Unified diff of *workspace* (index and work tree) against the commit *base*, untracked files included as new
    files. Reads only: nothing is added to the index."""
    diff = git_text(workspace, "diff", *_DIFF_OPTIONS, base, timeout=DIFF_TIMEOUT_SECONDS)
    others = git_text(workspace, "ls-files", "--others", "--exclude-standard", "-z", timeout=DIFF_TIMEOUT_SECONDS)
    for file in filter(None, others.split("\0")):
        # `--no-index` exits 1 when the files differ, which a new file always does.
        new_file = ("diff", *_DIFF_OPTIONS, "--no-index", "--", "/dev/null", file)
        changed = run_git(workspace, *new_file, timeout=DIFF_TIMEOUT_SECONDS, check=False)
        if changed.returncode > 1:
            raise GitError(changed.stderr.strip())
        diff += changed.stdout
    return diff


def _with_session(row: TaskRow, session: pi_host.SessionInfo | None) -> Task:
    if session is None:
        return Task(**row.model_dump(), factory=None, live=None)
    return Task(**row.model_dump(), factory=session.factory, live=session.live, usage=session.usage)


async def _sessions_of(rows: list[TaskRow], cwd: str | None) -> dict[str, pi_host.SessionInfo]:
    """pi-host's rows of the sessions of *rows*, all in *cwd* when given, by session id, archived or not.

    The unarchived listing covers Pi's normal root (older tasks) and the factory root. The archive is listed only when
    it misses one of them.
    """
    unarchived = await pi_host.list_sessions(cwd, archived=False, factory=True)
    sessions = {session.session_id: session for session in unarchived}
    if any(row.session_id not in sessions for row in rows):
        sessions.update((session.session_id, session) for session in await pi_host.list_sessions(cwd, archived=True))
    return sessions


async def _session_of(row: TaskRow) -> pi_host.SessionInfo | None:
    """pi-host's row of the task's session, None when it knows none."""
    return (await _sessions_of([row], row.cwd)).get(row.session_id)


async def list_repos(store: FactoryStore) -> list[RepoInfo]:
    return [repo_info(repo) for repo in await asyncio.to_thread(store.repos)]


async def register_repo(store: FactoryStore, path: str) -> RepoInfo:
    """Register the primary git checkout at *path*, repeating a known path returns the same repo. `NotACheckout` otherwise."""
    repo = await asyncio.to_thread(checkout_path, path)
    await asyncio.to_thread(store.register_repo, repo)
    return repo_info(repo)


async def repo_settings(store: FactoryStore, repo: str) -> RepoSettings:
    """The settings new tasks of *repo* start with: the defaults under what was saved for it."""
    return resolve_settings(await asyncio.to_thread(store.repo_settings, repo))


async def repo_branches(store: FactoryStore, repo: str) -> list[str]:
    """The local branches of the registered *repo*, alphabetically (case-insensitive). `UnknownRepo` when it is not registered."""
    await asyncio.to_thread(store.repo_settings, repo)
    names = await asyncio.to_thread(git_text, Path(repo), "for-each-ref", "--format=%(refname:short)", "refs/heads/")
    return sorted(names.split(), key=lambda name: (name.casefold(), name))


def _config_path(repo: str) -> Path:
    return Path(repo) / CONFIG_FILE


def _validate_config(content: str) -> None:
    try:
        document = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise InvalidConfig(f"{CONFIG_FILE} does not parse: {exc}") from exc
    if not isinstance(document, dict):
        raise InvalidConfig(f"{CONFIG_FILE} must be a mapping")
    for key in ("gates", "merge_gates"):
        gates = document.get(key) or []
        if not isinstance(gates, list) or not all(
            isinstance(gate, list) and gate and all(isinstance(item, str) for item in gate) for gate in gates
        ):
            raise InvalidConfig(f"{key} must be a list of argv lists of strings")


async def repo_config(store: FactoryStore, repo: str) -> RepoConfig:
    """Whether the registered *repo* has a `factory.yaml`, else a draft of one. `UnknownRepo` when it is not registered."""
    await asyncio.to_thread(store.repo_settings, repo)
    if await asyncio.to_thread(_config_path(repo).exists):
        return RepoConfig(exists=True, draft=None)
    return RepoConfig(exists=False, draft=await pi_host.factory_draft(repo))


def _write_config(repo: str, content: str) -> None:
    _validate_config(content)
    try:
        with _config_path(repo).open("x", encoding="utf-8") as file:
            file.write(content)
    except FileExistsError as exc:
        raise ConfigExists(f"{repo} already has a {CONFIG_FILE}") from exc


async def save_repo_config(store: FactoryStore, repo: str, content: str) -> RepoConfig:
    """Write *content* as the uncommitted `factory.yaml` of the registered *repo*. `ConfigExists` rather than overwrite."""
    await asyncio.to_thread(store.repo_settings, repo)
    await asyncio.to_thread(_write_config, repo, content)
    return RepoConfig(exists=True, draft=None)


async def save_repo_settings(store: FactoryStore, repo: str, settings: RepoSettings) -> RepoSettings:
    await asyncio.to_thread(store.save_repo_settings, repo, settings)
    return settings


async def list_tasks(store: FactoryStore, repo: str) -> list[Task]:
    """The tasks of *repo*, newest first, each with its session's state from one pi-host listing (two when one of
    them is archived or lost)."""
    await asyncio.to_thread(store.repo_settings, repo)
    rows = await asyncio.to_thread(store.tasks, repo)
    if not rows:
        return []
    sessions = await _sessions_of(rows, None)
    return [_with_session(row, sessions.get(row.session_id)) for row in rows]


async def read_task(store: FactoryStore, task_id: str) -> Task:
    row = await asyncio.to_thread(store.task, task_id)
    return _with_session(row, await _session_of(row))


async def _started(session_id: str) -> bool:
    """Whether the session shows a `factory-task` entry within `START_WAIT_SECONDS`. A new session has no file at first."""
    deadline = asyncio.get_running_loop().time() + START_WAIT_SECONDS
    while True:
        try:
            entries = (await pi_host.session_entries(session_id, custom_type=ENTRY_PREFIX)).entries
        except pi_host.PiHostRejected as exc:
            if exc.status != 404:
                raise
            entries = []
        if _entry_data(entries, "factory-task"):
            return True
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(START_POLL_SECONDS)


async def _discard(session_id: str | None, workspace: Workspace) -> None:
    """Undo a start that failed: the session, then the workspace. What cannot be removed is logged, the start's own
    failure is what the caller reports."""
    if session_id is not None:
        with contextlib.suppress(pi_host.PiHostUnavailable, pi_host.PiHostRejected):
            await pi_host.delete_session(session_id)
    try:
        await asyncio.to_thread(remove_workspace, workspace)
    except WorkspaceError as exc:
        log.warning("The workspace %s of a task that did not start stays: %s", workspace.path, exc)


async def create_task(store: FactoryStore, new_task: NewTask) -> Task:
    """Start a task of a registered repo: its worktree, a coding session in it, `/factory start`, then its row.

    - The worktree starts at the tip of the resolved branch (`UnknownBranch`), which the task also merges into
    - `StartFailed` when the session shows no `factory-task` entry. Any failure after the workspace exists removes
      the session and the workspace again
    """
    settings = resolve_settings(await asyncio.to_thread(store.repo_settings, new_task.repo), new_task.overrides)
    repo = Path(new_task.repo)
    branch = await asyncio.to_thread(resolve_branch, repo, settings.branch)
    task_id = uuid4().hex
    workspace = await asyncio.to_thread(create_workspace, repo, workspace_name(new_task.title, task_id), branch)
    session_id: str | None = None
    try:
        session = await pi_host.open_session(str(workspace.path), None, factory_repo=new_task.repo)
        session_id = session.session_id
        command = start_command(new_task.title, new_task.request, settings, branch)
        disposition = await pi_host.prompt_session(session_id, command)
        if not await _started(session_id):
            raise StartFailed(
                f"The session did not start the task (Pi answered {disposition!r}, no factory-task entry): the factory"
                " plugin refused the start (not a git repository, an unknown model) or is not loaded."
                f" {pi_host.FACTORY_PLUGIN_HINT}"
            )
        row = await asyncio.to_thread(
            store.insert_task,
            task_id=task_id,
            repo=new_task.repo,
            session_id=session_id,
            session_path=session.session_path,
            worktree=str(workspace.path) if workspace.worktree else None,
            branch=workspace.branch,
            title=new_task.title,
        )
    except BaseException:
        await _discard(session_id, workspace)
        raise
    return _with_session(row, await _session_of(row))


async def _prompt(
    store: FactoryStore,
    task_id: str,
    message: str,
    streaming_behavior: Literal["followUp"] | None = None,
) -> Task:
    """Send *message* to the task's session, opened first when it is not live."""
    row = await asyncio.to_thread(store.task, task_id)
    session = await pi_host.open_session(row.cwd, pi_host.ResumeById(session_id=row.session_id))
    await pi_host.prompt_session(row.session_id, message, streaming_behavior)
    return _with_session(row, session)


async def open_task_session(store: FactoryStore, task_id: str) -> Task:
    """Make the task's session live in pi-host without sending it anything, so a client can attach to it."""
    row = await asyncio.to_thread(store.task, task_id)
    return _with_session(row, await pi_host.open_session(row.cwd, pi_host.ResumeById(session_id=row.session_id)))


async def command_task(store: FactoryStore, task_id: str, command: TaskCommand) -> Task:
    """Approve, retry or cancel the task. The answer is the task as it was just before: its new state follows in the session."""
    return await _prompt(store, task_id, f"/factory {command}")


async def message_task(store: FactoryStore, task_id: str, text: str) -> Task:
    """Hand the user's message to the task's coder: queued behind a running turn, new rounds for a task at rest.

    `TaskBusy` while it merges: the extension would drop the message.
    """
    row = await asyncio.to_thread(store.task, task_id)
    session = await _session_of(row)
    if session is not None and session.factory is not None and session.factory.state == "merging":
        raise TaskBusy("The task is merging, send the message when it has ended")
    return await _prompt(store, task_id, text, "followUp")


async def task_diff(store: FactoryStore, task_id: str) -> TaskDiff:
    """The task's changes against its diff base. `UnknownTask` when its session has no `factory-task` entry."""
    row = await asyncio.to_thread(store.task, task_id)
    entries = (await pi_host.session_entries(row.session_id, custom_type=ENTRY_PREFIX)).entries
    base = await asyncio.to_thread(diff_base, Path(row.cwd), entries)
    return TaskDiff(base=base, diff=await asyncio.to_thread(workspace_diff, Path(row.cwd), base))


async def delete_task(store: FactoryStore, task_id: str) -> None:
    """Remove the task for good: its preview, its session, its reviewer sessions, its worktree and branch, its row.

    `TaskBusy` while it is running or merging in a live session: an interrupted one (no Pi child) is removed. A session
    pi-host no longer knows is taken as removed. `factory_preview.PreviewError` when the preview does not stop: then
    nothing is removed.
    """
    row = await asyncio.to_thread(store.task, task_id)
    session = await _session_of(row)
    if session is not None and session.live is not None and session.factory is not None and session.factory.state in BUSY_STATES:
        raise TaskBusy(f"The task is {session.factory.state}, cancel it first")
    await asyncio.to_thread(factory_preview.stop_preview, task_id)
    try:
        await pi_host.delete_session(row.session_id)
    except pi_host.PiHostRejected as exc:
        if exc.status != 404:
            raise
    await asyncio.to_thread(shutil.rmtree, reviews_directory(row.session_id), True)
    await asyncio.to_thread(remove_workspace, row.workspace)
    await asyncio.to_thread(store.delete_task, task_id)
