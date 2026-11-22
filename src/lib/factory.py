"""The agent factory: tasks a Pi coding session works through in a workspace of a repo, driven by the factory extension.

- Consumes `lib.pi_host` (the task's session: opened, prompted with `/factory <command>`, its `factory-` entries, its
  removal), `lib.factory_workspace` (worktree or current checkout), `lib.factory_preview` (a deleted task's preview is
  stopped), git for the diff, and the tables `factory_repo_settings` and `factory_tasks` (`db/schema/0017_factory_tasks.sql`)
- Provides the repos with their settings and local branches, and a task's life: start, list, read, diff, command, message, delete
- A repo is the absolute path of a primary git checkout, its name the folder's, for display only
- The extension (`pi-extensions/factory`) runs the rounds, gates, review and merge inside the session. A task's state is
  the session's last `factory-state` entry, which pi-host reports as `SessionInfo.factory`: no state is stored here
- Settings chain: the defaults of `RepoSettings`, the repo's stored settings, a task's overrides. `source_branch` and
  `target_branch` resolve at a worktree task's start (`resolve_branches`): the worktree starts at the source branch's
  tip, and is rebased onto and merged into the target branch
- One task at a time works in a repo's checkout: a start, a retry and a message there take the checkout first, under a
  lock all backend workers share (`FactoryStore.checkout_lock`)
- pi-host failures are `lib.pi_host`'s own, workspace failures `WorkspaceError` and `WorkspaceBusy`, git failures `GitError`
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
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

from lib import factory_preview, pi_host
from lib.factory_workspace import (
    Workspace,
    WorkspaceBusy,
    WorkspaceError,
    create_workspace,
    ensure_checkout_free,
    remove_workspace,
)
from lib.git import GitError, git_ok, git_text, run_git

log = logging.getLogger(__name__)

TaskCommand = Literal["approve", "retry", "cancel"]
Role = Literal["coder", "reviewer"]

# Prefix of every entry the extension writes into a task's session.
ENTRY_PREFIX = "factory-"
# A task in one of these states has its session at work while a Pi child runs it: it is not deleted.
BUSY_STATES = ("running", "merging")
# A task in one of these states no longer holds the checkout it worked in: another task may start there.
FINISHED_STATES = ("done", "cancelled")
# A task in one of these states is not at work in its checkout: another task of it takes a retry or a message.
RESTING_STATES = (*FINISHED_STATES, "blocked")
# How long a start may take to show its `factory-task` entry: Pi writes the file with the first coder prompt.
START_WAIT_SECONDS = 4.0
START_POLL_SECONDS = 0.2
DIFF_TIMEOUT_SECONDS = 30.0
_DIFF_OPTIONS = ("--no-color", "--no-ext-diff", "--src-prefix=a/", "--dst-prefix=b/")
# The extension's upstream refs of a task with its own source branch: `<prefix>/<task branch>/{base,next}`.
UPSTREAM_REFS = "refs/factory-upstream"
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
    """The task's source or target branch is not a local branch of the repo, or the repo has neither `main` nor `master`."""


class BranchConflict(ValueError):
    """The task's source branch does not merge cleanly into its target branch."""


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

    `source_branch` is the branch a worktree task starts from, None for `main`, else `master`. `target_branch` is the
    branch it is rebased onto and merges into, None for the source branch. A current-checkout task ignores both.
    """

    merge: Literal["auto", "review"] = "review"
    max_rounds: int = Field(default=3, ge=1, le=20)
    review: bool = True
    worktree: bool = True
    preview: bool = False
    models: dict[Role, str] = {}
    source_branch: BranchName | None = None
    target_branch: BranchName | None = None


class SettingsOverrides(FactoryModel):
    """The settings one task changes, None leaves the repo's."""

    merge: Literal["auto", "review"] | None = None
    max_rounds: int | None = Field(default=None, ge=1, le=20)
    review: bool | None = None
    worktree: bool | None = None
    preview: bool | None = None
    models: dict[Role, str] | None = None
    source_branch: BranchName | None = None
    target_branch: BranchName | None = None


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


class NewMessage(FactoryModel):
    text: str = Field(min_length=1)


class TaskRow(FactoryModel):
    """A task as stored. `worktree` and `branch` are None for a task that works in the repo's checkout."""

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
    """The task's changes as one unified diff against `base`: a commit, or the tree of the working tree a task in the
    checkout started on."""

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

    @contextlib.contextmanager
    def checkout_lock(self, repo: str) -> Iterator[None]:
        """Hold the checkout of *repo* until the block ends, across every backend worker. `WorkspaceBusy`, without
        waiting, while another block holds it."""
        with self._pool.connection() as connection:
            # Transaction-scoped: PostgreSQL releases it when the connection goes back to the pool, however the block ends.
            key = f"factory-checkout:{repo}"
            (locked,) = connection.execute("SELECT pg_try_advisory_xact_lock(hashtext(%s))", (key,)).fetchone()
            if not locked:
                raise WorkspaceBusy(f"Another task is starting or resuming in the checkout {repo}, try again")
            yield


def _task_row(row: tuple) -> TaskRow:
    return TaskRow(**dict(zip(_TASK_COLUMNS.split(", "), row, strict=True)))


def resolve_settings(stored: dict[str, Any], overrides: SettingsOverrides | None = None) -> RepoSettings:
    """The settings chain: defaults, the repo's *stored* settings, a task's *overrides*. `models` merge role by role.

    A stored `branch` (settings saved before source and target were apart) is both branches.
    """
    if "branch" in stored:
        legacy = stored.get("branch")
        stored = {"source_branch": legacy, "target_branch": legacy} | {k: v for k, v in stored.items() if k != "branch"}
    merged = RepoSettings().model_dump()
    for layer in (stored, overrides.model_dump(exclude_none=True) if overrides else {}):
        merged = {**merged, **layer, "models": {**merged["models"], **layer.get("models", {})}}
    return RepoSettings.model_validate(merged)


def repo_info(repo: str) -> RepoInfo:
    return RepoInfo(name=Path(repo).name, path=repo)


def checkout_path(path: str) -> str:
    """The resolved *path* when it is the top of a primary git checkout, `NotACheckout` otherwise.

    A linked worktree is refused: the extension would treat a task in it as a worktree task and merge into the primary's branch.
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


def resolve_branches(repo: Path, source: str | None, target: str | None) -> tuple[str, str]:
    """The source and target branch of a worktree task. The source: *source*, unset `main`, else `master`. The target:
    *target*, unset the source. Each must be a local branch of *repo*, `UnknownBranch` otherwise."""
    if source is None:
        source = next((name for name in ("main", "master") if _local_branch(repo, name)), None)
        if source is None:
            raise UnknownBranch("the repo has neither main nor master, set a source branch")
    elif not _local_branch(repo, source):
        raise UnknownBranch(f"the source branch {source} is not a local branch of {repo}")
    if target is None:
        target = source
    elif not _local_branch(repo, target):
        raise UnknownBranch(f"the target branch {target} is not a local branch of {repo}")
    if source != target:
        _ensure_merges(repo, source, target)
    return source, target


def _ensure_merges(repo: Path, source: str, target: str) -> None:
    """`BranchConflict` unless *source* merges cleanly into *target*: a task builds on their merge (the extension's
    upstream), which it could never build."""
    heads = (f"refs/heads/{target}", f"refs/heads/{source}")
    merged = run_git(repo, "merge-tree", "--write-tree", "--name-only", "--no-messages", *heads, check=False)
    if merged.returncode == 1:
        files = ", ".join(merged.stdout.split()[1:])
        raise BranchConflict(f"the source branch {source} does not merge cleanly into {target}: {files}")
    if merged.returncode != 0:
        raise GitError(merged.stderr.strip())


def _checked_out_branch(repo: Path) -> str:
    """The branch a current-checkout task works on, `HEAD` when it is detached."""
    return git_text(repo, "branch", "--show-current").strip() or "HEAD"


def workspace_name(title: str, task_id: str) -> str:
    """Branch and folder name of a task's worktree: the slug of its title and its short id."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40].strip("-") or "task"
    return f"{slug}-{task_id[:8]}"


def start_command(title: str, request: str, settings: RepoSettings, source: str, target: str) -> str:
    """`/factory start <json>` as the extension reads it, settings in its camelCase shape. The extension's `branch` is the
    resolved *target* branch, the one the task is rebased onto and merges into, `sourceBranch` the one its worktree
    started from."""
    task = {
        "request": request,
        "title": title,
        "settings": {
            "merge": settings.merge,
            "maxRounds": settings.max_rounds,
            "review": settings.review,
            "models": settings.models,
            "branch": target,
            "sourceBranch": source,
        },
    }
    return f"/factory start {json.dumps(task, ensure_ascii=False)}"


def reviews_directory(session_id: str) -> Path:
    """Where the extension keeps the reviewer sessions of the task in session *session_id* (Pi's agent dir)."""
    agent = os.environ.get("PI_CODING_AGENT_DIR")
    return (Path(agent).expanduser() if agent else Path.home() / ".pi" / "agent") / "factory-reviews" / session_id


def _entry_data(entries: list[dict[str, Any]], custom_type: str) -> list[dict[str, Any]]:
    return [entry.get("data") or {} for entry in entries if entry.get("customType") == custom_type]


def _worktree_base(workspace: Path, task: dict[str, Any]) -> str:
    """Where a worktree task's own change starts, as the extension's `taskBase` tells it."""
    source = (task.get("settings") or {}).get("sourceBranch")
    if not source or source == task["mainBranch"]:
        return git_text(workspace, "merge-base", "HEAD", task["mainBranch"]).strip()
    for name in ("next", "base"):
        ref = f"{UPSTREAM_REFS}/{task['branch']}/{name}"
        if git_ok(workspace, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}") and git_ok(
            workspace, "merge-base", "--is-ancestor", ref, "HEAD"
        ):
            return git_text(workspace, "rev-parse", f"{ref}^{{commit}}").strip()
    base = f"{UPSTREAM_REFS}/{task['branch']}/base"
    fallback = base if git_ok(workspace, "rev-parse", "--verify", "--quiet", base) else f"refs/heads/{source}"
    return git_text(workspace, "merge-base", "HEAD", fallback).strip()


def diff_base(workspace: Path, worktree: bool, entries: list[dict[str, Any]]) -> str:
    """The commit or tree a task's changes are measured from, given its session's `factory-` *entries*.

    - Worktree task: the merge base of its branch and the main branch. With a source branch that is not the main
      branch (`settings.sourceBranch`), the extension's `taskBase`: the newest of its upstream refs the branch holds,
      before the first rebase the merge base with the source branch
    - Current-checkout task: whichever came last of the working tree right after a successful merge
      (`factory-merge.tree`, else that merge's commit) and the working tree a run on the done task opened on
      (`factory-state.base`), else the tree of the working tree it started on (`baseTree`, the user's uncommitted
      edits included), else the commit it started on (a task older than `baseTree`)
    """
    started = _entry_data(entries, "factory-task")
    if not started:
        raise UnknownTask("The task's session has no factory-task entry")
    task = started[-1]
    if worktree:
        return _worktree_base(workspace, task)
    bases = []
    for entry in entries:
        data = entry.get("data") or {}
        if entry.get("customType") == "factory-merge" and data.get("ok") and data.get("commit"):
            bases.append(data.get("tree") or data["commit"])
        elif entry.get("customType") == "factory-state" and data.get("base"):
            bases.append(data["base"])
    return bases[-1] if bases else task.get("baseTree") or task["baseSha"]


def workspace_diff(workspace: Path, base: str) -> str:
    """Unified diff of *workspace* (index and work tree) against the commit or tree *base*, untracked files included.

    - An untracked file *base* holds (one the user had not committed when a task in the checkout started) is compared
      with its content there, any other untracked file is a new file
    - Reads only: nothing is added to the index
    """
    in_base = set(git_text(workspace, "ls-tree", "-r", "--name-only", "-z", base, timeout=DIFF_TIMEOUT_SECONDS).split("\0"))
    others = git_text(workspace, "ls-files", "--others", "--exclude-standard", "-z", timeout=DIFF_TIMEOUT_SECONDS)
    untracked = [file for file in others.split("\0") if file]
    # Git compares *base* with the paths of the index: an untracked file of *base* would show as deleted.
    left_out = [f":(exclude,literal){file}" for file in untracked if file in in_base]
    diff = git_text(workspace, "diff", *_DIFF_OPTIONS, base, "--", ".", *left_out, timeout=DIFF_TIMEOUT_SECONDS)
    for file in untracked:
        sides = (f"{base}:{file}", "--", file) if file in in_base else ("--no-index", "--", "/dev/null", file)
        # `--no-index` exits 1 when the files differ, which a new file always does.
        changed = run_git(workspace, "diff", *_DIFF_OPTIONS, *sides, timeout=DIFF_TIMEOUT_SECONDS, check=False)
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

    The archive is listed only when the unarchived listing misses one of them.
    """
    sessions = {session.session_id: session for session in await pi_host.list_sessions(cwd, archived=False)}
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


async def _ensure_checkout_free(store: FactoryStore, repo: str, at_rest: tuple[str, ...], apart_from: str | None) -> None:
    """`WorkspaceBusy` while a task that works in the checkout of *repo*, other than the task *apart_from*, is in none
    of the states *at_rest*."""
    rows = [row for row in await asyncio.to_thread(store.tasks, repo) if row.worktree is None and row.id != apart_from]
    if not rows:
        return
    sessions = await _sessions_of(rows, repo)

    def holds_checkout(row: TaskRow) -> bool:
        # A session pi-host no longer knows cannot work. One without a state yet has just started.
        session = sessions.get(row.session_id)
        return session is not None and (session.factory is None or session.factory.state not in at_rest)

    await asyncio.to_thread(ensure_checkout_free, Path(repo), [Path(row.cwd) for row in rows if holds_checkout(row)])


@contextlib.asynccontextmanager
async def _checkout_taken(
    store: FactoryStore, repo: str, at_rest: tuple[str, ...], apart_from: str | None = None
) -> AsyncIterator[None]:
    """Take the checkout of *repo* for a task about to work in it: the lock, held until the block ends, then the check
    that no other task works there (`_ensure_checkout_free`). `WorkspaceBusy` when either says no."""
    lock = store.checkout_lock(repo)
    await asyncio.to_thread(lock.__enter__)
    try:
        await _ensure_checkout_free(store, repo, at_rest, apart_from)
        yield
    finally:
        await asyncio.to_thread(lock.__exit__, None, None, None)


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
    """Start a task of a registered repo: its workspace, a coding session in it, `/factory start`, then its row.

    - A task in the checkout itself takes the checkout from the check to its row (`WorkspaceBusy`): a second start
      there sees either the lock or the row
    - A worktree task starts at the tip of its resolved source branch and merges into its target branch
      (`UnknownBranch`), a task in the checkout ignores both and names the checked-out branch in its start
    - `StartFailed` when the session shows no `factory-task` entry. Any failure after the workspace exists removes
      the session and the workspace again
    """
    settings = resolve_settings(await asyncio.to_thread(store.repo_settings, new_task.repo), new_task.overrides)
    repo = Path(new_task.repo)
    async with contextlib.AsyncExitStack() as held:
        if settings.worktree:
            source, target = await asyncio.to_thread(resolve_branches, repo, settings.source_branch, settings.target_branch)
        else:
            await held.enter_async_context(_checkout_taken(store, new_task.repo, FINISHED_STATES))
            source = target = await asyncio.to_thread(_checked_out_branch, repo)
        task_id = uuid4().hex
        name = workspace_name(new_task.title, task_id)
        workspace = await asyncio.to_thread(create_workspace, repo, name, settings.worktree, source)
        session_id: str | None = None
        try:
            session = await pi_host.open_session(str(workspace.path), None)
            session_id = session.session_id
            command = start_command(new_task.title, new_task.request, settings, source, target)
            disposition = await pi_host.prompt_session(session_id, command)
            if not await _started(session_id):
                raise StartFailed(
                    f"The session did not start the task (Pi answered {disposition!r}, no factory-task entry): the factory"
                    " extension is not loaded or refused the start (not a git repository, an unknown model)"
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
    *,
    starts_rounds: bool,
) -> Task:
    """Send *message* to the task's session, opened first when it is not live.

    A message that *starts_rounds* in the repo's checkout takes the checkout first: `WorkspaceBusy` while another task
    of it is running, merging or waiting for its approval.
    """
    row = await asyncio.to_thread(store.task, task_id)
    async with contextlib.AsyncExitStack() as held:
        if starts_rounds and row.worktree is None:
            await held.enter_async_context(_checkout_taken(store, row.repo, RESTING_STATES, apart_from=row.id))
        session = await pi_host.open_session(row.cwd, pi_host.ResumeById(session_id=row.session_id))
        await pi_host.prompt_session(row.session_id, message, streaming_behavior)
    return _with_session(row, session)


async def open_task_session(store: FactoryStore, task_id: str) -> Task:
    """Make the task's session live in pi-host without sending it anything, so a client can attach to it."""
    row = await asyncio.to_thread(store.task, task_id)
    return _with_session(row, await pi_host.open_session(row.cwd, pi_host.ResumeById(session_id=row.session_id)))


async def command_task(store: FactoryStore, task_id: str, command: TaskCommand) -> Task:
    """Approve, retry or cancel the task. The answer is the task as it was just before: its new state follows in the session."""
    return await _prompt(store, task_id, f"/factory {command}", starts_rounds=command == "retry")


async def message_task(store: FactoryStore, task_id: str, text: str) -> Task:
    """Hand the user's message to the task's coder: queued behind a running turn, new rounds for a task at rest.

    `TaskBusy` while it merges: the extension would drop the message.
    """
    row = await asyncio.to_thread(store.task, task_id)
    session = await _session_of(row)
    if session is not None and session.factory is not None and session.factory.state == "merging":
        raise TaskBusy("The task is merging, send the message when it has ended")
    return await _prompt(store, task_id, text, "followUp", starts_rounds=True)


async def task_diff(store: FactoryStore, task_id: str) -> TaskDiff:
    """The task's changes against its diff base. `UnknownTask` when its session has no `factory-task` entry."""
    row = await asyncio.to_thread(store.task, task_id)
    entries = (await pi_host.session_entries(row.session_id, custom_type=ENTRY_PREFIX)).entries
    base = await asyncio.to_thread(diff_base, Path(row.cwd), row.worktree is not None, entries)
    return TaskDiff(base=base, diff=await asyncio.to_thread(workspace_diff, Path(row.cwd), base))


async def delete_task(store: FactoryStore, task_id: str) -> None:
    """Remove the task for good: its preview, its session, its reviewer sessions, its worktree and branch, the refs that
    keep its snapshot trees (`refs/factory/<session id>/`) and its upstreams (`refs/factory-upstream/<branch>/`), its row.

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
    await asyncio.to_thread(_delete_task_refs, Path(row.repo), row.session_id, row.branch)
    await asyncio.to_thread(store.delete_task, task_id)


def _delete_task_refs(repo: Path, session_id: str, branch: str | None) -> None:
    prefixes = [f"refs/factory/{session_id}/", *([f"{UPSTREAM_REFS}/{branch}/"] if branch else [])]
    refs = git_text(repo, "for-each-ref", "--format=%(refname)", *prefixes).split()
    for ref in refs:
        git_text(repo, "update-ref", "-d", ref)
