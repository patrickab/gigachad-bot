"""Where a factory task works: a fresh worktree of its repo or the current checkout, and the repo's `factory.yaml`.

- A worktree task gets branch `<name>` in `<root>/<name>` from `agent-worktree`: an overlay venv on the main checkout's
  `.venv`, and copy-on-write `node_modules` that arrive in the background. `hooks.after_create` then run in it
- The root is `GIGACHAD_FACTORY_WORKTREES`, default `~/git/worktrees/<repo name>/factory`
- A current-checkout task works in the repo itself, on the branch checked out there. Nothing is made, nothing is ever removed.
  One such task per checkout at a time: `ensure_checkout_free` is the rule, the caller knows the tasks that are not done
- `factory.yaml` is read from the repo: hooks and gates are argv lists with `{repo_path}` filled in as the main checkout
  (the repo, or the checkout it is a linked worktree of). `preview` is passed on as written, placeholders included
- Every function blocks on subprocesses: call them with `asyncio.to_thread`
"""

from collections.abc import Iterable
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import signal
import subprocess
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
import yaml

from lib.git import GitError, git_text

CONFIG_NAME = "factory.yaml"
# `git worktree add` plus the overlay venv, the packages are copied in the background.
CREATE_TIMEOUT_SECONDS = 120.0
REMOVE_TIMEOUT_SECONDS = 120.0
# A hook may install packages.
HOOK_TIMEOUT_SECONDS = 600.0
OUTPUT_TAIL_CHARACTERS = 2000


class WorkspaceError(RuntimeError):
    """A workspace command failed (the message names it and ends with its output tail), or `factory.yaml` is invalid."""


class WorkspaceBusy(RuntimeError):
    """The checkout already has a current-checkout task that is not done."""


class ConfigModel(BaseModel):
    """A part of `factory.yaml`. Keys this module does not know are ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True)


Argv = Annotated[list[str], Field(min_length=1)]


class Hooks(ConfigModel):
    """Commands run in the task's workspace (`after_create`, `after_code`) or in the main checkout (`after_merge`)."""

    after_create: list[Argv] = []
    after_code: list[Argv] = []
    after_merge: list[Argv] = []


class FactoryConfig(ConfigModel):
    hooks: Hooks = Hooks()
    gates: list[Argv] = []
    # Globs a task's diff must not touch.
    protected: list[str] = ["factory.yaml", ".factory/**"]
    preview: dict[str, Any] = {}


@dataclass(frozen=True)
class Workspace:
    """The directory a task works in. `branch` is the task's own branch, None in the current checkout."""

    path: Path
    branch: str | None
    worktree: bool


def _primary_checkout(checkout: Path) -> Path:
    """The main checkout of the repository *checkout* belongs to: itself, unless it is a linked worktree."""
    try:
        common_dir = git_text(checkout, "rev-parse", "--path-format=absolute", "--git-common-dir")
    except GitError as exc:
        raise WorkspaceError(f"{checkout} is not a git checkout: {exc}") from exc
    return Path(common_dir.strip()).parent


def load_factory_config(repo: Path) -> FactoryConfig:
    """The `factory.yaml` of *repo* with `{repo_path}` filled into hooks and gates. No file is an empty config."""
    file = repo / CONFIG_NAME
    try:
        text = file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return FactoryConfig()
    try:
        config = FactoryConfig.model_validate(yaml.safe_load(text) or {})
    except (yaml.YAMLError, ValidationError) as exc:
        raise WorkspaceError(f"{file} is invalid: {exc}") from exc
    repo_path = str(_primary_checkout(repo))

    def filled(commands: list[list[str]]) -> list[list[str]]:
        return [[argument.replace("{repo_path}", repo_path) for argument in argv] for argv in commands]

    hooks = Hooks(**{name: filled(commands) for name, commands in config.hooks.model_dump().items()})
    return config.model_copy(update={"hooks": hooks, "gates": filled(config.gates)})


def _run(argv: list[str], cwd: Path, timeout: float) -> str:
    """Stdout of *argv* run in *cwd*. A non-zero exit, a timeout or a command that does not start raises WorkspaceError.

    The command gets its own process group, so a timeout also ends what it started (`npm ci` and its children).
    """
    command = shlex.join(argv)
    try:
        process = subprocess.Popen(
            argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise WorkspaceError(f"{command} did not start: {exc}") from exc
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise WorkspaceError(f"{command} timed out after {timeout:g} s\n{_tail(stdout, stderr)}".rstrip())
    if process.returncode != 0:
        raise WorkspaceError(f"{command} failed with exit code {process.returncode}\n{_tail(stdout, stderr)}".rstrip())
    return stdout


def _tail(stdout: str, stderr: str) -> str:
    output = "\n".join(part.strip() for part in (stdout, stderr) if part.strip())
    return output[-OUTPUT_TAIL_CHARACTERS:]


def _worktree_root(primary: Path) -> Path:
    override = os.environ.get("GIGACHAD_FACTORY_WORKTREES")
    return Path(override).expanduser() if override else Path.home() / "git" / "worktrees" / primary.name / "factory"


def create_workspace(repo: Path, name: str, worktree: bool, base: str) -> Workspace:
    """The workspace of a new task of *repo*: branch and worktree *name* from the tip of the local branch *base* with
    `hooks.after_create` run, else *repo* itself (*base* unused). Nothing uncommitted in *repo* enters a worktree.

    A failing hook removes the worktree and its branch again. A name that is taken (branch or directory) is refused.
    """
    if not worktree:
        return Workspace(path=repo, branch=None, worktree=False)
    config = load_factory_config(repo)
    target = _worktree_root(_primary_checkout(repo)) / name
    create = ["agent-worktree", "create", name, "--base", base, "--path", str(target), "--background"]
    created = _run(create, repo, CREATE_TIMEOUT_SECONDS)
    workspace = Workspace(path=Path(created.splitlines()[-1]), branch=name, worktree=True)
    try:
        for argv in config.hooks.after_create:
            _run(argv, workspace.path, HOOK_TIMEOUT_SECONDS)
    except WorkspaceError:
        remove_workspace(workspace)
        raise
    return workspace


def remove_workspace(workspace: Workspace) -> None:
    """Remove a task's worktree and branch, uncommitted work included. The current checkout is never removed.

    A worktree whose directory is already gone is left as it is: its branch, if it still exists, stays.
    """
    if not workspace.worktree or not workspace.path.exists():
        return
    primary = _primary_checkout(workspace.path)
    _run(["agent-worktree", "remove", str(workspace.path), "--force", "--delete-branch"], primary, REMOVE_TIMEOUT_SECONDS)


def ensure_checkout_free(repo: Path, active: Iterable[Path]) -> None:
    """Raise WorkspaceBusy when *repo* is among *active*, the workspace paths of the tasks that are not done."""
    checkout = repo.resolve()
    if any(path.resolve() == checkout for path in active):
        raise WorkspaceBusy(f"{repo} already has a task working in the checkout")
