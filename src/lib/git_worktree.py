"""Clones and their worktrees: clone, list, create, remove, and delete branches.

- Clones live in `~/git/<repository>`
- The clone (a directory holding a `.git` folder) is the integration checkout, worktrees branch off it
- Worktrees live in `~/git/worktrees/<repository>/<name>`, the same layout as the `worktree` agent skill
- Listing asks git, so worktrees made by the skill or by hand show up too
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import shutil

from lib.git import DEFAULT_TIMEOUT_SECONDS, GitError, git_ok, git_text

CLONES_HOME = Path.home() / "git"
WORKTREES_HOME = CLONES_HOME / "worktrees"
PREFERRED_BASE_BRANCH = "dev"
_NETWORK_TIMEOUT_SECONDS = 120
_CLONE_TIMEOUT_SECONDS = 600

_HOST = r"[A-Za-z0-9][A-Za-z0-9.-]*"
_USER = r"(?:[\w.~-]+@)?"
_REPOSITORY_URL = re.compile(
    rf"(?:https://{_USER}{_HOST}(?::\d+)?/|ssh://{_USER}{_HOST}(?::\d+)?/|{_USER}{_HOST}:)"
    r"(?:[\w.~-]+/)*(?P<name>[A-Za-z0-9][\w.-]*?)(?:\.git)?/?",
    re.ASCII,
)


class WorktreeError(GitError):
    """A worktree operation refused, with the reason. Other git failures surface as plain GitError."""


class WorktreeDirty(WorktreeError):
    """The worktree holds uncommitted or untracked changes, removing it needs `force`."""


class BranchNotMerged(WorktreeError):
    """The branch has commits no other branch holds, deleting it needs `force`."""


@dataclass(frozen=True)
class Worktree:
    """*branch* is None for a detached HEAD, *is_main* marks the clone itself."""

    path: str
    branch: str | None
    is_main: bool


@dataclass(frozen=True)
class WorktreeStatus:
    """*ahead*/*behind* count commits against the repository's base branch."""

    changed_files: int
    ahead: int
    behind: int


def _ssh_command() -> str:
    """Skip `/etc/ssh` configs: the service's user namespace maps root to nobody, ssh then rejects them."""
    user_config = Path.home() / ".ssh" / "config"
    return f"ssh -F {shlex.quote(str(user_config) if user_config.is_file() else os.devnull)}"


def _git(directory: Path, *args: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """`git_text` that can reach remotes non-interactively."""
    env = {"GIT_SSH_COMMAND": os.environ.get("GIT_SSH_COMMAND", _ssh_command()), "GIT_TERMINAL_PROMPT": "0"}
    return git_text(directory, *args, timeout=timeout, env=env)


def is_clone(directory: Path) -> bool:
    """A repository's main checkout: it holds the `.git` folder its worktrees share."""
    return (directory / ".git").is_dir()


def list_worktrees(repository: Path) -> list[Worktree]:
    """The clone first, then its worktrees by path. Missing and prunable worktrees are skipped."""
    entries: list[dict[str, str]] = []
    for line in _git(repository, "worktree", "list", "--porcelain").splitlines():
        key, _, value = line.partition(" ")
        if key == "worktree":
            entries.append({"path": value})
        elif key in {"branch", "prunable"} and entries:
            entries[-1][key] = value.removeprefix("refs/heads/")
    main = repository.resolve()
    worktrees = []
    for entry in entries:
        path = Path(entry["path"]).resolve()
        if "prunable" not in entry and path.is_dir():
            worktrees.append(Worktree(path=str(path), branch=entry.get("branch"), is_main=path == main))
    return sorted(worktrees, key=lambda worktree: (not worktree.is_main, worktree.path))


def _has_origin(repository: Path) -> bool:
    return "origin" in _git(repository, "remote").split()


def _ref_exists(repository: Path, ref: str) -> bool:
    return git_ok(repository, "rev-parse", "--verify", "--quiet", ref)


def branches(repository: Path) -> list[str]:
    """Local branches plus those only on `origin`, by name."""
    names = _git(repository, "for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes/origin").split()
    short = {name.removeprefix("refs/heads/").removeprefix("refs/remotes/origin/") for name in names}
    short.discard("HEAD")
    return sorted(short, key=str.casefold)


def default_branch(repository: Path) -> str | None:
    """Origin's symbolic default branch, falling back to the main checkout's branch."""
    try:
        return _git(repository, "symbolic-ref", "--short", "refs/remotes/origin/HEAD").strip().removeprefix("origin/")
    except GitError:
        pass
    try:
        return _git(repository, "symbolic-ref", "--short", "HEAD").strip()
    except GitError:
        return None


def default_base_branch(repository: Path) -> str | None:
    """`dev` when it exists locally or on origin, else origin's default branch, else the clone's branch."""
    known = branches(repository)
    if PREFERRED_BASE_BRANCH in known:
        return PREFERRED_BASE_BRANCH
    return default_branch(repository) or (known[0] if known else None)


def status(worktree: Path, base: str | None) -> WorktreeStatus:
    """Uncommitted files of *worktree* and its commits ahead of / behind *base*."""
    changed_files = len(_git(worktree, "status", "--porcelain").splitlines())
    ahead = behind = 0
    if base and _ref_exists(worktree, f"refs/heads/{base}"):
        behind, ahead = map(int, _git(worktree, "rev-list", "--left-right", "--count", f"{base}...HEAD").split())
    return WorktreeStatus(changed_files=changed_files, ahead=ahead, behind=behind)


def worktree_name(name: str) -> str:
    """The skill's slug: lowercase, non-alphanumerics to single hyphens, trimmed."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def default_worktree(repository: Path) -> Worktree:
    """Reuse or check out the default branch without fetching or changing another checkout."""
    branch = default_branch(repository)
    if not branch:
        raise WorktreeError("Cannot infer the repository's default branch")
    checkout = next((worktree for worktree in list_worktrees(repository) if worktree.branch == branch), None)
    if checkout:
        return checkout
    name = worktree_name(branch)
    if not name:
        raise WorktreeError(f"Invalid worktree name: {branch!r}")
    _git(repository, "check-ref-format", "--branch", branch)
    path = WORKTREES_HOME / repository.name / name
    if path.exists() or path.is_symlink():
        raise WorktreeError(f"Directory already exists: {path}")
    local = _ref_exists(repository, f"refs/heads/{branch}")
    if not local and not _ref_exists(repository, f"refs/remotes/origin/{branch}"):
        raise WorktreeError(f"Unknown default branch: {branch}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorktreeError(str(exc)) from exc
    if local:
        _git(repository, "worktree", "add", str(path), branch)
    else:
        _git(repository, "worktree", "add", "--track", "-b", branch, str(path), f"refs/remotes/origin/{branch}")
    return Worktree(path=str(path.resolve()), branch=branch, is_main=False)


def _update_base(repository: Path, base: str) -> None:
    """Fetch, then fast-forward the local *base* branch to `origin/<base>` wherever it lives."""
    _git(repository, "fetch", "--all", "--prune", timeout=_NETWORK_TIMEOUT_SECONDS)
    if not _ref_exists(repository, f"refs/remotes/origin/{base}"):
        return
    checkout = next((worktree for worktree in list_worktrees(repository) if worktree.branch == base), None)
    if checkout:
        _git(Path(checkout.path), "merge", "--ff-only", f"origin/{base}")
    else:
        _git(repository, "fetch", "origin", f"{base}:{base}", timeout=_NETWORK_TIMEOUT_SECONDS)


def create_worktree(repository: Path, name: str, base: str) -> Worktree:
    """Pull *base*, then branch `<name>` off it into `~/git/worktrees/<repository>/<name>`.

    - *name* is slugged like the skill does, branch and folder share it
    - An existing branch or folder is refused, never reused
    """
    branch = worktree_name(name)
    if not branch:
        raise WorktreeError(f"Invalid worktree name: {name!r}")
    _git(repository, "check-ref-format", "--branch", branch)
    path = WORKTREES_HOME / repository.name / branch
    if _ref_exists(repository, f"refs/heads/{branch}"):
        raise WorktreeError(f"Branch already exists: {branch}")
    if path.exists():
        raise WorktreeError(f"Directory already exists: {path}")
    if _has_origin(repository):
        _update_base(repository, base)
    if not _ref_exists(repository, f"refs/heads/{base}"):
        raise WorktreeError(f"Unknown base branch: {base}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _git(repository, "worktree", "add", "-b", branch, str(path), base)
    return Worktree(path=str(path.resolve()), branch=branch, is_main=False)


def remove_worktree(repository: Path, path: str, *, force: bool = False) -> None:
    """Remove a linked worktree, keeping its branch. Raises WorktreeDirty when changes would be lost."""
    target = Path(path).resolve()
    worktree = next((worktree for worktree in list_worktrees(repository) if Path(worktree.path) == target), None)
    if worktree is None:
        raise WorktreeError(f"Not a worktree of {repository.name}: {path}")
    if worktree.is_main:
        raise WorktreeError("The clone itself cannot be removed")
    try:
        _git(repository, "worktree", "remove", *(["--force"] if force else []), worktree.path)
    except GitError as exc:
        if "contains modified or untracked files" in str(exc):
            raise WorktreeDirty(str(exc)) from exc
        raise


def delete_branch(repository: Path, branch: str, *, force: bool = False) -> None:
    """Delete a local branch, never the remote one. Raises BranchNotMerged when commits would be lost."""
    try:
        _git(repository, "branch", "-D" if force else "-d", "--", branch)
    except GitError as exc:
        if "not fully merged" in str(exc):
            raise BranchNotMerged(str(exc)) from exc
        raise


def clone_repository(url: str) -> Path:
    """Clone an https or ssh repository *url* into `~/git/<repository>`.

    - Accepts `https://host/owner/repo(.git)`, `ssh://git@host/owner/repo(.git)` and `git@host:owner/repo(.git)`
    - An existing folder is refused, never reused
    - A failed or timed out clone leaves no folder behind
    """
    url = url.strip()
    match = _REPOSITORY_URL.fullmatch(url)
    if match is None:
        raise WorktreeError(f"Not an https or ssh git repository link: {url!r}")
    target = CLONES_HOME / match["name"]
    if target.exists():
        raise WorktreeError(f"Directory already exists: {target}")
    CLONES_HOME.mkdir(parents=True, exist_ok=True)
    try:
        _git(CLONES_HOME, "clone", "--", url, match["name"], timeout=_CLONE_TIMEOUT_SECONDS)
    except GitError:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return target.resolve()
