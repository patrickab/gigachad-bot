"""Drop a merged factory task's code from its repository before the task is deleted.

- Consumes the core's `tasks.get` (repo, state, reason) and `tasks.timeline` (the `merge` entry's `commit`), and `git`
  in the repo's main checkout
- Provides `check_revert(repo, commit)` (a dry run through `git merge-tree`, the checkout untouched) and
  `revert_merge(repo, commit)`: `git revert --no-edit <commit>` when the commit is on the checked-out branch
  and not reverted yet. A failed revert is aborted, leaves the tree as it was and raises `RevertError`
- A task that merged nothing (`no_changes`, not done) has no commit and is skipped
"""

import subprocess
from typing import Any


class RevertError(Exception):
    """The merge could not be reverted; the task must not be deleted."""


def merged_commit(entries: list[dict[str, Any]]) -> str | None:
    """The commit of the last successful `merge` entry of a timeline, None when nothing merged."""
    for entry in reversed(entries):
        data = entry.get("data") or {}
        if entry.get("kind") == "merge" and data.get("ok") and data.get("commit"):
            return str(data["commit"])
    return None


def _git(repo: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=False)  # noqa: S603, S607


def _pending(repo: str, commit: str) -> bool:
    """Whether *commit* is on HEAD and not reverted yet."""
    if _git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        return False
    already = _git(repo, "log", "--fixed-strings", f"--grep=This reverts commit {commit}", "--format=%H", f"{commit}..HEAD")
    return not already.stdout.strip()


def check_revert(repo: str, commit: str) -> bool:
    """Whether reverting *commit* is pending and applies cleanly, decided without touching the checkout (a changed
    file reloads the dev server mid-request). `RevertError` on a conflict, False when there is nothing to revert."""
    if not _pending(repo, commit):
        return False
    merge = _git(repo, "merge-tree", "--write-tree", f"--merge-base={commit}", "HEAD", f"{commit}^")
    if merge.returncode != 0:
        raise RevertError(f"Could not revert {commit[:8]}: it conflicts with later changes")
    return True


def revert_merge(repo: str, commit: str) -> bool:
    """Revert *commit* in *repo*'s checkout. False when there was nothing to do (not on HEAD, or already reverted)."""
    if not _pending(repo, commit):
        return False
    result = _git(repo, "revert", "--no-edit", commit)
    if result.returncode != 0:
        _git(repo, "revert", "--abort")
        reason = (result.stderr or result.stdout).strip().splitlines()
        raise RevertError(f"Could not revert {commit[:8]}: {reason[-1] if reason else 'git revert failed'}")
    return True
