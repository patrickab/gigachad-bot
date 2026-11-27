"""Drop a merged factory task's code from its repository before the task is deleted.

- Consumes the core's `TaskView.merged` (the commit the task's last merge landed) and `git` in the repo's main checkout
- Provides `check_revert(repo, commit)` (a dry run through `git merge-tree`, the checkout untouched) and
  `revert_merge(repo, commit)`: `git revert --no-edit <commit>` unless it is reverted already. A failed revert is
  aborted, leaves the tree as it was and raises `RevertError`
- A commit that is not on the checked-out branch raises `RevertError`: a task of a squash wave merged into its wave's
  branch, and only the wave's squash commit reached the target. Deleting the task would keep its code, so it is refused
"""

import subprocess


class RevertError(Exception):
    """The merge could not be reverted; the task must not be deleted."""


def _git(repo: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=False)  # noqa: S603, S607


def _pending(repo: str, commit: str) -> bool:
    """Whether *commit* is not reverted yet. `RevertError` when it is not on HEAD."""
    if _git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        raise RevertError(f"Could not revert {commit[:8]}: it is not on the checked-out branch (a squash wave landed it squashed)")
    already = _git(repo, "log", "--fixed-strings", f"--grep=This reverts commit {commit}", "--format=%H", f"{commit}..HEAD")
    return not already.stdout.strip()


def check_revert(repo: str, commit: str) -> bool:
    """Whether reverting *commit* is pending and applies cleanly, decided without touching the checkout (a changed
    file reloads the dev server mid-request). `RevertError` when it cannot, False when it is reverted already."""
    if not _pending(repo, commit):
        return False
    merge = _git(repo, "merge-tree", "--write-tree", f"--merge-base={commit}", "HEAD", f"{commit}^")
    if merge.returncode != 0:
        raise RevertError(f"Could not revert {commit[:8]}: it conflicts with later changes")
    return True


def revert_merge(repo: str, commit: str) -> bool:
    """Revert *commit* in *repo*'s checkout. False when it is reverted already."""
    if not _pending(repo, commit):
        return False
    result = _git(repo, "revert", "--no-edit", commit)
    if result.returncode != 0:
        _git(repo, "revert", "--abort")
        reason = (result.stderr or result.stdout).strip().splitlines()
        raise RevertError(f"Could not revert {commit[:8]}: {reason[-1] if reason else 'git revert failed'}")
    return True
