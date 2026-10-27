"""One way to run git: captured output, a timeout, and a locale-independent environment."""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 10


def run_git(
    directory: Path,
    *args: str,
    timeout: float | None = DEFAULT_TIMEOUT_SECONDS,
    text: bool = True,
    check: bool = True,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run ``git -C directory *args`` with `LC_ALL=C`, *env* added over the inherited environment."""
    return subprocess.run(
        ["git", "-C", str(directory), *args],
        capture_output=True, text=text, timeout=timeout, check=check,
        env={**os.environ, "LC_ALL": "C", **(env or {})},
    )


class GitError(RuntimeError):
    """A git command failed, with git's own message (or the timeout/OS error)."""


def git_text(
    directory: Path, *args: str, timeout: float | None = DEFAULT_TIMEOUT_SECONDS, env: Mapping[str, str] | None = None,
) -> str:
    """Stdout of ``git -C directory *args``. Any failure raises GitError carrying git's stderr (else stdout)."""
    try:
        return run_git(directory, *args, timeout=timeout, env=env).stdout
    except subprocess.CalledProcessError as exc:
        raise GitError((exc.stderr or exc.stdout or str(exc)).strip()) from exc
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise GitError(str(exc)) from exc


def git_ok(directory: Path, *args: str) -> bool:
    """Whether ``git -C directory *args`` succeeds, e.g. a ``rev-parse --verify`` probe."""
    try:
        git_text(directory, *args)
    except GitError:
        return False
    return True
