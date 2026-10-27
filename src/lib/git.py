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
