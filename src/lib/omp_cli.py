"""Run the `omp` command line and return what it printed."""

from __future__ import annotations

import shutil
import subprocess


class OmpNotFound(FileNotFoundError):
    """The `omp` binary is not on PATH."""


def run_omp(*arguments: str, timeout: float) -> str:
    """Stdout of `omp *arguments`. Raises `OmpNotFound`, or `OSError`/`SubprocessError` when it fails or times out."""
    executable = shutil.which("omp")
    if executable is None:
        raise OmpNotFound("omp binary not found")
    return subprocess.run([executable, *arguments], capture_output=True, text=True, timeout=timeout, check=True).stdout
