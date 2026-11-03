"""Host processes the backend starts never outlive it: each carries its owner, which stops what is left of them.

- `owned_environment` marks a child's environment with `GIGACHAD_PROCESS_OWNER=<pid>:<start time>` of the calling process.
  The child's own children (kernels, language servers, programs a notebook or Neovim runs) inherit the mark
- Children end with the backend by themselves where they can (see the Jupyter Server bootstrap and Neovim's PTY hangup).
  `stop_owned` at the end of the backend SIGKILLs what still carries its mark (a language server Neovim left running),
  `reap_orphans` at startup what carries the mark of a process that no longer runs (a backend killed without its shutdown)
- Neither touches the caller or its ancestors. Both read `/proc` (Linux), elsewhere they find nothing
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import signal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

OWNER_VARIABLE = "GIGACHAD_PROCESS_OWNER"
_PROC = Path("/proc")


def _status(pid: int) -> list[str] | None:
    """The fields of `/proc/<pid>/stat` after the process name (state first), None when it does not run."""
    try:
        stat = (_PROC / str(pid) / "stat").read_text()
    except OSError:
        return None
    return stat[stat.rindex(")") + 2 :].split()  # the name in parentheses may hold spaces and parentheses


def _identity(pid: int) -> str | None:
    """*pid* with its start time (clock ticks after boot), which a reused pid never repeats."""
    status = _status(pid)
    return None if status is None else f"{pid}:{status[19]}"


def owned_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """*environment* for a child of this process, marked as owned by it."""
    return {**environment, OWNER_VARIABLE: _identity(os.getpid()) or str(os.getpid())}


def _owner(pid: int) -> str | None:
    """The owner *pid* was started with, None when it is unmarked or another user's."""
    try:
        environment = (_PROC / str(pid) / "environ").read_bytes()
    except OSError:
        return None
    prefix = f"{OWNER_VARIABLE}=".encode()
    for entry in environment.split(b"\0"):
        if entry.startswith(prefix):
            return entry[len(prefix) :].decode(errors="replace")
    return None


def _outlived(owner: str) -> bool:
    """Whether the process *owner* names has ended, False for a mark that names none."""
    pid, _, started = owner.partition(":")
    return pid.isdigit() and started.isdigit() and _identity(int(pid)) != owner


def _lineage() -> set[int]:
    """This process and its ancestors."""
    lineage: set[int] = set()
    pid = os.getpid()
    while pid > 0 and pid not in lineage:
        lineage.add(pid)
        status = _status(pid)
        pid = int(status[1]) if status is not None else 0
    return lineage


def _kill_marked(doomed: Callable[[str], bool]) -> list[int]:
    """SIGKILL the processes whose mark *doomed* accepts, except this one and its ancestors, return their pids."""
    try:
        pids = [int(entry.name) for entry in _PROC.iterdir() if entry.name.isdigit()]
    except OSError:
        return []
    spared = _lineage()
    killed = []
    for pid in pids:
        if pid in spared:
            continue
        try:
            handle = os.pidfd_open(pid)  # pins the process: a pid reused after the check below is never signalled
        except OSError:
            continue
        try:
            owner = _owner(pid)
            if owner is not None and doomed(owner):
                with contextlib.suppress(ProcessLookupError):
                    signal.pidfd_send_signal(handle, signal.SIGKILL)
                    killed.append(pid)
        finally:
            os.close(handle)
    return killed


def stop_owned() -> list[int]:
    """SIGKILL what is left of this process's marked children and their descendants, for its end, return their pids."""
    identity = _identity(os.getpid())
    return _kill_marked(lambda owner: owner == identity) if identity is not None else []


def reap_orphans() -> list[int]:
    """SIGKILL the processes of this user whose owner no longer runs (left behind by an earlier backend), return their pids."""
    return _kill_marked(_outlived)
