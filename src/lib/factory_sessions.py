"""Copy a factory task agent's Pi session into the host's Pi sessions, so pi-host can open and continue it.

- The engine runs each phase as its own headless `pi` (on the host or in a sandbox), pi-host cannot attach to it
- The copy lives where Pi keeps the sessions of the task's worktree, resolved as Pi 1.5.2 does (`pi-host/src/host/local/paths.ts`:
  `PI_CODING_AGENT_SESSION_DIR`, else `PI_CODING_AGENT_DIR` or `~/.pi/agent`, plus `sessions/--<cwd>--`)
- Its header `cwd` becomes the worktree: a sandbox session names the sandbox's folder
- The copy follows the engine's session while it only grows (a later turn of the phase), and stays once it gained entries
  of its own (the user continued it), like a git fast-forward
"""

import json
import os
from pathlib import Path
import re


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def sessions_dir(cwd: str) -> Path:
    """The folder Pi keeps the sessions of *cwd* in."""
    flat = os.environ.get("PI_CODING_AGENT_SESSION_DIR")
    if flat:
        return _expand(flat)
    agent = _expand(os.environ["PI_CODING_AGENT_DIR"]) if os.environ.get("PI_CODING_AGENT_DIR") else Path.home() / ".pi" / "agent"
    name = re.sub(r"[/\\:]", "-", re.sub(r"^[/\\]", "", str(Path(cwd).resolve())))
    return agent / "sessions" / f"--{name}--"


def _rehomed(content: str, cwd: str) -> tuple[list[str], str]:
    """The session's lines with its header's `cwd` set to *cwd*, and its file name `<timestamp>_<id>.jsonl`."""
    lines = content.splitlines()
    if not lines:
        raise ValueError("The task's session is empty")
    header = json.loads(lines[0])
    if header.get("type") != "session" or not isinstance(header.get("id"), str) or not isinstance(header.get("timestamp"), str):
        raise ValueError("The task's session has no session header")
    if not re.fullmatch(r"[A-Za-z0-9-]+", header["id"]):
        raise ValueError("The task's session has an invalid id")
    header["cwd"] = cwd
    lines[0] = json.dumps(header, separators=(",", ":"))
    stamp = re.sub(r"[:.]", "-", header["timestamp"])
    return lines, f"{stamp}_{header['id']}.jsonl"


def import_session(content: str, cwd: str) -> Path:
    """Write the engine's session JSONL *content* as a Pi session of *cwd* and return its path (see the module rules)."""
    lines, name = _rehomed(content, cwd)
    target = sessions_dir(cwd) / name
    if target.is_file():
        existing = target.read_text(encoding="utf-8").splitlines()
        if len(existing) >= len(lines) or existing != lines[: len(existing)]:
            return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target
