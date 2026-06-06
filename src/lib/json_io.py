"""Shared I/O helpers for backend routes."""
import json
from pathlib import Path
from typing import Any


def safe_read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    """Read a JSON file at `path`, returning `default` if the file is missing or malformed.

    Always returns a dict (since this helper backs `project.json` / `projects-meta.json`
    / chat history files, all of which use object payloads at the top level).
    """
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    if not isinstance(data, dict):
        return default
    return data
