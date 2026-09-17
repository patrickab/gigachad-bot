"""Shared JSON I/O helpers for backend routes, over the DataStore abstraction."""

import json
from typing import Any

from lib.data_store import DataStorePath


def load_json(path: DataStorePath) -> dict[str, Any] | list[Any] | None:
    """Read JSON through a DataStore path."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_read_json(path: DataStorePath, default: dict[str, Any]) -> dict[str, Any]:
    """Read a JSON file at `path`, returning `default` if the file is missing or malformed.

    Always returns a dict (since this helper backs `project.json` / `projects-meta.json`
    / chat history files, all of which use object payloads at the top level).
    """
    data = load_json(path)
    if not isinstance(data, dict):
        return default
    return data


def safe_write_json(path: DataStorePath, data: dict[str, Any] | list[Any]) -> None:
    """Write `data` as JSON to `path`."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
