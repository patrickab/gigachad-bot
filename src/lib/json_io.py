"""Shared I/O helpers for backend routes."""

import json
from pathlib import Path
import tempfile
from typing import Any

from lib.data_store import DataStorePath


def load_json(path: Path | DataStorePath) -> dict[str, Any] | list[Any] | None:
    """Read JSON through a DataStore path or the local compatibility path."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_read_json(path: Path | DataStorePath, default: dict[str, Any]) -> dict[str, Any]:
    """Read a JSON file at `path`, returning `default` if the file is missing or malformed.

    Always returns a dict (since this helper backs `project.json` / `projects-meta.json`
    / chat history files, all of which use object payloads at the top level).
    """
    data = load_json(path)
    if not isinstance(data, dict):
        return default
    return data


def safe_write_json(path: Path | DataStorePath, data: dict[str, Any] | list[Any]) -> None:
    """Write `data` as JSON to `path`, creating parent directories if needed.

    Uses atomic write (write to temp file, then rename) for robustness.
    """
    content = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if isinstance(path, DataStorePath):
        path.write_text(content)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        with open(fd, "w", encoding="utf-8") as f:
            f.write(content)
        tmp_path = Path(tmp)
        tmp_path.replace(path)
    except OSError:
        path.write_text(content, encoding="utf-8")
