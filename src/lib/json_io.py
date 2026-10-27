"""Shared JSON I/O helpers for backend routes, over the DataStore abstraction."""

import json
from typing import Any

from lib.data_store import DataStorePath


def load_json(path: DataStorePath) -> dict[str, Any] | list[Any] | None:
    """Read JSON through a DataStore path. Missing or malformed files return None.

    Skips an `exists()` pre-check: the store raises on a missing key, saving a round trip.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_read_json(path: DataStorePath, default: dict[str, Any]) -> dict[str, Any]:
    """Read a JSON object, returning `default` when the file is missing, malformed, or not an object."""
    data = load_json(path)
    if not isinstance(data, dict):
        return default
    return data


def dump_json(data: dict[str, Any] | list[Any]) -> str:
    """Serialize *data* in the stored format: 2-space indent, unescaped unicode, trailing newline."""
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def safe_write_json(path: DataStorePath, data: dict[str, Any] | list[Any]) -> None:
    """Write `data` as JSON to `path`."""
    path.write_text(dump_json(data))
