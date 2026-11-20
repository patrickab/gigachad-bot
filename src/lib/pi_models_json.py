"""Pi's `models.json`, which two syncs edit: Ollama's provider (`ollama_models`) and catalog models Pi lacks (`pi_catalog_models`).

- Consumes `$PI_CODING_AGENT_DIR` (default `~/.pi/agent`) like Pi resolves it
- Provides `update`, a locked read-change-write of the file's `providers` that keeps every other key, and
  `write_json`, the atomic write it uses

pi-host's gateway reloads `models.json` when it changes, Pi sessions pick it up like Pi's own `/model` does.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# Both syncs read, change and rewrite the whole file: one at a time, or one would drop the other's change.
# Reentrant, so a sync can hold it across `update` and its own bookkeeping.
lock = threading.RLock()


class ModelsJsonError(RuntimeError):
    """`models.json` cannot be read as Pi's config (left untouched) or cannot be written."""


def agent_dir() -> Path:
    """Pi's agent dir, `$PI_CODING_AGENT_DIR` or `~/.pi/agent`."""
    configured = os.environ.get("PI_CODING_AGENT_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".pi" / "agent"


def models_json_path() -> Path:
    return agent_dir() / "models.json"


def write_json(path: Path, data: object) -> None:
    """Replace *path* with *data* in one rename, keeping an existing file's mode (new files are owner-only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staging = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.write("\n")
        if path.exists():
            os.chmod(staging, path.stat().st_mode & 0o777)
        os.replace(staging, path)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise


def update(change: Callable[[dict[str, Any]], dict[str, Any] | None]) -> bool:
    """Replace the file's `providers` with `change(providers)` unless it returns None, True when the file was rewritten.

    A missing file reads as no providers. Raises `ModelsJsonError` for a file Pi could not read either (never overwritten)
    or a failed write.
    """
    path = models_json_path()
    with lock:
        try:
            config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError) as exc:
            raise ModelsJsonError(f"could not read Pi's {path}") from exc
        if not isinstance(config, dict) or not isinstance(config.get("providers", {}), dict):
            raise ModelsJsonError(f"Pi's {path} has no providers object")
        providers = change(config.get("providers", {}))
        if providers is None:
            return False
        try:
            write_json(path, {**config, "providers": providers})
        except OSError as exc:
            raise ModelsJsonError(f"could not write Pi's {path}") from exc
        return True
