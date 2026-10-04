"""Client for the loopback LikeC4 service in ``src/c4`` (parsing, layout, surgical edits)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from lib.architecture_workspace import ArchitectureError

C4_SERVICE_URL = os.environ.get("C4_SERVICE_URL", "http://127.0.0.1:8011")
# A cold LikeC4 parse plus Graphviz layout takes ~1s; multi-operation batches re-parse per step.
_TIMEOUT = httpx.Timeout(60.0, connect=2.0)


class C4ServiceUnavailable(RuntimeError):
    """The C4 service is not running or failed internally."""


async def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(base_url=C4_SERVICE_URL, timeout=_TIMEOUT) as client:
            response = await client.post(path, json=payload)
    except httpx.HTTPError as exc:
        raise C4ServiceUnavailable("Architecture service is unavailable") from exc
    if response.status_code == 422:
        raise ArchitectureError(response.json().get("error", "Rejected by the architecture service"))
    if response.status_code != 200:
        raise C4ServiceUnavailable(f"Architecture service failed with HTTP {response.status_code}")
    return response.json()


async def render(sources: dict[str, str], snapshots: dict[str, str]) -> dict[str, Any]:
    """The parsed, laid-out ``model`` (views, elements, kinds, parse errors) with
    saved positions applied, plus the ``snapshots`` its pinned views now have."""
    return await _post("/model", {"sources": sources, "snapshots": snapshots})


async def apply_operations(sources: dict[str, str], snapshots: dict[str, str], ops: list[dict[str, Any]], home: str | None) -> dict[str, Any]:
    """Apply *ops* in order, new top-level elements into *home*; returns the edited
    ``sources``, every ``snapshots`` file the workspace should hold, the ``model``,
    and per operation what it ``created``."""
    return await _post("/apply", {"sources": sources, "snapshots": snapshots, "ops": ops, "home": home})
