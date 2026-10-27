"""Client for the loopback LikeC4 service in ``src/c4`` (parsing, layout, surgical edits)."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from lib.architecture_workspace import ArchitectureError
from lib.upstream import UpstreamClient, UpstreamRequestFailed, UpstreamUnavailable

# A cold LikeC4 parse plus Graphviz layout takes ~1s; multi-operation batches re-parse per step.
_client = UpstreamClient(
    env="C4_SERVICE_URL", default="http://127.0.0.1:8011", label="architecture service", timeout=httpx.Timeout(60.0, connect=2.0),
)


class C4ServiceUnavailable(RuntimeError):
    """The C4 service is not running or failed internally."""


class C4Error(BaseModel):
    """A LikeC4 parse error, *line* zero-based."""

    message: str
    file: str
    line: int


class C4Model(BaseModel):
    """The model as the UI draws it; only its parse errors are read here, the rest passes through to the browser as is."""

    model_config = ConfigDict(extra="allow")

    errors: list[C4Error]


class RenderResult(BaseModel):
    model: C4Model
    snapshots: dict[str, str]


class ApplyResult(RenderResult):
    sources: dict[str, str]
    # Per operation, what it created: an element id or a new file's path.
    created: list[str | None]


async def _post[Result](path: str, payload: dict[str, Any], result: type[Result]) -> Result:
    try:
        return await _client.request("POST", path, result, json=payload)
    except UpstreamUnavailable as exc:
        raise C4ServiceUnavailable("Architecture service is unavailable") from exc
    except UpstreamRequestFailed as exc:
        if exc.status == 422:
            raise ArchitectureError(str(exc)) from exc
        raise C4ServiceUnavailable(f"Architecture service failed with HTTP {exc.status}") from exc


async def render(sources: dict[str, str], snapshots: dict[str, str]) -> RenderResult:
    """The parsed, laid-out model with saved positions applied, plus the snapshots its pinned views now have."""
    return await _post("/model", {"sources": sources, "snapshots": snapshots}, RenderResult)


async def apply_operations(
    sources: dict[str, str], snapshots: dict[str, str], ops: list[dict[str, Any]], home: str | None,
) -> ApplyResult:
    """Apply *ops* in order, new top-level elements into *home*.

    Returns the edited sources, every snapshot file the workspace should hold, the model, and what each operation created.
    """
    return await _post("/apply", {"sources": sources, "snapshots": snapshots, "ops": ops, "home": home}, ApplyResult)
