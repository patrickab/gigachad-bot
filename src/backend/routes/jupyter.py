"""Jupyter notebooks for logins with host access (`backend.identity.HOST_CAPABILITY`), kernels run as the host user.

- `GET /api/jupyter/kernelspecs?path=<abs .ipynb>` → `{"default", "specs": [{"name", "display_name", "language", "needs_ipykernel"}]}`
- `POST /api/jupyter/kernel {"path", "name"}` → `{"kernel_id", "name"}`: the notebook's kernel, started in its folder
- `DELETE /api/jupyter/kernel?path=` → 204, shuts the notebook's kernel down
- `/api/jupyter/server/{path}` (HTTP and WebSocket): the Jupyter Server API (kernels, kernelspecs, sessions, events), the
  browser's own credentials are replaced by the server's token. Other upstream paths answer 404, and changes are limited to a
  kernel's restart, interrupt and end plus the sessions of kernels started above (kernels start only through `POST /kernel`).

Every HTTP route refuses requests that no page of the app sent (`backend.websocket_origin.require_trusted_request`): changes need
a trusted `Origin`, reads may not be cross-site.

How the server runs, what the specs are and where kernels live is `lib.jupyter`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, WebSocket
from pydantic import BaseModel

from backend.identity import RequestIdentity, get_request_identity, host_access_only
from backend.reverse_proxy import relay_websocket, streaming_response
from backend.routes.deps import get_file_roots
from backend.routes.upstream import call_upstream
from backend.websocket_origin import POLICY_VIOLATION, is_allowed_websocket, require_trusted_request
from lib import jupyter
from lib.file_roots import FileRoots
from lib.jupyter import JupyterHost, KernelRef, KernelSpecList
from lib.upstream import UpstreamRequestFailed, UpstreamUnavailable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jupyter", tags=["jupyter"])
_host_only = host_access_only("Jupyter")
_trusted = Depends(require_trusted_request)

_TRY_AGAIN_LATER = 1013
_PROXIED_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"]
_MAX_SESSION_BODY = 64 * 1024


class StartKernelRequest(BaseModel):
    path: str
    name: str


class NotebookDirectory(BaseModel):
    path: str


def get_jupyter_host() -> JupyterHost:
    return jupyter.host()


def _allowed_notebook(roots: FileRoots, path: str) -> str:
    """The resolved path of the notebook *path* when the file roots reach it, else a 404 like any other path outside them."""
    try:
        notebook = jupyter.notebook_path(path)
    except UpstreamRequestFailed as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    if not roots.contains(notebook):
        raise HTTPException(status_code=404, detail=f"Path outside file roots: {path}")
    return str(notebook)


def _working_directory(roots: FileRoots, notebook: str) -> str:
    """Where the notebook *notebook* works (kernel and chat): the mounted root holding it, so repo-relative paths resolve, else its folder."""
    with contextlib.suppress(ValueError):
        return str(roots.root_of(notebook))
    return str(Path(notebook).parent)


@router.get("/directory", response_model=NotebookDirectory, dependencies=[_host_only, _trusted])
async def notebook_directory(path: str = Query(...), roots: FileRoots = Depends(get_file_roots)) -> NotebookDirectory:
    return NotebookDirectory(path=_working_directory(roots, _allowed_notebook(roots, path)))


@router.get("/kernelspecs", response_model=KernelSpecList, dependencies=[_host_only, _trusted])
async def list_kernelspecs(
    path: str = Query(...), host: JupyterHost = Depends(get_jupyter_host), roots: FileRoots = Depends(get_file_roots)
) -> KernelSpecList:
    return await call_upstream(asyncio.to_thread(host.kernelspecs, _allowed_notebook(roots, path)))


@router.post("/kernel", response_model=KernelRef, dependencies=[_host_only, _trusted])
async def start_kernel(
    req: StartKernelRequest, host: JupyterHost = Depends(get_jupyter_host), roots: FileRoots = Depends(get_file_roots)
) -> KernelRef:
    notebook = _allowed_notebook(roots, req.path)
    return await call_upstream(asyncio.to_thread(host.start_kernel, notebook, req.name, _working_directory(roots, notebook)))


@router.delete("/kernel", status_code=204, dependencies=[_host_only, _trusted])
async def shutdown_kernel(
    path: str = Query(...), host: JupyterHost = Depends(get_jupyter_host), roots: FileRoots = Depends(get_file_roots)
) -> Response:
    await call_upstream(asyncio.to_thread(host.shutdown_kernel, _allowed_notebook(roots, path)))
    return Response(status_code=204)


def _denied(path: str) -> HTTPException:
    log.debug("Jupyter proxy denied %s", path)
    return HTTPException(status_code=404, detail="Not found")


def _has_body(request: Request) -> bool:
    """Whether the browser sent a request body, an empty one is not forwarded as a chunked stream."""
    return "transfer-encoding" in request.headers or request.headers.get("content-length", "0") != "0"


async def _session_body(request: Request) -> bytes:
    """The whole body of a session request, which is read to be checked before it is forwarded."""
    body = b""
    async for chunk in request.stream():
        body += chunk
        if len(body) > _MAX_SESSION_BODY:
            raise HTTPException(status_code=413, detail="Session request too large")
    return body


async def _replay(body: bytes) -> AsyncIterator[bytes]:
    yield body


@router.api_route("/server/{path:path}", methods=_PROXIED_METHODS, dependencies=[_host_only, _trusted])
async def proxy_http(path: str, request: Request, host: JupyterHost = Depends(get_jupyter_host)) -> Response:
    if not jupyter.is_proxied_path(path):
        raise _denied(path)
    if not jupyter.allows_method(request.method, path):
        log.debug("Jupyter proxy denied %s %s", request.method, path)
        raise HTTPException(status_code=405, detail="Method not allowed")
    body = request.stream() if _has_body(request) else None
    if request.method in {"POST", "PATCH"} and path.startswith("api/sessions"):
        content = await _session_body(request)
        await call_upstream(asyncio.to_thread(host.check_session, request.method, content))
        body = _replay(content)
    upstream = await call_upstream(
        host.open_http(request.method, path, request.query_params.multi_items(), request.headers.items(), body)
    )
    return streaming_response(upstream, jupyter.drops_response_header)


@router.websocket("/server/{path:path}")
async def proxy_websocket(
    websocket: WebSocket,
    path: str,
    identity: RequestIdentity = Depends(get_request_identity),
    host: JupyterHost = Depends(get_jupyter_host),
) -> None:
    if not is_allowed_websocket(websocket, identity):
        await websocket.close(code=POLICY_VIOLATION)
        return
    if not jupyter.is_proxied_path(path):
        log.debug("Jupyter proxy denied %s", path)
        await websocket.close(code=POLICY_VIOLATION)
        return
    try:
        upstream = await host.open_websocket(path, websocket.query_params.multi_items(), websocket.scope.get("subprotocols", []))
    except UpstreamUnavailable as exc:
        log.warning("%s", exc)
        await websocket.close(code=_TRY_AGAIN_LATER)
        return
    await relay_websocket(websocket, upstream)
