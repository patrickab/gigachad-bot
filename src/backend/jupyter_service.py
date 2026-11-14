"""The Jupyter service: the one process that runs the Jupyter Server, its kernels and the map from notebook to kernel.

- Every backend worker reaches it through `lib.jupyter_service` over a Unix socket (`GIGACHAD_JUPYTER_SOCKET`), so all workers
  share one server and one kernel map. Workers check the browser (host access, trusted origin, file roots), this service trusts its
  socket: it lives in a folder only the host user can enter, like the server's token in this process's memory
- `run-backend.sh` starts it before the workers, one process. Kernels end with it, a backend reload or worker restart keeps them
- Routes mirror `lib.jupyter.JupyterHost`, failures answer like `call_upstream` maps them: unreachable 503, unknown shape 502,
  refusal passed through

Run: `uvicorn backend.jupyter_service:app --app-dir src --uds "$GIGACHAD_JUPYTER_SOCKET"`
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket
from pydantic import BaseModel

from backend.reverse_proxy import relay_websocket, streaming_response
from backend.routes.upstream import call_upstream
from lib import jupyter
from lib.jupyter import KernelRef, KernelSpecList
from lib.process_owner import stop_owned
from lib.upstream import UpstreamUnavailable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = logging.getLogger(__name__)

_TRY_AGAIN_LATER = 1013
_PROXIED_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"]


class StartKernel(BaseModel):
    path: str
    name: str
    directory: str


class MoveKernel(BaseModel):
    old: str
    new: str


class KernelsUnder(BaseModel):
    path: str


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    await jupyter.shutdown()
    if stopped := stop_owned():  # what the Jupyter Server's kernels left running
        log.info("Stopped %d processes the Jupyter Server left running: %s", len(stopped), stopped)


app = FastAPI(title="gigachad-bot jupyter", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/kernelspecs", response_model=KernelSpecList)
async def kernelspecs(path: str = Query(...)) -> KernelSpecList:
    return await call_upstream(asyncio.to_thread(jupyter.host().kernelspecs, path))


@app.post("/kernel", response_model=KernelRef)
async def start_kernel(request: StartKernel) -> KernelRef:
    return await call_upstream(asyncio.to_thread(jupyter.host().start_kernel, request.path, request.name, request.directory))


@app.delete("/kernel", status_code=204)
async def shutdown_kernel(path: str = Query(...)) -> Response:
    await call_upstream(asyncio.to_thread(jupyter.host().shutdown_kernel, path))
    return Response(status_code=204)


@app.post("/kernels/move", status_code=204)
async def move_kernel(request: MoveKernel) -> Response:
    jupyter.host().move_kernel(request.old, request.new)
    return Response(status_code=204)


@app.post("/kernels/shutdown-under", status_code=204)
async def shutdown_kernels_under(request: KernelsUnder) -> Response:
    await asyncio.to_thread(jupyter.host().shutdown_kernels_under, request.path)
    return Response(status_code=204)


@app.post("/sessions/check", status_code=204)
async def check_session(request: Request, method: str = Query(...)) -> Response:
    await call_upstream(asyncio.to_thread(jupyter.host().check_session, method, await request.body()))
    return Response(status_code=204)


def _has_body(request: Request) -> bool:
    return "transfer-encoding" in request.headers or request.headers.get("content-length", "0") != "0"


@app.api_route("/server/{path:path}", methods=_PROXIED_METHODS)
async def proxy_http(path: str, request: Request) -> Response:
    if not (jupyter.is_proxied_path(path) and jupyter.allows_method(request.method, path)):
        raise HTTPException(status_code=404, detail="Not found")
    body = request.stream() if _has_body(request) else None
    upstream = await call_upstream(
        jupyter.host().open_http(request.method, path, request.query_params.multi_items(), request.headers.items(), body)
    )
    return streaming_response(upstream, jupyter.drops_response_header)


@app.websocket("/server/{path:path}")
async def proxy_websocket(websocket: WebSocket, path: str) -> None:
    if not jupyter.is_proxied_path(path):
        await websocket.close()
        return
    try:
        upstream = await jupyter.host().open_websocket(
            path, websocket.query_params.multi_items(), websocket.scope.get("subprotocols", [])
        )
    except UpstreamUnavailable as exc:
        log.warning("%s", exc)
        await websocket.close(code=_TRY_AGAIN_LATER)
        return
    await relay_websocket(websocket, upstream)
