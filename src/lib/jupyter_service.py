"""The backend workers' side of the Jupyter service (`backend.jupyter_service`), over its Unix socket `GIGACHAD_JUPYTER_SOCKET`.

- `RemoteJupyterHost` has the methods of `lib.jupyter.JupyterHost` the routes call, so every worker shares the service's one server
  and kernel map. Its failures are the same `lib.upstream` exceptions, the service's 503 and 502 come back as such
- A service that does not answer has no kernels (they end with it): moving or shutting them down after a rename or delete is
  then nothing
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING
from urllib.parse import quote, urlencode

import httpx

from lib.jupyter import KernelRef, KernelSpecList, browser_request_headers
from lib.upstream import (
    UpstreamContractError,
    UpstreamRequestFailed,
    UpstreamUnavailable,
    error_message,
    unreachable,
    validate_answer,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Sequence

    from websockets.asyncio.client import ClientConnection

    from lib.jupyter import JupyterHost

log = logging.getLogger(__name__)

SOCKET_VARIABLE = "GIGACHAD_JUPYTER_SOCKET"
_LABEL = "Jupyter service"
_ORIGIN = "http://jupyter-service"
_REQUEST_SECONDS = 60.0
# Starting a kernel may first start the server (40 s) and install `ipykernel` into a venv (180 s).
_START_KERNEL_SECONDS = 240.0


def _socket() -> str:
    path = os.environ.get(SOCKET_VARIABLE)
    if not path:
        raise UpstreamUnavailable(f"{_LABEL} is not configured: {SOCKET_VARIABLE} is unset")
    return path


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 503:
        raise UpstreamUnavailable(error_message(response))
    if response.status_code == 502:
        raise UpstreamContractError(error_message(response))
    if response.is_error:
        raise UpstreamRequestFailed(response.status_code, error_message(response))


class RemoteJupyterHost:
    """`JupyterHost` in the Jupyter service, safe to call from any thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    def _sync_client(self) -> httpx.Client:
        with self._lock:
            if self._client is None:
                self._client = httpx.Client(transport=httpx.HTTPTransport(uds=_socket()), base_url=_ORIGIN, timeout=_REQUEST_SECONDS)
            return self._client

    def _call(self, method: str, path: str, *, timeout: float = _REQUEST_SECONDS, **kwargs: object) -> httpx.Response:
        with unreachable(_LABEL):
            response = self._sync_client().request(method, path, timeout=timeout, **kwargs)
        _raise_for_status(response)
        return response

    def kernelspecs(self, path: str) -> KernelSpecList:
        response = self._call("GET", "/kernelspecs", params={"path": path})
        return validate_answer(KernelSpecList, response.content, label=_LABEL, path="/kernelspecs")

    def start_kernel(self, path: str, name: str, directory: str) -> KernelRef:
        payload = {"path": path, "name": name, "directory": directory}
        response = self._call("POST", "/kernel", json=payload, timeout=_START_KERNEL_SECONDS)
        return validate_answer(KernelRef, response.content, label=_LABEL, path="/kernel")

    def shutdown_kernel(self, path: str) -> None:
        self._call("DELETE", "/kernel", params={"path": path})

    def move_kernel(self, old: str, new: str) -> None:
        try:
            self._call("POST", "/kernels/move", json={"old": old, "new": new})
        except UpstreamUnavailable as exc:
            log.warning("No kernel moved from %s: %s", old, exc)

    def shutdown_kernels_under(self, path: str) -> None:
        try:
            self._call("POST", "/kernels/shutdown-under", json={"path": path})
        except UpstreamUnavailable as exc:
            log.warning("No kernels shut down under %s: %s", path, exc)

    def check_session(self, method: str, body: bytes) -> None:
        self._call("POST", "/sessions/check", params={"method": method}, content=body)

    async def open_http(
        self,
        method: str,
        path: str,
        query: Iterable[tuple[str, str]],
        headers: Iterable[tuple[str, str]],
        body: AsyncIterator[bytes] | None,
    ) -> httpx.Response:
        """The service's streaming answer to the browser's request for upstream *path*, the caller closes it."""
        if self._async_client is None:
            transport = httpx.AsyncHTTPTransport(uds=_socket())
            timeout = httpx.Timeout(_REQUEST_SECONDS, read=None)
            self._async_client = httpx.AsyncClient(transport=transport, base_url=_ORIGIN, timeout=timeout)
        request = self._async_client.build_request(
            method,
            _target(path, query),
            headers=browser_request_headers(headers),
            content=body,
        )
        with unreachable(_LABEL):
            response = await self._async_client.send(request, stream=True)
        if response.status_code in {502, 503}:
            await response.aread()
            await response.aclose()
            _raise_for_status(response)
        return response

    async def open_websocket(self, path: str, query: Iterable[tuple[str, str]], subprotocols: Sequence[str]) -> ClientConnection:
        """The service's WebSocket for upstream *path*, `UpstreamUnavailable` when it refuses or cannot be reached."""
        from websockets.asyncio.client import unix_connect
        from websockets.exceptions import WebSocketException

        try:
            return await unix_connect(
                _socket(),
                f"ws://jupyter-service{_target(path, query)}",
                subprotocols=[*subprotocols] or None,  # type: ignore[list-item]
                max_size=None,
                compression=None,
                ping_interval=None,
            )
        except (OSError, WebSocketException, TimeoutError) as exc:
            raise UpstreamUnavailable(f"{_LABEL} refused the connection: {exc}") from exc

    async def aclose(self) -> None:
        """Close the clients, for the end of the worker."""
        client, self._async_client = self._async_client, None
        if client is not None:
            await client.aclose()
        with self._lock:
            sync_client, self._client = self._client, None
        if sync_client is not None:
            sync_client.close()


def _target(path: str, query: Iterable[tuple[str, str]]) -> str:
    items = list(query)
    return f"/server/{quote(path, safe='/')}" + (f"?{urlencode(items)}" if items else "")


_host: RemoteJupyterHost | JupyterHost | None = None
_host_lock = threading.Lock()


def host() -> RemoteJupyterHost | JupyterHost:
    """The worker's way to the Jupyter service (tests put an in-process `JupyterHost` in its place)."""
    global _host
    with _host_lock:
        if _host is None:
            _host = RemoteJupyterHost()
        return _host


async def aclose() -> None:
    """Close the worker's clients to the service."""
    with _host_lock:
        current = _host
    if isinstance(current, RemoteJupyterHost):
        await current.aclose()
