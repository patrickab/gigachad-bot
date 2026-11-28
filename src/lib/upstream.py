"""Shared HTTP plumbing of the server-to-server adapters (LearnHarness, the factory engine, C4, the Jupyter Server).

- Each adapter owns its URL, headers and contract models, this module owns the request, the error mapping and the validation
- Failures are one of three exceptions the routes map to HTTP: unreachable 503, unknown shape 502, refusal passed through
"""

import asyncio
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from functools import cache
import os
import ssl

import httpx
from pydantic import TypeAdapter, ValidationError

# RFC 9110 connection-specific fields, never forwarded by a proxy. httpx sets `content-length` for the request itself.
HOP_BY_HOP_HEADERS = frozenset(
    {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"}
)
TIMEOUT_SECONDS = 10.0
_CLIENTS: list["UpstreamClient"] = []


class UpstreamUnavailable(Exception):
    """The upstream service does not answer."""


class UpstreamContractError(Exception):
    """The upstream service answered with a shape the adapter does not know."""


class UpstreamRequestFailed(Exception):
    """The upstream service refused the request, `status` and `message` are its own."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@cache
def tls_context() -> ssl.SSLContext:
    """The one TLS context of the process: building one per client reloads the CA bundle each time."""
    return httpx.create_ssl_context()


def upstream_url(env: str, default: str) -> str:
    """Base URL from environment variable *env*, else *default*, without a trailing slash."""
    return os.environ.get(env, default).rstrip("/")


def error_message(response: httpx.Response) -> str:
    """The upstream's own error text: `detail` or `error` of a JSON body, else the body's start, else the status."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200] or f"HTTP {response.status_code}"
    if isinstance(body, dict):
        return str(body.get("detail") or body.get("error") or body)
    return str(body)


def forwardable_headers(headers: Iterable[tuple[str, str]], dropped: frozenset[str] = frozenset()) -> list[tuple[str, str]]:
    """*headers* of a proxied message without the hop-by-hop fields and the lowercase names in *dropped*."""
    refused = HOP_BY_HOP_HEADERS | dropped
    return [(name, value) for name, value in headers if name.lower() not in refused]


@contextmanager
def unreachable(service: str) -> Iterator[None]:
    """Report a failed HTTP exchange inside the block as `UpstreamUnavailable` naming *service*."""
    try:
        yield
    except httpx.HTTPError as exc:
        raise UpstreamUnavailable(f"{service} does not answer: {exc}") from exc


def validate_answer[Result](result: type[Result], content: bytes, *, label: str, path: str) -> Result:
    """The JSON *content* of *label*'s answer to *path* as *result*, `UpstreamContractError` naming the first bad field otherwise."""
    try:
        return TypeAdapter(result).validate_json(content)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or "<root>"
        raise UpstreamContractError(f"{label} {path}: field `{field}` {first['msg'].lower()}") from exc


class UpstreamClient:
    """One upstream service: its URL (from *env*, else *default*), its *label* in errors and its default timeout."""

    def __init__(
        self,
        *,
        env: str,
        default: str,
        label: str,
        timeout: float | httpx.Timeout = TIMEOUT_SECONDS,
    ) -> None:
        self._env = env
        self._default = default
        self._label = label
        self._timeout = timeout
        # Tests swap in an `httpx.MockTransport`, None is the real network.
        self.transport: httpx.AsyncBaseTransport | None = None
        # One pooled client, built for a transport and an event loop: a swapped transport or a new loop (a test's) builds anew.
        self.built: tuple[tuple[object, object], httpx.AsyncClient] | None = None
        _CLIENTS.append(self)

    @property
    def url(self) -> str:
        """Base URL, read from the environment on every access."""
        return upstream_url(self._env, self._default)

    async def send(self, method: str, path: str, *, timeout: float | httpx.Timeout | None = None, **kwargs: object) -> httpx.Response:
        """Send one request (*kwargs* of `httpx.AsyncClient.request`) and return the answer, raising a refusal."""
        url = self.url
        with unreachable(f"{self._label} at {url}"):
            key = (self.transport, asyncio.get_running_loop())
            if self.built is None or self.built[0] != key:
                self.built = (key, httpx.AsyncClient(transport=self.transport, verify=tls_context()))
            timeout = self._timeout if timeout is None else timeout
            response = await self.built[1].request(method, f"{url}{path}", timeout=timeout, **kwargs)
        if response.is_error:
            raise UpstreamRequestFailed(response.status_code, error_message(response))
        return response

    async def request[Result](
        self,
        method: str,
        path: str,
        result: type[Result],
        *,
        json: object = None,
        params: dict[str, str | int] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Result:
        """Send one request and validate the JSON answer as *result*, *timeout* overriding the client's default."""
        response = await self.send(method, path, json=json, params=params, timeout=timeout)
        return validate_answer(result, response.content, label=self._label, path=path)


async def close_clients() -> None:
    """Close the pooled connections of every `UpstreamClient` (the app's shutdown)."""
    for client in _CLIENTS:
        if client.built is not None:
            await client.built[1].aclose()
            client.built = None
