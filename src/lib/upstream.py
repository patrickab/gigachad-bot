"""Shared HTTP plumbing of the server-to-server adapters (omp-web, the factory engine).

- Each adapter owns its URL, headers and contract models, this module owns the request, the error mapping and the validation
- Failures are one of three exceptions the routes map to HTTP: unreachable 503, unknown shape 502, refusal passed through
"""

from collections.abc import Callable
import os

import httpx
from pydantic import TypeAdapter, ValidationError

TIMEOUT_SECONDS = 10.0


class UpstreamUnavailable(Exception):
    """The upstream service does not answer."""


class UpstreamContractError(Exception):
    """The upstream service answered with a shape the adapter does not know."""


class UpstreamRequestFailed(Exception):
    """The upstream service refused the request, `status` and `message` are its own."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def upstream_url(env: str, default: str) -> str:
    """Base URL from environment variable *env*, else *default*, without a trailing slash."""
    return os.environ.get(env, default).rstrip("/")


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200] or f"HTTP {response.status_code}"
    if isinstance(body, dict):
        return str(body.get("detail") or body.get("error") or body)
    return str(body)


async def request_json[Result](
    base_url: str,
    method: str,
    path: str,
    result: type[Result],
    *,
    label: str,
    headers: dict[str, str] | None = None,
    json: object = None,
    params: dict[str, str | int] | None = None,
    timeout: float | httpx.Timeout = TIMEOUT_SECONDS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Result:
    """Send one request and validate the JSON answer as *result*. *label* names the service in errors, *transport* is for tests."""
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout, headers=headers, transport=transport) as client:
            response = await client.request(method, path, json=json, params=params)
    except httpx.HTTPError as exc:
        raise UpstreamUnavailable(f"{label} at {base_url} does not answer: {exc}") from exc
    if response.is_error:
        raise UpstreamRequestFailed(response.status_code, _error_message(response))
    try:
        return TypeAdapter(result).validate_json(response.content)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or "<root>"
        raise UpstreamContractError(f"{label} {path}: field `{field}` {first['msg'].lower()}") from exc


class UpstreamClient:
    """One upstream service: its URL (from *env*, else *default*), its *label* in errors, its headers and default timeout."""

    def __init__(
        self,
        *,
        env: str,
        default: str,
        label: str,
        headers: Callable[[], dict[str, str]] | None = None,
        timeout: float | httpx.Timeout = TIMEOUT_SECONDS,
    ) -> None:
        self._env = env
        self._default = default
        self._label = label
        self._headers = headers
        self._timeout = timeout
        # Tests swap in an `httpx.MockTransport`, None is the real network.
        self.transport: httpx.AsyncBaseTransport | None = None

    @property
    def url(self) -> str:
        """Base URL, read from the environment on every access."""
        return upstream_url(self._env, self._default)

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
        return await request_json(
            self.url,
            method,
            path,
            result,
            label=self._label,
            headers=self._headers() if self._headers else None,
            json=json,
            params=params,
            timeout=self._timeout if timeout is None else timeout,
            transport=self.transport,
        )
