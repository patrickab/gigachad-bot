"""Shared HTTP plumbing of the server-to-server adapters (omp-web, the factory engine).

- Each adapter owns its URL, headers and contract models, this module owns the request, the error mapping and the validation
- Failures are one of three exceptions the routes map to HTTP: unreachable 503, unknown shape 502, refusal passed through
"""

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
    timeout: float = TIMEOUT_SECONDS,
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
