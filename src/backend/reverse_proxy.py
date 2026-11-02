"""Reverse-proxy plumbing shared by the omp-web gate, the LearnHarness gate and the Jupyter proxy route.

- `streaming_response` relays an upstream answer unbuffered, hop-by-hop fields dropped (`lib.upstream.forwardable_headers`)
- `forward` is the whole request path of the Tailnet gates, standalone Starlette apps whose `app.state.client` has the upstream as
  `base_url`
"""

from collections.abc import Callable

import httpx
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response, StreamingResponse

from lib.upstream import forwardable_headers


def streaming_response(upstream: httpx.Response, drop: Callable[[str], bool] | None = None) -> StreamingResponse:
    """Relay the open streaming *upstream* answer, closed when the response is done. *drop* refuses more lowercase header names."""
    response = StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code, background=BackgroundTask(upstream.aclose))
    # Raw headers keep repeated fields (several `set-cookie`) that a mapping would merge.
    response.raw_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in forwardable_headers(upstream.headers.multi_items())
        if drop is None or not drop(name.lower())
    ]
    return response


def _gate_headers(request: Request) -> list[tuple[str, str]]:
    """Request headers for the upstream behind a Tailnet gate: the browser's `Host` kept, the scheme marked HTTPS."""
    return [
        *forwardable_headers(request.headers.items(), frozenset({"content-length", "x-forwarded-proto"})),
        ("x-forwarded-proto", "https"),
    ]


def _gate_target(request: Request) -> str:
    """Path and query exactly as the browser sent them, so encoded segments (`%2F` in file paths) stay encoded."""
    raw_path = request.scope.get("raw_path") or request.url.path.encode()
    query = request.scope.get("query_string", b"")
    return (raw_path + (b"?" + query if query else b"")).decode("latin-1")


async def forward(request: Request, label: str) -> Response:
    """Stream *request* to `request.app.state.client` and its answer back, 502 naming *label* when the upstream does not answer."""
    client: httpx.AsyncClient = request.app.state.client
    upstream_request = client.build_request(
        request.method, _gate_target(request), headers=_gate_headers(request), content=request.stream()
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return PlainTextResponse(f"{label} does not answer: {exc}\n", status_code=502)
    return streaming_response(upstream)
