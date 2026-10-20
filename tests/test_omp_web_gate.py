"""The omp-web gate forwards only Tailnet logins granted the host capability, and forwards them so omp-web's same-origin guard still holds."""

import httpx
import pytest
from starlette.testclient import TestClient

from backend import omp_web_gate
from backend.identity import HOST_CAPABILITY

OWNER = "owner@example.com"
GRANTED = f'{{"{HOST_CAPABILITY}":[{{}}]}}'
HOST = "gigachad-backend.tail.ts.net:8443"


class _StreamedBody(httpx.AsyncByteStream):
    """A body omp-web streams, the way the gate reads it (`aiter_raw`)."""

    async def __aiter__(self):
        yield b"ok"


@pytest.fixture
def upstream(monkeypatch) -> list[httpx.Request]:
    """Record what reaches omp-web and answer with two cookies."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, headers=[("set-cookie", "a=1"), ("set-cookie", "b=2")], stream=_StreamedBody())

    monkeypatch.setattr(omp_web_gate, "_transport", httpx.MockTransport(handler))
    return seen


def _get(path: str, login: str | None, capabilities: str | None = GRANTED) -> httpx.Response:
    headers = {"host": HOST, "origin": f"https://{HOST}"}
    headers |= {"tailscale-user-login": login} if login else {}
    headers |= {"tailscale-app-capabilities": capabilities} if capabilities else {}
    with TestClient(omp_web_gate.app) as client:
        return client.get(path, headers=headers)


@pytest.mark.parametrize(
    ("login", "capabilities"),
    [(None, GRANTED), (OWNER, None), (OWNER, '{"example.com/cap/other":[{}]}'), (OWNER, "not json")],
)
def test_refuses_logins_without_the_host_capability(upstream, login, capabilities):
    assert _get("/", login, capabilities).status_code == 403
    assert upstream == []


def test_forwards_with_the_browser_host_marked_https(upstream):
    response = _get("/api/files/dir%2Fa%20b.txt?cwd=%2Frepo", OWNER)

    assert response.status_code == 200
    forwarded = upstream[0]
    assert forwarded.headers["host"] == HOST
    assert forwarded.headers["x-forwarded-proto"] == "https"
    assert forwarded.url.raw_path == b"/api/files/dir%2Fa%20b.txt?cwd=%2Frepo"
    assert response.headers.get_list("set-cookie") == ["a=1", "b=2"]
