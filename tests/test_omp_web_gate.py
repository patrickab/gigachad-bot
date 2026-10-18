"""The omp-web gate forwards only allowlisted Tailnet logins, and forwards them so omp-web's same-origin guard still holds."""

import httpx
import pytest
from starlette.testclient import TestClient

from backend import omp_web_gate

OWNER = "owner@example.com"
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
    monkeypatch.setenv("GIGACHAD_OMP_WEB_USERS", f" {OWNER} , second@example.com")
    return seen


def _get(path: str, login: str | None) -> httpx.Response:
    headers = {"host": HOST, "origin": f"https://{HOST}"} | ({"tailscale-user-login": login} if login else {})
    with TestClient(omp_web_gate.app) as client:
        return client.get(path, headers=headers)


@pytest.mark.parametrize("login", [None, "guest@example.com"])
def test_refuses_logins_off_the_allowlist(upstream, login):
    assert _get("/", login).status_code == 403
    assert upstream == []


def test_an_empty_allowlist_refuses_everyone(upstream, monkeypatch):
    monkeypatch.setenv("GIGACHAD_OMP_WEB_USERS", "")
    assert _get("/", OWNER).status_code == 403


def test_forwards_with_the_browser_host_marked_https(upstream):
    response = _get("/api/files/dir%2Fa%20b.txt?cwd=%2Frepo", OWNER)

    assert response.status_code == 200
    forwarded = upstream[0]
    assert forwarded.headers["host"] == HOST
    assert forwarded.headers["x-forwarded-proto"] == "https"
    assert forwarded.url.raw_path == b"/api/files/dir%2Fa%20b.txt?cwd=%2Frepo"
    assert response.headers.get_list("set-cookie") == ["a=1", "b=2"]
