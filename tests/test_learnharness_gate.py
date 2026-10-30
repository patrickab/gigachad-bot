"""The LearnHarness gate forwards only Tailnet logins granted the host capability, unchanged, to LearnHarness."""

import httpx
import pytest
from starlette.testclient import TestClient

from backend import learnharness_gate
from backend.identity import HOST_CAPABILITY

OWNER = "owner@example.com"
GRANTED = f'{{"{HOST_CAPABILITY}":[{{}}]}}'
HOST = "gigachad-backend.tail.ts.net:8444"


class _StreamedBody(httpx.AsyncByteStream):
    """A body LearnHarness streams, the way the gate reads it (`aiter_raw`)."""

    async def __aiter__(self):
        yield b"ok"


@pytest.fixture
def upstream(monkeypatch) -> list[httpx.Request]:
    """Record what reaches LearnHarness and answer with two cookies."""
    monkeypatch.setenv("GIGACHAD_LEARNHARNESS_URL", "http://127.0.0.1:30181")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, headers=[("set-cookie", "a=1"), ("set-cookie", "b=2")], stream=_StreamedBody())

    monkeypatch.setattr(learnharness_gate, "_transport", httpx.MockTransport(handler))
    return seen


def _request(method: str, path: str, login: str | None, capabilities: str | None = GRANTED, **kwargs) -> httpx.Response:
    headers = {"host": HOST, "origin": f"https://{HOST}"}
    headers |= {"tailscale-user-login": login} if login else {}
    headers |= {"tailscale-app-capabilities": capabilities} if capabilities else {}
    with TestClient(learnharness_gate.app) as client:
        return client.request(method, path, headers=headers, **kwargs)


@pytest.mark.parametrize(
    ("login", "capabilities"),
    [(None, GRANTED), (OWNER, None), (OWNER, '{"example.com/cap/other":[{}]}'), (OWNER, "not json")],
)
def test_refuses_logins_without_the_host_capability(upstream, login, capabilities):
    response = _request("GET", "/", login, capabilities)

    assert response.status_code == 403
    assert "LearnHarness" in response.text
    assert upstream == []


def test_forwards_to_learnharness_with_the_browser_host_marked_https(upstream):
    response = _request("POST", "/api/chats?x=%2Fa", OWNER, json={"title": "Fourier"})

    assert response.status_code == 200
    assert response.text == "ok"
    forwarded = upstream[0]
    assert forwarded.method == "POST"
    assert forwarded.url.raw_path == b"/api/chats?x=%2Fa"
    assert forwarded.url.netloc == b"127.0.0.1:30181"
    assert forwarded.headers["host"] == HOST
    assert forwarded.headers["x-forwarded-proto"] == "https"
    assert forwarded.content == b'{"title":"Fourier"}'
    assert response.headers.get_list("set-cookie") == ["a=1", "b=2"]


def test_reports_an_unreachable_learnharness_as_bad_gateway(monkeypatch):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(learnharness_gate, "_transport", httpx.MockTransport(refuse))

    response = _request("GET", "/", OWNER)

    assert response.status_code == 502
    assert response.text.startswith("LearnHarness does not answer")


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("PUT", "/api/keys/OPENAI_API_KEY"),
        ("DELETE", "/api/keys/OPENAI_API_KEY"),
        ("POST", "/api/keys"),
        ("POST", "/api/custom-provider"),
        ("PATCH", "/api/custom-provider"),
        ("DELETE", "/api/custom-provider"),
        ("POST", "/api/subscriptions"),
        ("GET", "/api/subscriptions/browser?id=00000000-0000-0000-0000-000000000000"),
        ("POST", "/api/subscriptions/browser"),
        # Other spellings of the same routes.
        ("PUT", "/api/keys/OPENAI_API_KEY/"),
        ("PUT", "/api//keys//OPENAI_API_KEY"),
        ("PUT", "/api/chats/../keys/OPENAI_API_KEY"),
        ("PUT", "/API/Keys/OPENAI_API_KEY"),
        ("PUT", "/api/%6Beys/OPENAI_API_KEY"),
    ],
)
def test_refuses_provider_key_and_subscription_writes(upstream, method, path):
    response = _request(method, path, OWNER, json={"value": "sk-provider"})

    assert response.status_code == 403
    assert "provider keys" in response.text
    assert upstream == []


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/keys"),
        ("GET", "/api/custom-provider"),
        ("GET", "/api/subscriptions"),
        ("PUT", "/api/settings/models"),
        ("POST", "/api/keysmith"),
    ],
)
def test_forwards_reads_and_other_writes(upstream, method, path):
    response = _request(method, path, OWNER, **({} if method == "GET" else {"json": {}}))

    assert response.status_code == 200
    assert [request.method for request in upstream] == [method]
