import asyncio
from pathlib import Path
import time

import pytest
from starlette.websockets import WebSocketDisconnect

from backend import websocket_origin
from backend.identity import get_request_identity
from backend.routes import terminal as terminal_route
from lib.pty_process import PtyProcess
from tests.conftest import request_identity, route_client

TRUSTED_ORIGIN = {"origin": "http://localhost:2999"}


@pytest.fixture
def shell_home(tmp_path, monkeypatch):
    """`sh` in an empty home, so no profile of the developer runs."""
    monkeypatch.setattr(websocket_origin, "CORS_ORIGINS", ["*"])
    monkeypatch.setenv("SHELL", "/bin/sh")
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize(("has_host_access", "origin"), [(False, "http://localhost:2999"), (True, "https://evil.example")])
def test_session_refuses_untrusted_requests(shell_home, has_host_access, origin):
    client = route_client(terminal_route.router, overrides={get_request_identity: request_identity(has_host_access)})
    with (
        pytest.raises(WebSocketDisconnect) as refused,
        client.websocket_connect("/api/terminal/session", headers={"origin": origin}),
    ):
        pass
    assert refused.value.code == 1008


def test_shell_runs_typed_commands_in_home_and_reports_its_exit_code(shell_home):
    client = route_client(terminal_route.router, overrides={get_request_identity: request_identity()})
    with client.websocket_connect("/api/terminal/session", headers=TRUSTED_ORIGIN) as socket:
        socket.send_json({"type": "open", "rows": 24, "cols": 80})
        socket.send_bytes(b'echo "at-$(pwd)-$((40+2))"\r')
        output = b""
        while f"at-{shell_home}-42".encode() not in output:
            output += socket.receive_bytes()
        socket.send_bytes(b"exit 7\r")
        while (message := socket.receive())["type"] == "websocket.send" and message.get("bytes") is not None:
            pass
        assert message.get("text") == '{"type":"exit","code":7}'


async def test_close_ends_a_shell_that_ignores_sigterm(tmp_path):
    ready = tmp_path / "ready"
    shell = PtyProcess.spawn(["/bin/sh", "-c", f"trap '' TERM; touch {ready}; sleep 30"], cwd=Path(tmp_path), rows=24, cols=80)
    while not ready.exists():
        await asyncio.sleep(0.01)
    started = time.monotonic()
    await shell.close()
    assert time.monotonic() - started < 1
    assert await shell.read_output() is None
