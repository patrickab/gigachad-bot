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


def _read_until(socket, marker: bytes) -> bytes:
    output = b""
    while marker not in output:
        output += socket.receive_bytes()
    return output


def test_escape_leaves_at_the_prompt_and_reaches_a_running_program(shell_home):
    client = route_client(terminal_route.router, overrides={get_request_identity: request_identity()})
    with client.websocket_connect("/api/terminal/session", headers=TRUSTED_ORIGIN) as socket:
        socket.send_json({"type": "open", "rows": 24, "cols": 80})
        socket.send_bytes(b"echo ready-$((1+1))\r")
        _read_until(socket, b"ready-2")
        socket.send_json({"type": "escape"})
        while (message := socket.receive()).get("text") is None:
            pass
        assert message["text"] == '{"type":"leave"}'
        # `cat` holds the foreground and echoes what it gets: the tty shows the Esc as `^[`. Until `cat` started, Esc still leaves
        socket.send_bytes(b"cat\r")
        output = b""
        for _ in range(200):
            socket.send_json({"type": "escape"})
            output += socket.receive().get("bytes") or b""
            if b"^[" in output:
                break
        else:
            pytest.fail("Esc never reached cat")
        socket.send_bytes(b"\r\x04exit\r")  # Ctrl+D ends `cat` only on an empty line
        while '"exit"' not in (socket.receive().get("text") or ""):
            pass


async def test_close_ends_a_shell_that_ignores_sigterm(tmp_path):
    ready = tmp_path / "ready"
    shell = PtyProcess.spawn(["/bin/sh", "-c", f"trap '' TERM; touch {ready}; sleep 30"], cwd=Path(tmp_path), rows=24, cols=80)
    while not ready.exists():
        await asyncio.sleep(0.01)
    started = time.monotonic()
    await shell.close()
    assert time.monotonic() - started < 1
    assert await shell.read_output() is None
