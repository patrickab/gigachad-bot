import asyncio
from pathlib import Path
import shutil
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

from backend.identity import RequestIdentity, get_request_identity
from backend.routes import neovim as neovim_route
from backend.routes.deps import get_file_roots
from lib.file_roots import FileRoots
from lib.neovim_session import NeovimSession, is_neovim_available

requires_nvim = pytest.mark.skipif(shutil.which("nvim") is None, reason="nvim is not installed")


@pytest.fixture
def neovim_home(tmp_path, monkeypatch):
    """Empty Neovim config and state, so tests never load the developer's setup."""
    for variable in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
        monkeypatch.setenv(variable, str(tmp_path / variable.lower()))
    config_dir = tmp_path / "xdg_config_home" / "nvim"
    config_dir.mkdir(parents=True)
    return config_dir


@pytest.mark.parametrize(
    ("cors_origins", "origin", "trusted"),
    [
        (["*"], "http://localhost:2999", True),
        (["*"], "http://127.0.0.1:2999", True),
        (["*"], "https://evil.example", False),
        (["*"], None, False),
        (["https://gigachad-bot.vercel.app"], "https://gigachad-bot.vercel.app", True),
        (["https://gigachad-bot.vercel.app"], "http://localhost:2999", False),
    ],
)
def test_terminal_origin_policy(monkeypatch, cors_origins, origin, trusted):
    monkeypatch.setattr(neovim_route, "CORS_ORIGINS", cors_origins)
    assert neovim_route.is_trusted_origin(origin) is trusted


@requires_nvim
def test_neovim_requires_allow_listed_login_and_config(neovim_home, monkeypatch):
    monkeypatch.setenv("GIGACHAD_NVIM_USERS", "alice@example.test, bob@example.test")
    assert is_neovim_available("bob@example.test")
    assert not is_neovim_available("mallory@example.test")
    neovim_home.rmdir()
    assert not is_neovim_available("bob@example.test")


@pytest.mark.parametrize(
    ("allowed_users", "origin"),
    [("someone-else@example.test", "http://localhost:2999"), ("me@example.test", "https://evil.example")],
)
def test_session_refuses_untrusted_requests(monkeypatch, neovim_home, allowed_users, origin):
    monkeypatch.setenv("GIGACHAD_NVIM_USERS", allowed_users)
    monkeypatch.setattr(neovim_route, "CORS_ORIGINS", ["*"])
    app = FastAPI()
    app.include_router(neovim_route.router)
    app.dependency_overrides[get_request_identity] = lambda: RequestIdentity(uuid4(), "me@example.test", None)
    app.dependency_overrides[get_file_roots] = lambda: None
    with (
        pytest.raises(WebSocketDisconnect) as refused,
        TestClient(app).websocket_connect("/api/neovim/session", headers={"origin": origin}),
    ):
        pass
    assert refused.value.code == 1008


def test_neovim_starts_in_innermost_root_holding_a_file_root_document(tmp_path):
    vault = tmp_path / "vault"
    project = vault / "projects" / "app"
    (project / "docs").mkdir(parents=True)
    roots = FileRoots(repository=SimpleNamespace(load=lambda: [str(vault), str(project)], save=lambda roots: None))

    assert neovim_route.working_directory(roots, str(project / "docs" / "notes.md")) == project.resolve()
    assert neovim_route.working_directory(roots, str(vault / "inbox.md")) == vault.resolve()
    assert neovim_route.working_directory(roots, str(tmp_path / "elsewhere.md")) == Path.home()
    assert neovim_route.working_directory(roots, "project/demo/document/notes.md") == Path.home()


@requires_nvim
async def test_write_persists_through_save_and_failed_save_keeps_buffer_modified(neovim_home):
    saved: asyncio.Queue[str] = asyncio.Queue()
    rejections = iter([False, True, False])

    async def save(text: str) -> str:
        await saved.put(text)
        if next(rejections):
            raise RuntimeError("stale revision")
        return text.replace("line", "LINE")

    session = NeovimSession.spawn("notes.md", "# Title\nbody", rows=24, cols=80, cwd=neovim_home)
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(save), 15)

        session.write_input(b"Goline\x1b:w\r")
        assert await asyncio.wait_for(saved.get(), 10) == "# Title\nbody\nline\n"

        session.write_input(b"Goextra\x1b:w\r")
        assert await asyncio.wait_for(saved.get(), 10) == "# Title\nbody\nLINE\nextra\n"
        # The rejected save left the buffer modified, so `:q` must refuse to quit.
        session.write_input(b"\r:q\r:w\r")
        assert await asyncio.wait_for(saved.get(), 10) == "# Title\nbody\nLINE\nextra\n"

        session.write_input(b":q\r")
        assert await asyncio.wait_for(session.wait(), 10) == 0
    finally:
        await session.close()
        await drain


async def _drain(session: NeovimSession) -> None:
    while await session.read_output() is not None:
        pass
