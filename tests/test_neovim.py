import asyncio
from pathlib import Path
import shutil

import pytest
from starlette.websockets import WebSocketDisconnect

from backend.identity import get_request_identity
from backend.routes import neovim as neovim_route
from backend.routes.deps import get_file_roots
from lib.neovim_session import NeovimSession, is_neovim_installed
from tests.conftest import mounted_roots, request_identity, route_client

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
def test_neovim_needs_a_config_to_load(neovim_home):
    assert is_neovim_installed()
    neovim_home.rmdir()
    assert not is_neovim_installed()


@pytest.mark.parametrize(
    ("has_host_access", "origin"),
    [(False, "http://localhost:2999"), (True, "https://evil.example")],
)
def test_session_refuses_untrusted_requests(monkeypatch, neovim_home, has_host_access, origin):
    monkeypatch.setattr(neovim_route, "CORS_ORIGINS", ["*"])
    client = route_client(
        neovim_route.router, overrides={get_request_identity: request_identity(has_host_access), get_file_roots: None}
    )
    with (
        pytest.raises(WebSocketDisconnect) as refused,
        client.websocket_connect("/api/neovim/session", headers={"origin": origin}),
    ):
        pass
    assert refused.value.code == 1008


def test_neovim_starts_in_innermost_root_holding_a_file_root_document(tmp_path):
    vault = tmp_path / "vault"
    project = vault / "projects" / "app"
    (project / "docs").mkdir(parents=True)
    roots = mounted_roots(vault, project)

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
        await asyncio.wait_for(session.connect(save, lambda text: None), 15)

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


@requires_nvim
async def test_base_content_opens_readonly_left_diff_and_only_working_buffer_saves(neovim_home):
    saved: asyncio.Queue[str] = asyncio.Queue()

    async def save(text: str) -> str:
        await saved.put(text)
        return text

    session = NeovimSession.spawn("notes.md", "one\nTWO\n", rows=24, cols=100, cwd=neovim_home, base_content="one\ntwo\n")
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(save, lambda text: None), 15)
        request = session._rpc.request
        state = await request(
            "nvim_exec_lua",
            """
            local wins = vim.api.nvim_tabpage_list_wins(0)
            table.sort(wins, function(a, b) return vim.fn.win_screenpos(a)[2] < vim.fn.win_screenpos(b)[2] end)
            local function info(w)
              local b = vim.api.nvim_win_get_buf(w)
              return { diff = vim.wo[w].diff, modifiable = vim.bo[b].modifiable, readonly = vim.bo[b].readonly,
                       text = table.concat(vim.api.nvim_buf_get_lines(b, 0, -1, false), "\\n"),
                       current = w == vim.api.nvim_get_current_win() }
            end
            return { #wins, info(wins[1]), info(wins[2]) }
            """,
            [],
        )
        count, left, right = state
        assert count == 2
        assert left == {"diff": True, "modifiable": False, "readonly": True, "text": "one\ntwo", "current": False}
        assert right == {"diff": True, "modifiable": True, "readonly": False, "text": "one\nTWO", "current": True}

        # `:w` in the baseline must never reach the save callback.
        session.write_input(b"\x17h:w\r")
        await asyncio.sleep(0.5)
        assert saved.empty()

        session.write_input(b"\x17lGoedit\x1b:w\r")
        assert await asyncio.wait_for(saved.get(), 10) == "one\nTWO\nedit\n"
    finally:
        await session.close()
        await drain


@requires_nvim
async def test_empty_base_content_still_opens_diff_and_no_base_means_single_window(neovim_home):
    async def save(text: str) -> str:
        return text

    async def window_count(base_content: str | None) -> int:
        session = NeovimSession.spawn("a.txt", "x\n", rows=24, cols=80, cwd=neovim_home, base_content=base_content)
        drain = asyncio.create_task(_drain(session))
        try:
            await asyncio.wait_for(session.connect(save, lambda text: None), 15)
            return await session._rpc.request("nvim_exec_lua", "return #vim.api.nvim_tabpage_list_wins(0)", [])
        finally:
            await session.close()
            await drain

    assert await window_count("") == 2
    assert await window_count(None) == 1


def test_open_message_base_content_is_optional():
    fields = {"type": "open", "path": "/a", "filename": "a", "content": "x", "rows": 1, "cols": 1}
    assert neovim_route.OpenMessage.model_validate(fields).base_content is None
    assert neovim_route.OpenMessage.model_validate({**fields, "base_content": ""}).base_content == ""
    assert neovim_route.OpenMessage.model_validate({**fields, "base_content": None}).base_content is None


async def _drain(session: NeovimSession) -> None:
    while await session.read_output() is not None:
        pass


@requires_nvim
async def test_yank_notifies_with_linewise_and_charwise_text(neovim_home):
    yanked: asyncio.Queue[str] = asyncio.Queue()

    async def save(text: str) -> str:
        return text

    session = NeovimSession.spawn("notes.md", "alpha beta\ngamma\n", rows=24, cols=80, cwd=neovim_home)
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(save, yanked.put_nowait), 15)
        session.write_input(b"yw")
        assert await asyncio.wait_for(yanked.get(), 10) == "alpha "
        session.write_input(b"yj")
        assert await asyncio.wait_for(yanked.get(), 10) == "alpha beta\ngamma\n"
        session.write_input(b"dd")  # deletes are not yanks to the clipboard
        session.write_input(b"yy")
        assert await asyncio.wait_for(yanked.get(), 10) == "gamma\n"
    finally:
        await session.close()
        await drain
