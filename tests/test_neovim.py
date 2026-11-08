import asyncio
import contextlib
import json
import os
from pathlib import Path
import shutil

from pydantic import ValidationError
import pytest
from starlette.websockets import WebSocketDisconnect, WebSocketState

from backend.identity import get_request_identity
from backend import websocket_origin
from backend.routes import neovim as neovim_route
from backend.routes.deps import get_file_roots
from lib import neovim_session
from lib.neovim_session import ClipboardError, NeovimSession, is_neovim_installed
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
    monkeypatch.setattr(websocket_origin, "CORS_ORIGINS", cors_origins)
    assert websocket_origin.is_trusted_origin(origin) is trusted


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
    monkeypatch.setattr(websocket_origin, "CORS_ORIGINS", ["*"])
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

    async def save(path: str, text: str) -> str:
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

    async def save(path: str, text: str) -> str:
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
    async def save(path: str, text: str) -> str:
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


def test_open_message_needs_a_document_and_base_content_is_optional():
    fields = {"type": "open", "path": "/a", "filename": "a", "content": "x", "rows": 1, "cols": 1}
    assert neovim_route.OpenMessage.model_validate(fields).base_content is None
    assert neovim_route.OpenMessage.model_validate({**fields, "base_content": ""}).base_content == ""
    assert neovim_route.OpenMessage.model_validate({**fields, "base_content": None}).base_content is None
    assert neovim_route.OpenMessage.model_validate({**fields, "notebook": True}).notebook
    with pytest.raises(ValidationError):
        neovim_route.OpenMessage.model_validate({**fields, "content": None, "notebook": True})


async def _drain(session: NeovimSession) -> None:
    while await session.read_output() is not None:
        pass


async def _until(condition, seconds: float = 10) -> None:
    """Poll the async *condition* until it holds, failing instead of hanging after *seconds*."""

    async def poll() -> None:
        while not await condition():
            await asyncio.sleep(0.05)

    await asyncio.wait_for(poll(), seconds)


async def _evaluates_to(session: NeovimSession, expression: str, expected: object) -> None:
    async def holds() -> bool:
        return await session._rpc.request("nvim_eval", expression) == expected

    await _until(holds)


async def _shows(session: NeovimSession, text: str) -> None:
    """Wait until the last screen line (the message area) contains *text*."""

    async def holds() -> bool:
        line = await session._rpc.request("nvim_eval", "join(map(range(1, &columns), {_, c -> screenstring(&lines, c)}), '')")
        return text in line

    await _until(holds)


@requires_nvim
async def test_yank_notifies_with_linewise_and_charwise_text(neovim_home):
    yanked: asyncio.Queue[str] = asyncio.Queue()

    async def save(path: str, text: str) -> str:
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


@requires_nvim
async def test_escape_in_normal_mode_leaves_after_the_users_own_escape_mapping(neovim_home):
    (neovim_home / "init.lua").write_text('vim.keymap.set("n", "<Esc>", "<Cmd>let g:user_escapes = get(g:, \'user_escapes\', 0) + 1<CR>")\n')
    left: asyncio.Queue[None] = asyncio.Queue()

    async def save(path: str, text: str) -> str:
        return text

    session = NeovimSession.spawn("notes.md", "alpha\n", rows=24, cols=80, cwd=neovim_home)
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(save, lambda text: None, on_leave=lambda: left.put_nowait(None)), 15)
        session.write_input(b"ibeta\x1b")  # leaves insert mode only
        await _evaluates_to(session, "mode()", "n")
        assert left.empty()
        session.write_input(b"\x1b")
        await asyncio.wait_for(left.get(), 10)
        await _evaluates_to(session, "get(g:, 'user_escapes', 0)", 1)
    finally:
        await session.close()
        await drain


@requires_nvim
async def test_set_text_applies_hunks_without_moving_the_cursor_and_leaves_the_buffer_unmodified(neovim_home):
    async def save(path: str, text: str) -> str:
        return text

    session = NeovimSession.spawn(
        "main.tex", "".join(f"line {number}\n" for number in range(1, 21)), rows=24, cols=80, cwd=neovim_home
    )
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(save, lambda text: None), 15)
        session.write_input(b"10Gl\x1b")
        await _evaluates_to(session, "line('.')", 10)

        await session.set_text("line 1\nline TWO\n" + "".join(f"line {number}\n" for number in range(3, 21)) + "last\n", "main.tex")

        assert await session._rpc.request("nvim_eval", "[line('.'), col('.'), &modified]") == [10, 2, 0]
        text = await session._rpc.request("nvim_eval", "join(getline(1, '$'), '|')")
        assert text.startswith("line 1|line TWO|line 3|") and text.endswith("|line 20|last")
    finally:
        await session.close()
        await drain


@requires_nvim
async def test_set_text_highlights_each_changed_hunk(neovim_home):
    async def save(path: str, text: str) -> str:
        return text

    session = NeovimSession.spawn(
        "notes.md", "".join(f"line {number}\n" for number in range(1, 21)), rows=24, cols=80, cwd=neovim_home
    )
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(save, lambda text: None), 15)

        lines = [f"line {number}" for number in range(1, 21)]
        lines[2] = "line THREE"
        lines[14:15] = ["line FIFTEEN", "inserted"]
        await session.set_text("\n".join(lines) + "\n", "notes.md")

        rows = await session._lua(
            "local marks = vim.api.nvim_buf_get_extmarks(0, vim.api.nvim_get_namespaces().gigachad_remote_edit, 0, -1, { details = true })"
            " return vim.tbl_map(function(mark) return { mark[2], mark[4].line_hl_group } end, marks)"
        )
        assert rows == [[2, "DiffText"], [14, "DiffText"], [15, "DiffText"]]
    finally:
        await session.close()
        await drain


@requires_nvim
async def test_set_text_merges_into_a_modified_buffer_which_stays_modified(neovim_home):
    saved: asyncio.Queue[str] = asyncio.Queue()

    async def save(path: str, text: str) -> str:
        await saved.put(text)
        return text

    session = NeovimSession.spawn("notes.md", "one\ntwo\nthree\nfour\nfive\n", rows=24, cols=80, cwd=neovim_home)
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(save, lambda text: None), 15)
        session.write_input(b"5GAx\x1b")
        await _evaluates_to(session, "&modified", 1)

        disk = "zero\none\nTWO\nthree\nfour\nfive\n"
        await session.set_text(disk, "notes.md")

        state = await session._rpc.request("nvim_eval", "[join(getline(1, '$'), '|'), line('.'), &modified]")
        # The unrelated remote edits land, the unsaved one stays, the cursor follows its line down.
        assert state == ["zero|one|TWO|three|four|fivex", 6, 1]

        session.write_input(b":w\r")
        assert await asyncio.wait_for(saved.get(), 10) == "zero\none\nTWO\nthree\nfour\nfivex\n"
    finally:
        await session.close()
        await drain


@requires_nvim
async def test_write_after_the_file_changed_on_disk_saves_the_merged_text(neovim_home, tmp_path):
    saved: asyncio.Queue[str] = asyncio.Queue()

    async def save(path: str, text: str) -> str:
        await saved.put(text)
        return text

    file = tmp_path / "notes.md"
    file.write_text("a\nb\nc\n")
    session = NeovimSession.spawn("notes.md", "a\nb\nc\n", rows=24, cols=80, cwd=neovim_home, disk_path=file)
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(save, lambda text: None), 15)
        file.write_text("A\nb\nc\n")

        session.write_input(b"Gox\x1b:w\r")

        assert await asyncio.wait_for(saved.get(), 10) == "A\nb\nc\nx\n"
        # `save` returned without suspending, so its reply reached Neovim before this request.
        assert await session._rpc.request("nvim_eval", "[join(getline(1, '$'), '|'), &modified]") == ["A|b|c|x", 0]
        await _shows(session, "Saved, merged with the agent's edits")

        # Disk now matches the base, so the next save sends the buffer as it is.
        file.write_text("A\nb\nc\nx\n")
        session.write_input(b"Goy\x1b:w\r")
        assert await asyncio.wait_for(saved.get(), 10) == "A\nb\nc\nx\ny\n"
        await _shows(session, "Saved notes.md")
    finally:
        await session.close()
        await drain


@requires_nvim
async def test_rename_keeps_the_unsaved_text_and_the_next_write_merges_with_the_renamed_file(neovim_home, tmp_path):
    saved: asyncio.Queue[str] = asyncio.Queue()

    async def save(path: str, text: str) -> str:
        await saved.put(text)
        return text

    old, new = tmp_path / "notes.md", tmp_path / "ideas.md"
    old.write_text("a\nb\nc\n")
    session = NeovimSession.spawn("notes.md", "a\nb\nc\n", rows=24, cols=80, cwd=neovim_home, disk_path=old)
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(save, lambda text: None), 15)
        session.write_input(b"Gox\x1b")
        await _evaluates_to(session, "&modified", 1)

        old.rename(new)
        await session.rename("notes.md", "ideas.md", new)

        state = await session._rpc.request("nvim_eval", "[expand('%:t'), join(getline(1, '$'), '|'), &modified]")
        assert state == ["ideas.md", "a|b|c|x", 1]
        new.write_text("A\nb\nc\n")
        session.write_input(b":w\r")
        assert await asyncio.wait_for(saved.get(), 10) == "A\nb\nc\nx\n"
        await _shows(session, "Saved, merged with the agent's edits")

        # Back to the first name: the buffer left under it by the rename must not refuse it
        await session.rename("ideas.md", "notes.md", old)
        assert await session._rpc.request("nvim_eval", "expand('%:t')") == "notes.md"
    finally:
        await session.close()
        await drain


@requires_nvim
async def test_set_text_during_a_write_waits_for_it_and_merges_into_the_stored_text(neovim_home, tmp_path):
    saved: asyncio.Queue[str] = asyncio.Queue()
    answer = asyncio.Event()
    file = tmp_path / "notes.md"

    async def save(path: str, text: str) -> str:  # the browser's round trip, it stores the text once answered
        await saved.put(text)
        await answer.wait()
        file.write_text(text)
        return text

    file.write_text("a\nb\nc\n")
    session = NeovimSession.spawn("notes.md", "a\nb\nc\n", rows=24, cols=80, cwd=neovim_home, disk_path=file)
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(save, lambda text: None), 15)
        session.write_input(b"Gox\x1b:w\r")
        assert await asyncio.wait_for(saved.get(), 10) == "a\nb\nc\nx\n"

        # The agent's edit of the old text arrives while the save is pending.
        agent_edit = asyncio.create_task(session.set_text("A\nb\nc\n", "notes.md"))
        await asyncio.sleep(0.2)
        assert not agent_edit.done()
        answer.set()
        await asyncio.wait_for(agent_edit, 10)

        # Both edits are in the buffer, which differs from the stored text and so stays modified.
        assert await session._rpc.request("nvim_eval", "[join(getline(1, '$'), '|'), &modified]") == ["A|b|c|x", 1]

        session.write_input(b":w\r")
        assert await asyncio.wait_for(saved.get(), 10) == "A\nb\nc\nx\n"
        await _evaluates_to(session, "&modified", 0)
        assert file.read_text() == "A\nb\nc\nx\n"
    finally:
        await session.close()
        await drain


@requires_nvim
@pytest.mark.parametrize(
    ("disk", "error"),
    [(b"\xff\xfeA\nb\nc\n", "not UTF-8 text"), (b"A\x00\nb\nc\n", "could not merge with the changed file")],
)
async def test_write_fails_without_writing_when_the_changed_file_cannot_be_merged(neovim_home, tmp_path, disk, error):
    saved: asyncio.Queue[str] = asyncio.Queue()

    async def save(path: str, text: str) -> str:
        await saved.put(text)
        return text

    file = tmp_path / "notes.md"
    file.write_text("a\nb\nc\n")
    session = NeovimSession.spawn("notes.md", "a\nb\nc\n", rows=24, cols=80, cwd=neovim_home, disk_path=file)
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(save, lambda text: None), 15)
        file.write_bytes(disk)
        session.write_input(b"Gox\x1b:w\r")

        async def failed() -> bool:
            session.write_input(b"\r")  # dismisses the hit-enter prompt of the error, which holds off requests
            return "Save failed" in await session._rpc.request("nvim_eval", "v:errmsg")

        await _until(failed)
        assert error in await session._rpc.request("nvim_eval", "v:errmsg")
        assert await session._rpc.request("nvim_eval", "[join(getline(1, '$'), '|'), &modified]") == ["a|b|c|x", 1]
        assert saved.empty()
        assert file.read_bytes() == disk
    finally:
        await session.close()
        await drain


def test_only_readable_absolute_documents_have_a_disk_file(tmp_path):
    (tmp_path / "notes.md").write_text("x")
    roots = mounted_roots(tmp_path)

    assert neovim_route.disk_file(roots, str(tmp_path / "notes.md")) == (tmp_path / "notes.md").resolve()
    assert neovim_route.disk_file(roots, str(tmp_path / "missing.md")) is None
    assert neovim_route.disk_file(roots, str(tmp_path.parent / "outside.md")) is None
    assert neovim_route.disk_file(roots, "project/demo/document/notes.md") is None


@requires_nvim
async def test_background_asked_before_startup_wins_over_detection_and_follows_later_changes(neovim_home):
    async def save(path: str, text: str) -> str:
        return text

    session = NeovimSession.spawn("main.tex", "text\n", rows=24, cols=80, cwd=neovim_home)
    drain = asyncio.create_task(_drain(session))
    try:
        await session.set_background(True)
        await asyncio.wait_for(session.connect(save, lambda text: None), 15)
        assert await session._rpc.request("nvim_eval", "&background") == "light"

        await session.set_background(False)
        assert await session._rpc.request("nvim_eval", "&background") == "dark"
    finally:
        await session.close()
        await drain


@requires_nvim
async def test_visual_ask_keymap_reports_the_selected_lines_only_when_requested(neovim_home):
    selections: asyncio.Queue[dict] = asyncio.Queue()

    async def save(path: str, text: str) -> str:
        return text

    session = NeovimSession.spawn("main.tex", "a\nb\nc\nd\n", rows=24, cols=80, cwd=neovim_home)
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(save, lambda text: None, selections.put_nowait), 15)
        session.write_input(b"3GVk\\a")
        assert await asyncio.wait_for(selections.get(), 10) == {"path": "main.tex", "start_line": 2, "end_line": 3, "text": "b\nc"}
        assert await session._rpc.request("nvim_eval", "mode()") == "n"
    finally:
        await session.close()
        await drain


def _two_files(tmp_path: Path) -> tuple[Path, Path]:
    """Two files with the same basename in different folders."""
    first, second = tmp_path / "intro" / "main.tex", tmp_path / "appendix" / "main.tex"
    for file, text in ((first, "intro\n"), (second, "appendix\n")):
        file.parent.mkdir()
        file.write_text(text)
    return first, second


@contextlib.asynccontextmanager
async def _editing(first: Path, home: Path, save, on_switched=None, read_clipboard=None, on_forward_search=None):
    """A connected session on *first* (path = its absolute path), closed afterwards."""
    session = NeovimSession.spawn(first.name, first.read_text(), rows=24, cols=80, cwd=home, disk_path=first, path=str(first))
    drain = asyncio.create_task(_drain(session))
    try:
        connecting = session.connect(save, lambda text: None, None, on_switched, read_clipboard, on_forward_search=on_forward_search)
        await asyncio.wait_for(connecting, 15)
        yield session
    finally:
        await session.close()
        await drain


async def _store(path: str, text: str) -> str:
    return text


async def _buffer_state(session: NeovimSession, path: Path) -> dict:
    return await session._lua("return _G.gigachad_buffer_state(...)", str(path))


@requires_nvim
async def test_each_document_saves_to_its_own_path_and_equal_basenames_do_not_collide(neovim_home, tmp_path):
    saved: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

    async def save(path: str, text: str) -> str:
        await saved.put((path, text))
        return text

    (neovim_home / "init.lua").write_text("vim.o.hidden = false")  # a config that would refuse to leave a modified buffer
    first, second = _two_files(tmp_path)
    async with _editing(first, neovim_home, save) as session:
        session.write_input(b"Goone\x1b")
        await _evaluates_to(session, "&modified", 1)

        await session.open_file(str(second), second)
        assert await session._rpc.request("nvim_eval", "[getline(1), &modified]") == ["appendix", 0]
        session.write_input(b"Gotwo\x1b:w\r")
        assert await asyncio.wait_for(saved.get(), 10) == (str(second), "appendix\ntwo\n")

        # Switching saved nothing: back on the first document its edit is still unsaved, and `:w` stores it under its own path
        assert saved.empty()
        await session.open_file(str(first), first)
        assert await session._rpc.request("nvim_eval", "[getline(1, '$'), &modified]") == [["intro", "one"], 1]
        session.write_input(b":w\r")
        assert await asyncio.wait_for(saved.get(), 10) == (str(first), "intro\none\n")


@requires_nvim
async def test_set_text_reaches_a_hidden_buffer_and_merges_its_unsaved_edits(neovim_home, tmp_path):
    first, second = _two_files(tmp_path)
    first.write_text("a\nb\nc\n")
    async with _editing(first, neovim_home, _store) as session:
        session.write_input(b"Gox\x1b")
        await _evaluates_to(session, "&modified", 1)
        await session.open_file(str(second), second)

        await session.set_text("A\nb\nc\n", str(first))

        state = await _buffer_state(session, first)
        assert (state["text"], state["modified"]) == ("A\nb\nc\nx\n", True)
        assert await session._rpc.request("nvim_eval", "[getline(1), &modified]") == ["appendix", 0]

        # An unmodified hidden buffer takes the text and stays unmodified, one that is not open is ignored
        await session.open_file(str(first), first)
        await session.set_text("APPENDIX\n", str(second))
        await session.set_text("lost\n", str(tmp_path / "elsewhere.tex"))

        state = await _buffer_state(session, second)
        assert (state["text"], state["modified"]) == ("APPENDIX\n", False)


@requires_nvim
async def test_rename_renames_the_buffer_of_the_old_path_only(neovim_home, tmp_path):
    saved: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

    async def save(path: str, text: str) -> str:
        await saved.put((path, text))
        return text

    first, second = _two_files(tmp_path)
    renamed = second.with_name("appendix.tex")
    async with _editing(first, neovim_home, save) as session:
        await session.open_file(str(second), second)
        second.rename(renamed)
        await session.rename(str(second), str(renamed), renamed)

        loaded = "map(getbufinfo({'bufloaded': 1}), {_, buffer -> fnamemodify(buffer.name, ':t')})"
        assert await session._rpc.request("nvim_eval", loaded) == ["main.tex", "appendix.tex"]

        # Nothing happens for a document that is not open, and the renamed buffer saves under its new path
        await session.rename(str(tmp_path / "elsewhere.tex"), str(tmp_path / "other.tex"), None)
        session.write_input(b"Gox\x1b:w\r")
        assert await asyncio.wait_for(saved.get(), 10) == (str(renamed), "appendix\nx\n")


@requires_nvim
async def test_leader_v_reports_the_cursor_line_of_tex_documents_only(neovim_home, tmp_path):
    searches: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    tex = tmp_path / "main.tex"
    tex.write_text("a\nb\nc\n")
    notes = _notes(tmp_path)
    async with _editing(tex, neovim_home, _store, on_forward_search=lambda path, line: searches.put_nowait((path, line))) as session:
        await session._rpc.request("nvim_win_set_cursor", 0, [3, 0])
        session.write_input(b"\\v")
        assert await asyncio.wait_for(searches.get(), 10) == (str(tex), 3)

        await session.open_file(str(notes), notes)
        session.write_input(b"\\v")
        await _evaluates_to(session, "mode()", "v")  # unmapped, `v` starts a selection
        session.write_input(b"\x1b")
        await _evaluates_to(session, "mode()", "n")

        # A rename to a `.tex` file maps it
        renamed = notes.rename(notes.with_name("ideas.tex"))
        await session.rename(str(notes), str(renamed), renamed)
        session.write_input(b"\\v")
        assert await asyncio.wait_for(searches.get(), 10) == (str(renamed), 1)

        # A rename to another kind of file unmaps it
        markdown = renamed.rename(renamed.with_name("ideas.md"))
        await session.rename(str(renamed), str(markdown), markdown)
        session.write_input(b"\\v")
        await _evaluates_to(session, "mode()", "v")
        session.write_input(b"\x1b")
        await _evaluates_to(session, "mode()", "n")
    assert searches.empty()


@requires_nvim
async def test_insert_lines_land_below_the_cursor_with_the_cursor_on_the_first_inserted_line(neovim_home, tmp_path):
    file = tmp_path / "main.tex"
    file.write_text("a\nb\nc\n")
    async with _editing(file, neovim_home, _store) as session:
        session.write_input(b"2G")
        await _evaluates_to(session, "line('.')", 2)

        await session.insert_lines("x\ny\n")
        assert await session._rpc.request("nvim_eval", "[getline(1, '$'), line('.'), &modified]") == [["a", "b", "x", "y", "c"], 3, 1]

        await session.insert_lines("z")
        assert await session._rpc.request("nvim_eval", "[getline(1, '$'), line('.')]") == [["a", "b", "x", "z", "y", "c"], 4]


SLIDES = [
    r"\documentclass{beamer}",
    r"\begin{document}",
    r"\begin{frame}{One}",
    "a",
    r"\end{frame}",
    "",
    r"\begin{frame}{Two}",
    "b",
    r"\end{frame}",
    r"\end{document}",
]
NEW_FRAME = [r"\begin{frame}{New}", "x", r"\end{frame}"]
ROUNDIMAGE = r"\roundimage{example-image-16x9}"


@requires_nvim
async def test_insert_frame_lands_after_the_enclosing_frame_with_the_cursor_on_its_begin_line(neovim_home, tmp_path):
    file = tmp_path / "main.tex"
    file.write_text("\n".join(SLIDES) + "\n")
    async with _editing(file, neovim_home, _store) as session:
        # Over RPC: typed keys arrive later, and waiting for line 1 holds at once (the cursor starts there) so a typed `1G` could
        # move the cursor after the insert
        await session._rpc.request("nvim_win_set_cursor", 0, [4, 0])

        await session.insert_frame("\n".join(NEW_FRAME) + "\n")
        inserted = [*SLIDES[:5], "", *NEW_FRAME, *SLIDES[5:]]
        assert await session._rpc.request("nvim_eval", "[getline(1, '$'), line('.')]") == [inserted, 7]


@requires_nvim
@pytest.mark.parametrize(
    ("cursor", "expected", "landed"),
    [
        (1, [r"\roundimage{imgs/100%.png}", "text"], 1),
        (2, [ROUNDIMAGE, "text", r"\roundimage{imgs/100%.png}"], 3),
    ],
)
async def test_insert_image_fills_the_first_placeholder_on_the_cursor_line_else_inserts_the_reference_below(
    neovim_home, tmp_path, cursor, expected, landed
):
    file = tmp_path / "main.tex"
    file.write_text(f"{ROUNDIMAGE}\ntext\n")
    async with _editing(file, neovim_home, _store) as session:
        await session._rpc.request("nvim_win_set_cursor", 0, [cursor, 0])

        await session.insert_image("imgs/100%.png", "\\roundimage{imgs/100%.png}\n")
        assert await session._rpc.request("nvim_eval", "[getline(1, '$'), line('.')]") == [expected, landed]


@requires_nvim
@pytest.mark.parametrize(
    ("edits", "expected", "outcome"),
    [
        # Worked out again on the changed lines
        (1, ["% typed", *SLIDES[:5], "", *NEW_FRAME, *SLIDES[5:]], contextlib.nullcontext()),
        # Changed again: fails without inserting
        (2, ["% typed", "% typed", *SLIDES], pytest.raises(RuntimeError, match="kept changing")),
    ],
)
async def test_insert_frame_is_worked_out_once_more_when_the_buffer_changed_before_it_applied(
    neovim_home, tmp_path, monkeypatch, edits, expected, outcome
):
    file = tmp_path / "main.tex"
    file.write_text("\n".join(SLIDES) + "\n")
    async with _editing(file, neovim_home, _store) as session:
        await session._rpc.request("nvim_win_set_cursor", 0, [4, 0])
        lua, typed = session._lua, 0

        async def typing_after_each_read(code: str, *args: object) -> object:
            nonlocal typed
            result = await lua(code, *args)
            if "gigachad_cursor_state" in code and typed < edits:
                typed += 1
                await session._rpc.request("nvim_buf_set_lines", 0, 0, 0, False, ["% typed"])
            return result

        monkeypatch.setattr(session, "_lua", typing_after_each_read)
        with outcome:
            await session.insert_frame("\n".join(NEW_FRAME))
        assert await session._rpc.request("nvim_eval", "getline(1, '$')") == expected


@requires_nvim
async def test_insert_frame_is_refused_when_another_buffer_with_the_same_tick_is_shown_before_it_applied(
    neovim_home, tmp_path, monkeypatch
):
    file = tmp_path / "main.tex"
    file.write_text("\n".join(SLIDES) + "\n")
    async with _editing(file, neovim_home, _store) as session:
        await session._rpc.request("nvim_win_set_cursor", 0, [4, 0])
        lua, first_apply = session._lua, None

        async def switching_after_the_first_read(code: str, *args: object) -> object:
            nonlocal first_apply
            result = await lua(code, *args)
            if "gigachad_cursor_state" in code and first_apply is None:
                # A second loaded buffer brought to the read buffer's tick, then shown in the work window
                await session._rpc.request(
                    "nvim_exec_lua",
                    "local read, other = ..., vim.api.nvim_create_buf(true, false)"
                    " while vim.api.nvim_buf_get_changedtick(other) < vim.api.nvim_buf_get_changedtick(read) do"
                    "   vim.api.nvim_buf_set_lines(other, 0, -1, false, { 'other' })"
                    " end"
                    " assert(vim.api.nvim_buf_get_changedtick(other) == vim.api.nvim_buf_get_changedtick(read))"
                    " vim.api.nvim_win_set_buf(0, other)",
                    [result["buffer"]],
                )
            elif "gigachad_insert_at" in code and first_apply is None:
                lines = await session._rpc.request(
                    "nvim_exec_lua",
                    "return { vim.api.nvim_buf_get_lines(..., 0, -1, false), vim.api.nvim_buf_get_lines(0, 0, -1, false) }",
                    [args[3]],
                )
                first_apply = [result, *lines]
            return result

        monkeypatch.setattr(session, "_lua", switching_after_the_first_read)
        await session.insert_frame("\n".join(NEW_FRAME))
        assert first_apply == [False, SLIDES, ["other"]]

@requires_nvim
async def test_switched_reports_buffer_changes_made_in_neovim_but_not_by_open_file(neovim_home, tmp_path):
    switched: asyncio.Queue[str] = asyncio.Queue()
    first, second = _two_files(tmp_path)
    async with _editing(first, neovim_home, _store, switched.put_nowait) as session:
        await session.open_file(str(second), second)
        await session.open_file(str(second), second)
        assert switched.empty()

        session.write_input(b":bnext\r")
        assert await asyncio.wait_for(switched.get(), 10) == str(first)
        session.write_input(b":bprevious\r")
        assert await asyncio.wait_for(switched.get(), 10) == str(second)


@requires_nvim
@pytest.mark.parametrize(("disk", "error"), [(None, "not a readable file"), (b"\xff\xfe\n", "not UTF-8 text")])
async def test_open_file_fails_for_a_file_it_cannot_read_and_stays_on_the_shown_one(neovim_home, tmp_path, disk, error):
    first, second = _two_files(tmp_path)
    if disk is not None:
        second.write_bytes(disk)
    async with _editing(first, neovim_home, _store) as session:
        with pytest.raises(RuntimeError, match=error):
            await session.open_file(str(second), None if disk is None else second)

        assert await session._rpc.request("nvim_eval", "getline(1)") == "intro"


@requires_nvim
async def test_open_file_loads_a_deleted_buffer_again_from_disk(neovim_home, tmp_path):
    first, second = _two_files(tmp_path)
    async with _editing(first, neovim_home, _store) as session:
        await session.open_file(str(second), second)
        session.write_input(b":bdelete\r")

        await _evaluates_to(session, "getline(1)", "intro")
        second.write_text("appendix, edited\n")
        await session.open_file(str(second), second)

        assert await session._rpc.request("nvim_eval", "[getline(1), &modified]") == ["appendix, edited", 0]


@requires_nvim
async def test_open_file_with_a_line_moves_the_cursor_there_centred_also_on_the_shown_file(neovim_home, tmp_path):
    file = tmp_path / "main.tex"
    file.write_text("".join(f"row {number}\n" for number in range(1, 101)))
    async with _editing(file, neovim_home, _store) as session:
        await session._rpc.request("nvim_win_set_cursor", 0, [5, 3])
        await session.open_file(str(file), file, line=60)

        # `zz` puts the cursor line in the middle of the 24-row terminal's window
        cursor, top, bottom = await session._rpc.request("nvim_eval", "[getpos('.')[1:2], line('w0'), line('w$')]")
        assert cursor == [60, 1]
        assert top < 60 < bottom and abs((60 - top) - (bottom - 60)) <= 1


@requires_nvim
async def test_open_file_with_a_line_lands_there_in_a_freshly_opened_file_and_clamps_past_the_end(neovim_home, tmp_path):
    first, second = _two_files(tmp_path)
    second.write_text("one\ntwo\nthree\n")
    async with _editing(first, neovim_home, _store) as session:
        await session.open_file(str(second), second, line=2)
        assert await session._rpc.request("nvim_eval", "[getline('.'), col('.')]") == ["two", 1]

        await session.open_file(str(first), first, line=1)
        await session.open_file(str(second), second, line=99)
        assert await session._rpc.request("nvim_eval", "[getline('.'), line('.')]") == ["three", 3]

        await session.open_file(str(first), first)
        await session.open_file(str(second), second)
        assert await session._rpc.request("nvim_eval", "line('.')") == 3


@requires_nvim
async def test_open_file_with_a_column_puts_the_cursor_on_that_byte_clamped_to_the_line(neovim_home, tmp_path):
    file = tmp_path / "main.tex"
    file.write_text("intro\n\\point{Café note}{See the état.}\n")
    async with _editing(file, neovim_home, _store) as session:
        # "état" starts at character 26, byte 27 after the two-byte "é" of "Café"
        await session.open_file(str(file), file, line=2, column=27)
        assert await session._rpc.request("nvim_win_get_cursor", 0) == [2, 27]
        assert await session._rpc.request("nvim_eval", "strcharpart(getline('.'), charcol('.') - 1, 2)") == "ét"

        await session.open_file(str(file), file, line=1, column=99)
        assert await session._rpc.request("nvim_win_get_cursor", 0) == [1, 4]
        await session.open_file(str(file), file, line=2)
        assert await session._rpc.request("nvim_win_get_cursor", 0) == [2, 0]



def _clipboard_config(home: Path, *lua: str) -> None:
    """A config like the developer's, `clipboard=unnamedplus` so plain `p` reads the `+` register, then *lua*."""
    (home / "init.lua").write_text("\n".join(['vim.o.clipboard = "unnamedplus"', *lua]))


# User configs whose provider is already in use when the session starts.
_HOST_PROVIDER = [
    'local function host() return { { "from the host" }, "v" } end',
    'vim.g.clipboard = { name = "host", copy = { ["+"] = host, ["*"] = host }, paste = { ["+"] = host, ["*"] = host } }',
    'vim.fn.getreg("+")',
]
_NO_PROVIDER = ["vim.g.loaded_clipboard_provider = 1", 'vim.fn.has("clipboard")']


def _notes(directory: Path, text: str = "abc\n") -> Path:
    file = directory / "notes.md"
    file.write_text(text)
    return file


@requires_nvim
@pytest.mark.parametrize("keys", [b'"+p', b'"*p', b"p"])
async def test_paste_puts_the_text_of_the_browsers_clipboard_in_charwise(neovim_home, tmp_path, keys):
    _clipboard_config(neovim_home)
    file = _notes(tmp_path)
    asked: list[str] = []

    async def read_clipboard(path: str) -> str:
        asked.append(path)
        return "XY\nZ"

    async with _editing(file, neovim_home, _store, read_clipboard=read_clipboard) as session:
        session.write_input(keys)

        await _evaluates_to(session, "getline(1, '$')", ["aXY", "Zbc"])
        assert asked == [str(file)]


@requires_nvim
async def test_paste_of_text_ending_in_a_newline_goes_below_the_cursor_line(neovim_home, tmp_path):
    _clipboard_config(neovim_home)
    file = _notes(tmp_path, "a\nb\n")

    async def read_clipboard(path: str) -> str:
        return "x\ny\n"

    async with _editing(file, neovim_home, _store, read_clipboard=read_clipboard) as session:
        session.write_input(b'"+p')

        await _evaluates_to(session, "[getline(1, '$'), line('.')]", [["a", "x", "y", "b"], 2])


@requires_nvim
async def test_paste_of_an_empty_text_changes_nothing(neovim_home, tmp_path):
    _clipboard_config(neovim_home)
    file = _notes(tmp_path)

    async def read_clipboard(path: str) -> str:
        return ""

    async with _editing(file, neovim_home, _store, read_clipboard=read_clipboard) as session:
        session.write_input(b'"+px')  # `x` runs once the paste finished, on the text the paste left alone

        await _evaluates_to(session, "getline(1, '$')", ["bc"])


@requires_nvim
@pytest.mark.parametrize(
    ("yank", "text"), [(b"", ["abc"]), (b"yy", ["abc", "abc"]), (b"yl", ["aabc"])], ids=["nothing yanked", "line", "character"]
)
async def test_clipboard_error_shows_the_message_and_pastes_the_last_yank(neovim_home, tmp_path, yank, text):
    _clipboard_config(neovim_home)
    file = _notes(tmp_path)

    async def read_clipboard(path: str) -> str:
        raise ClipboardError("Clipboard access denied")

    async with _editing(file, neovim_home, _store, read_clipboard=read_clipboard) as session:
        session.write_input(yank + b'"+p')

        await _evaluates_to(session, "getline(1, '$')", text)
        await _shows(session, "Clipboard access denied")


@requires_nvim
async def test_no_answer_pastes_nothing_after_the_timeout_and_stops_waiting(neovim_home, tmp_path, monkeypatch):
    monkeypatch.setattr(neovim_session, "_CLIPBOARD_SECONDS", 0.3)
    _clipboard_config(neovim_home)
    file = _notes(tmp_path)
    stopped = asyncio.Event()

    async def read_clipboard(path: str) -> str:
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return ""

    async with _editing(file, neovim_home, _store, read_clipboard=read_clipboard) as session:
        session.write_input(b'"+p')

        await _shows(session, "Clipboard did not answer")
        assert await session._rpc.request("nvim_eval", "[getline(1, '$'), &modified]") == [["abc"], 0]
        assert stopped.is_set()


@requires_nvim
@pytest.mark.parametrize("user_config", [_HOST_PROVIDER, _NO_PROVIDER], ids=["provider of its own", "no provider"])
async def test_the_browsers_clipboard_replaces_the_provider_of_the_users_config(neovim_home, tmp_path, user_config):
    _clipboard_config(neovim_home, *user_config)
    file = _notes(tmp_path)

    async def read_clipboard(path: str) -> str:
        return "XY"

    async with _editing(file, neovim_home, _store, read_clipboard=read_clipboard) as session:
        session.write_input(b'"+p')
        await _evaluates_to(session, "getline(1, '$')", ["aXYbc"])

        session.write_input(b"p")
        await _evaluates_to(session, "getline(1, '$')", ["aXYXYbc"])


@requires_nvim
async def test_a_session_without_a_clipboard_reader_keeps_the_provider_of_the_users_config(neovim_home, tmp_path):
    _clipboard_config(neovim_home, *_HOST_PROVIDER)
    file = _notes(tmp_path)
    async with _editing(file, neovim_home, _store) as session:
        session.write_input(b'"+p')

        await _evaluates_to(session, "getline(1, '$')", ["afrom the hostbc"])


@requires_nvim
async def test_requests_are_answered_while_a_paste_waits_for_the_browser(neovim_home, tmp_path):
    _clipboard_config(neovim_home)
    file = _notes(tmp_path)
    asking, answer = asyncio.Event(), asyncio.Event()

    async def read_clipboard(path: str) -> str:
        asking.set()
        await answer.wait()
        return "XY"

    async with _editing(file, neovim_home, _store, read_clipboard=read_clipboard) as session:
        session.write_input(b'"+p')
        await asyncio.wait_for(asking.wait(), 10)

        await asyncio.wait_for(session.set_text("abc\ndef\n", str(file)), 5)
        answer.set()

        await _evaluates_to(session, "getline(1, '$')", ["aXYbc", "def"])


class _Socket:
    """The part of the websocket that `_BrowserSaves` writes to."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, frame: dict) -> None:
        self.sent.append(frame)


async def _asking(saves, socket: _Socket, path: str = "notes.md") -> tuple[asyncio.Task[str], int]:
    """Start a clipboard read and wait until the browser was asked, with the id of the question."""
    questions = len(socket.sent)
    read = asyncio.create_task(saves.read_clipboard(path))

    async def asked() -> bool:
        return len(socket.sent) > questions

    await _until(asked)
    question = socket.sent[-1]
    assert (question["type"], question["path"]) == ("clipboard_read", path)
    return read, question["id"]


@pytest.mark.parametrize(("reply", "text"), [({"text": "pasted"}, "pasted"), ({"text": ""}, ""), ({"text": 5}, ""), ({}, "")])
async def test_clipboard_read_returns_the_text_the_browser_answers(reply, text):
    socket = _Socket()
    saves = neovim_route._BrowserSaves(socket)
    read, read_id = await _asking(saves, socket)

    saves.resolve(read_id, reply)

    assert await asyncio.wait_for(read, 5) == text


async def test_clipboard_read_raises_the_error_the_browser_answers():
    socket = _Socket()
    saves = neovim_route._BrowserSaves(socket)
    read, read_id = await _asking(saves, socket)

    saves.resolve(read_id, {"error": "Clipboard access denied"})

    with pytest.raises(ClipboardError, match="Clipboard access denied"):
        await asyncio.wait_for(read, 5)


class _Browser:
    """The browser's end of the session websocket, driven by the test."""

    def __init__(self) -> None:
        self.headers = {"origin": "http://localhost:2999"}
        self.client_state = WebSocketState.CONNECTED
        self._incoming: asyncio.Queue[dict] = asyncio.Queue()
        self._frames: asyncio.Queue[dict] = asyncio.Queue()
        self.terminal = bytearray()
        self.closed: int | None = None

    async def accept(self) -> None:
        pass

    async def close(self, code: int = 1000) -> None:
        self.closed = code

    async def receive_json(self) -> dict:
        return json.loads((await self._incoming.get())["text"])

    async def receive(self) -> dict:
        return await self._incoming.get()

    async def send_json(self, frame: dict) -> None:
        await self._frames.put(frame)

    async def send_bytes(self, data: bytes) -> None:
        self.terminal += data

    def send(self, frame: dict) -> None:
        self._incoming.put_nowait({"type": "websocket.receive", "text": json.dumps(frame)})

    def type(self, keys: bytes) -> None:
        self._incoming.put_nowait({"type": "websocket.receive", "bytes": keys})

    def leave(self) -> None:
        self._incoming.put_nowait({"type": "websocket.disconnect"})

    async def next_frame(self, kind: str) -> dict:
        """The next frame of type *kind* the server sent, skipping the others."""

        async def receive() -> dict:
            while (frame := await self._frames.get())["type"] != kind:
                pass
            return frame

        return await asyncio.wait_for(receive(), 10)

    async def frames(self, count: int) -> list[dict]:
        """The next *count* JSON frames the server sent, in order."""
        return [await asyncio.wait_for(self._frames.get(), 10) for _ in range(count)]

    async def silent(self, seconds: float = 0.3) -> None:
        """No frame arrives within *seconds*."""
        await asyncio.sleep(seconds)
        assert self._frames.empty(), f"unexpected frame {self._frames.get_nowait()}"

    async def shows(self, text: str) -> None:
        """Wait until the terminal output holds *text*."""

        async def holds() -> bool:
            return text.encode() in self.terminal

        await _until(holds)


@contextlib.asynccontextmanager
async def _browsing(monkeypatch, file: Path, **opening):
    """The session endpoint serving a browser that opened *file* (or what *opening* says), Neovim connected once the body starts."""
    monkeypatch.setattr(websocket_origin, "CORS_ORIGINS", ["*"])
    browser = _Browser()
    serving = asyncio.create_task(neovim_route.neovim_session(browser, request_identity(), mounted_roots(file.parent)))
    document = {"path": str(file), "filename": file.name, "content": file.read_text()}
    browser.send({"type": "open", **document, **opening, "rows": 24, "cols": 80})
    try:
        # A message frame is only shown once Neovim is connected, which is when the clipboard is the browser's
        browser.send({"type": "message", "text": "Connected"})
        await browser.shows("Connected")
        yield browser
    finally:
        browser.leave()
        await asyncio.wait_for(serving, 10)


@requires_nvim
async def test_paste_asks_the_browser_for_the_shown_file_and_pastes_its_answer(neovim_home, tmp_path, monkeypatch):
    _clipboard_config(neovim_home)
    file = _notes(tmp_path)
    async with _browsing(monkeypatch, file) as browser:
        browser.type(b'"+p:w\r')

        read = await browser.next_frame("clipboard_read")
        assert read["path"] == str(file)
        browser.send({"type": "clipboard", "id": read["id"], "text": "XY"})

        save = await browser.next_frame("save")
        assert save["content"] == "aXYbc\n"
        browser.send({"type": "saved", "id": save["id"], "content": save["content"]})


@requires_nvim
async def test_message_frame_shows_its_text_in_neovim(neovim_home, tmp_path, monkeypatch):
    file = _notes(tmp_path)
    async with _browsing(monkeypatch, file) as browser:
        browser.send({"type": "message", "text": "Image is too large"})

        await browser.shows("Image is too large")


@requires_nvim
async def test_frames_that_fit_no_message_are_ignored(neovim_home, tmp_path, monkeypatch):
    file = _notes(tmp_path)
    async with _browsing(monkeypatch, file) as browser:
        browser.send({"type": "insert_lines", "text": 5})
        browser.send({"type": "insert_image", "path": "imgs/cat.png"})
        browser.send({"type": "teleport"})
        browser.send({"type": "insert_lines", "text": "kept"})
        await browser.shows("kept")
        browser.type(b":w\r")

        save = await browser.next_frame("save")
        assert save["content"] == "abc\nkept\n"
        browser.send({"type": "saved", "id": save["id"], "content": save["content"]})


@requires_nvim
async def test_insert_frame_frame_puts_the_slide_into_the_shown_file(neovim_home, tmp_path, monkeypatch):
    file = _notes(tmp_path)
    async with _browsing(monkeypatch, file) as browser:
        browser.send({"type": "insert_frame", "text": "\n".join(NEW_FRAME)})
        await browser.shows("{New}")
        browser.type(b":w\r")

        save = await browser.next_frame("save")
        assert save["content"] == "abc\n\n" + "\n".join(NEW_FRAME) + "\n"
        browser.send({"type": "saved", "id": save["id"], "content": save["content"]})


@requires_nvim
async def test_insert_image_frame_fills_the_placeholder_of_the_shown_file(neovim_home, tmp_path, monkeypatch):
    file = _notes(tmp_path, "\\roundimage{example-image}\n")
    async with _browsing(monkeypatch, file) as browser:
        browser.send({"type": "insert_image", "path": "imgs/cat.png", "reference": "\\roundimage{imgs/cat.png}"})
        await browser.shows("imgs/cat.png")
        browser.type(b":w\r")

        save = await browser.next_frame("save")
        assert save["content"] == "\\roundimage{imgs/cat.png}\n"
        browser.send({"type": "saved", "id": save["id"], "content": save["content"]})


@requires_nvim
async def test_open_file_frame_with_a_line_scrolls_the_shown_file_to_it(neovim_home, tmp_path, monkeypatch):
    file = tmp_path / "main.tex"
    file.write_text("x\n" * 79 + "SYNCTARGET\n" + "x\n" * 20)
    async with _browsing(monkeypatch, file) as browser:
        browser.send({"type": "open_file", "path": str(file), "line": 80})

        await browser.shows("SYNCTARGET")


@requires_nvim
async def test_leader_v_in_a_tex_file_sends_a_forward_search_frame_with_the_cursor_line(neovim_home, tmp_path, monkeypatch):
    file = tmp_path / "main.tex"
    file.write_text("a\nb\nc\n")
    async with _browsing(monkeypatch, file) as browser:
        browser.type(b"2G\\v")

        assert await browser.next_frame("forward_search") == {"type": "forward_search", "path": str(file), "line": 2}


SHIFT_ENTER, CTRL_ENTER = b"\x1b[13;2u", b"\x1b[13;5u"  # what the browser sends, Neovim reads them as <S-CR> and <C-CR>
NOTEBOOK_TEXT = "# %%\nimport os\nx = 1\n\n# %% [markdown]\n# # Title\n\n# %%\nprint(os.sep)\n"


def _notebook(directory: Path) -> tuple[Path, Path]:
    """A notebook in a new `project` folder of *directory* and the virtual name of its text beside it."""
    notebook = directory / "project" / "analysis.ipynb"
    notebook.parent.mkdir()
    notebook.write_text("{}")
    return notebook, Path(f"{notebook}.py")


@contextlib.asynccontextmanager
async def _notebook_text(home: Path, text_name: Path, save=_store, on_run_cell=None):
    """A connected session on the notebook text *text_name* (path = that name), closed afterwards."""
    session = NeovimSession.spawn(text_name.name, NOTEBOOK_TEXT, rows=24, cols=80, cwd=home, path=str(text_name), notebook=text_name)
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(save, lambda text: None, None, None, None, on_run_cell), 15)
        yield session
    finally:
        await session.close()
        await drain


@pytest.mark.parametrize(
    ("path", "named"),
    [
        ("project/nb.ipynb.py", "nb.ipynb.py"),
        ("project/../project/nb.ipynb.jl", "nb.ipynb.jl"),
        ("project/missing.ipynb.py", None),
        ("project/sub/../../outside/nb.ipynb.py", None),
        ("project/nb.ipynb", None),
        ("project/nb.ipynb.toolongext", None),
        ("project/nb.ipynb.p y", None),
        ("project/notes.md.py", None),
    ],
)
def test_only_a_notebook_text_beside_a_notebook_under_the_file_roots_gets_a_buffer_name(tmp_path, path, named):
    for folder in ("project", "outside"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "nb.ipynb").write_text("{}")
    (tmp_path / "project" / "notes.md").write_text("")
    roots = mounted_roots(tmp_path / "project")

    expected = (tmp_path / "project").resolve() / named if named else None
    assert neovim_route.notebook_buffer(roots, str(tmp_path / path)) == expected
    assert neovim_route.notebook_buffer(roots, "project/nb.ipynb.py") is None  # relative


@requires_nvim
async def test_opening_a_notebook_text_without_its_notebook_is_refused(neovim_home, tmp_path, monkeypatch):
    monkeypatch.setattr(websocket_origin, "CORS_ORIGINS", ["*"])
    browser = _Browser()
    serving = asyncio.create_task(neovim_route.neovim_session(browser, request_identity(), mounted_roots(tmp_path)))
    opening = {"path": str(tmp_path / "missing.ipynb.py"), "filename": "missing.ipynb.py", "content": "", "notebook": True}
    browser.send({"type": "open", **opening, "rows": 24, "cols": 80})

    await asyncio.wait_for(serving, 10)
    assert browser.closed == 1003


@requires_nvim
async def test_no_kind_of_write_creates_the_virtual_file_or_reads_one_that_is_there(neovim_home, tmp_path):
    """`:w`, `:w!`, `:up` and `:sav!` save through the caller, `:1,2w!` and `:1w >>` are refused, `:e!` reloads the saved text.

    `:r` of the virtual name reads nothing, another file still is. `:sav!` gives the text its virtual name back, `:e #` finds no
    buffer left under that name, a later `:w` writes nothing there (Neovim may ask for `:w!`, the name looks not edited) and
    `:w!` saves through the caller."""
    saved: asyncio.Queue[str] = asyncio.Queue()

    async def save(path: str, text: str) -> str:
        await saved.put(text)
        return text

    _notebook_file, name = _notebook(tmp_path)
    name.write_text("stale = 1\n")
    async with _notebook_text(neovim_home, name, save) as session:
        state = "[expand('%:p'), getline(1, '$'), &filetype, &swapfile, &undofile, &modified, undotree().seq_last]"
        assert await session._rpc.request("nvim_eval", state) == [str(name), NOTEBOOK_TEXT.splitlines(), "python", 0, 0, 0, 0]
        # Started on an empty placeholder file like `nvim file`, its buffer is gone
        assert await session._rpc.request("nvim_eval", "[argc(), map(getbufinfo(), 'v:val.name')]") == [1, [str(name)]]

        session.write_input(b"2GAx\x1b:w\r")
        assert await asyncio.wait_for(saved.get(), 10) == NOTEBOOK_TEXT.replace("import os", "import osx")
        session.write_input(b"Ay\x1b:e!\r")
        await _evaluates_to(session, "[getline(2), &modified]", ["import osx", 0])
        for command in (b":w!\r", b":up\r", b":1,2w!\r", b":1w >>\r", b":sav! copy.py\r"):
            session.write_input(command)
            await asyncio.sleep(0.2)
        assert [await asyncio.wait_for(saved.get(), 10) for _ in range(2)] == [NOTEBOOK_TEXT.replace("import os", "import osx")] * 2
        await _evaluates_to(session, "expand('%:p')", str(name))

        (neovim_home / "other.txt").write_text("other = 2\n")  # in Neovim's working directory
        name_from_cwd = os.path.relpath(name, neovim_home).encode()
        commands = (b":e #\r", b":r\r", b":r %\r", b":r " + name_from_cwd + b"\r", b":3r other.txt\r", b"2GAz\x1b:w\r", b":w!\r")
        for command in commands:
            session.write_input(command)
            await asyncio.sleep(0.2)
        text = NOTEBOOK_TEXT.replace("import os", "import osxz").replace("x = 1\n", "x = 1\nother = 2\n")
        assert await asyncio.wait_for(saved.get(), 10) == text
        await _evaluates_to(session, "[expand('%:p'), &modified]", [str(name), 0])

    assert sorted(path.name for path in name.parent.iterdir()) == ["analysis.ipynb", "analysis.ipynb.py"]
    assert name.read_text() == "stale = 1\n"


@requires_nvim
async def test_a_renamed_notebook_text_keeps_its_virtual_name_and_writes_stay_intercepted(neovim_home, tmp_path):
    saved: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

    async def save(path: str, text: str) -> str:
        await saved.put((path, text))
        return text

    notebook, name = _notebook(tmp_path)
    name.write_text("stale = 1\n")
    async with _notebook_text(neovim_home, name, save) as session:
        session.write_input(b"2GAx\x1b")
        await _evaluates_to(session, "&modified", 1)
        renamed = notebook.rename(notebook.with_name("report.ipynb"))
        assert session.is_notebook_text(str(name))
        with pytest.raises(RuntimeError, match="no notebook text"):
            await session.rename(str(name), str(name.with_name("report.md")), None)

        new_name = Path(f"{renamed}.py")
        await session.rename(str(name), str(new_name), new_name)
        state = "[expand('%:p'), &modified, &filetype, getline(2)]"
        assert await session._rpc.request("nvim_eval", state) == [str(new_name), 1, "python", "import osx"]
        session.write_input(b":e #\r")
        await asyncio.sleep(0.2)
        assert await session._rpc.request("nvim_eval", state) == [str(new_name), 1, "python", "import osx"]
        session.write_input(b":w\r")
        assert await asyncio.wait_for(saved.get(), 10) == (str(new_name), NOTEBOOK_TEXT.replace("import os", "import osx"))

    assert sorted(path.name for path in name.parent.iterdir()) == ["analysis.ipynb.py", "report.ipynb"]
    assert name.read_text() == "stale = 1\n"


async def _shows_renamed(browser: _Browser, filename: str) -> None:
    """Wait until the shown buffer is named *filename*, the rename runs beside the relay."""

    async def renamed() -> bool:
        browser.type(b":echo 'shown:' . expand('%:t')\r")
        await asyncio.sleep(0.1)
        return f"shown:{filename}".encode() in browser.terminal

    await _until(renamed)


@requires_nvim
async def test_renamed_frame_moves_a_file_to_its_new_path_and_merges_with_the_renamed_file(neovim_home, tmp_path, monkeypatch):
    file = _notes(tmp_path, "a\nb\nc\n")
    async with _browsing(monkeypatch, file) as browser:
        renamed = file.rename(file.with_name("ideas.md"))
        renamed.write_text("a\nb\nc\nd\n")
        browser.send({"type": "renamed", "from": str(file), "path": str(renamed)})
        await _shows_renamed(browser, "ideas.md")

        browser.type(b"ggAx\x1b:w\r")
        save = await browser.next_frame("save")
        assert (save["path"], save["content"]) == (str(renamed), "ax\nb\nc\nd\n")
        browser.send({"type": "saved", "id": save["id"], "content": save["content"]})


@requires_nvim
async def test_renamed_frame_gives_a_notebook_text_the_virtual_name_of_the_renamed_notebook(neovim_home, tmp_path, monkeypatch):
    notebook, name = _notebook(tmp_path)
    async with _browsing(monkeypatch, notebook, path=str(name), filename=name.name, content=NOTEBOOK_TEXT, notebook=True) as browser:
        new_name = Path(f"{notebook.rename(notebook.with_name('report.ipynb'))}.py")
        browser.send({"type": "renamed", "from": str(name), "path": str(new_name)})
        await _shows_renamed(browser, new_name.name)

        browser.type(b":w\r")
        save = await browser.next_frame("save")
        assert (save["path"], save["content"]) == (str(new_name), NOTEBOOK_TEXT)
        browser.send({"type": "saved", "id": save["id"], "content": save["content"]})

    assert sorted(path.name for path in name.parent.iterdir()) == ["report.ipynb"]


@requires_nvim
async def test_ctrl_enter_saves_then_runs_the_cell_at_the_cursor_and_a_failed_save_runs_nothing(neovim_home, tmp_path, monkeypatch):
    notebook, name = _notebook(tmp_path)
    async with _browsing(monkeypatch, notebook, path=str(name), filename=name.name, content=NOTEBOOK_TEXT, notebook=True) as browser:
        browser.type(b"3GAz")
        browser.type(CTRL_ENTER)
        save = (await browser.frames(1))[0]
        assert (save["type"], save["path"], save["content"]) == ("save", str(name), NOTEBOOK_TEXT.replace("x = 1", "x = 1z"))
        browser.send({"type": "saved", "id": save["id"], "content": save["content"]})
        assert await browser.frames(1) == [{"type": "run_cell", "path": str(name), "line": 3}]

        browser.type(CTRL_ENTER)
        save = await browser.next_frame("save")
        browser.send({"type": "saved", "id": save["id"], "error": "stale revision"})
        await browser.silent()


@requires_nvim
async def test_shift_enter_runs_the_cell_and_moves_on_to_the_next_cell_appending_one_after_the_last(neovim_home, tmp_path):
    runs: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    _notebook_file, name = _notebook(tmp_path)
    async with _notebook_text(neovim_home, name, on_run_cell=lambda path, line: runs.put_nowait((path, line))) as session:
        session.write_input(b"2G" + SHIFT_ENTER)
        assert await asyncio.wait_for(runs.get(), 10) == (str(name), 2)
        await _evaluates_to(session, "line('.')", 6)  # past the markdown marker with its kind

        session.write_input(b"i" + SHIFT_ENTER)
        assert await asyncio.wait_for(runs.get(), 10) == (str(name), 6)
        await _evaluates_to(session, "[line('.'), mode()]", [9, "i"])

        session.write_input(SHIFT_ENTER)
        assert await asyncio.wait_for(runs.get(), 10) == (str(name), 9)
        lines = [*NOTEBOOK_TEXT.splitlines(), "", "# %%", ""]
        await _evaluates_to(session, "[getline(1, '$'), line('.'), mode(), &modified]", [lines, 12, "i", 1])
