import asyncio
import contextlib
from pathlib import Path
import shutil

import pytest
from starlette.websockets import WebSocketDisconnect

from backend.identity import get_request_identity
from backend import websocket_origin
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


def test_open_message_base_content_is_optional():
    fields = {"type": "open", "path": "/a", "filename": "a", "content": "x", "rows": 1, "cols": 1}
    assert neovim_route.OpenMessage.model_validate(fields).base_content is None
    assert neovim_route.OpenMessage.model_validate({**fields, "base_content": ""}).base_content == ""
    assert neovim_route.OpenMessage.model_validate({**fields, "base_content": None}).base_content is None


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
async def _editing(first: Path, home: Path, save, on_switched=None):
    """A connected session on *first* (path = its absolute path), closed afterwards."""
    session = NeovimSession.spawn(first.name, first.read_text(), rows=24, cols=80, cwd=home, disk_path=first, path=str(first))
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(save, lambda text: None, None, on_switched), 15)
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
