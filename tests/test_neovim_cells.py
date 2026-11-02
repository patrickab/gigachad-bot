"""Notebook cells in Neovim: scratch buffers, their notifications, and the frames the session socket carries."""

import asyncio
import contextlib

import pytest

from backend import websocket_origin
from backend.routes import neovim as neovim_route
from lib.neovim_session import NeovimSession
from tests.conftest import mounted_roots, request_identity
from tests.test_neovim import _Browser, _drain, _evaluates_to, neovim_home, requires_nvim  # noqa: F401

SHIFT_ENTER = b"\x1b[13;2u"  # what the browser sends for Shift+Enter, Neovim reads it as <S-CR>
DOWN, UP = b"\x1b[B", b"\x1b[A"
ESC = b"\x1b"


async def _save(path: str, text: str) -> str:
    return text


@contextlib.asynccontextmanager
async def _cells(home):
    """A connected notebook session and the queue of cell events it reports."""
    events: asyncio.Queue[dict] = asyncio.Queue()
    session = NeovimSession.spawn("placeholder.txt", "", rows=24, cols=80, cwd=home, path="placeholder", cell_session=True)
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(_save, lambda text: None, on_cell=events.put_nowait), 15)
        yield session, events
    finally:
        await session.close()
        await drain


async def _type(session: NeovimSession, *keys: bytes) -> None:
    """Send each of *keys* on its own, as the browser does: an Escape glued to the next key would read as Alt."""
    for key in keys:
        session.write_input(key)
        await asyncio.sleep(0.1)


async def _next(events: asyncio.Queue[dict]) -> dict:
    return await asyncio.wait_for(events.get(), 10)


async def _quiet(events: asyncio.Queue[dict], seconds: float = 0.4) -> None:
    """No event arrives within *seconds*, long past the debounce."""
    await asyncio.sleep(seconds)
    assert events.empty(), f"unexpected event {events.get_nowait()}"


async def _eval(session: NeovimSession, expression: str) -> object:
    return await session._rpc.request("nvim_eval", expression)


async def _cell_text(session: NeovimSession) -> str:
    return await _eval(session, "join(getline(1, '$'), \"\\n\")")


@requires_nvim
@pytest.mark.parametrize("text", ["x = 1", "", "a\n", "a\n\n", "\nb", "é\tz\n  y", "line1\nline2\nline3"])
async def test_open_cell_holds_exactly_the_text_in_an_unlisted_scratch_buffer(neovim_home, text):
    async with _cells(neovim_home) as (session, events):
        await session.open_cell("tab#1", "cell.py", text, "first")

        assert await _cell_text(session) == text
        assert await _eval(session, "&filetype") == "python"
        assert await _eval(session, "&buflisted") == 0
        assert await _eval(session, "&modified") == 0
        # The placeholder is gone, and the first text is no undo step
        assert await _eval(session, "len(getbufinfo())") == 1
        session.write_input(b"u")
        await _quiet(events)
        assert await _cell_text(session) == text


@requires_nvim
@pytest.mark.parametrize(("cursor", "line"), [("first", 1), ("last", 3)])
async def test_open_cell_places_the_cursor_and_switching_keeps_every_cells_buffer(neovim_home, cursor, line):
    async with _cells(neovim_home) as (session, events):
        await session.open_cell("tab#1", "cell.py", "one", "first")
        await session.open_cell("tab#2", "cell.py", "a\nb\nc", cursor)

        assert await _eval(session, "line('.')") == line
        assert await _eval(session, "col('.')") == 1
        assert await _cell_text(session) == "a\nb\nc"

        # An edit typed in the first cell is still undoable after a visit to the other one
        await session.open_cell("tab#1", "cell.py", "one", "first")
        session.write_input(b"Ax")
        assert (await _next(events))["text"] == "onex"
        await session.open_cell("tab#2", "cell.py", "a\nb\nc", "first")
        await session.open_cell("tab#1", "cell.py", "onex", "last")
        assert await _cell_text(session) == "onex"
        await _type(session, ESC, b"u")
        assert (await _next(events))["text"] == "one"


@requires_nvim
async def test_typing_notifies_the_full_text_once_per_change_and_never_a_repeat(neovim_home):
    async with _cells(neovim_home) as (session, events):
        await session.open_cell("tab#1", "cell.py", "a\nb", "first")
        await _quiet(events)  # opening is no edit

        session.write_input(b"Ixyz")
        seen = [await _next(events)]
        while seen[-1]["text"] != "xyza\nb":
            seen.append(await _next(events))
        assert all(event == {"type": "cell_edit", "key": "tab#1", "text": event["text"]} for event in seen)
        assert [event["text"] for event in seen] == list(dict.fromkeys(event["text"] for event in seen))  # no two alike
        await _quiet(events)

        # Typing and undoing within the debounce leaves the text the caller knows, nothing to say
        await _type(session, ESC, b"Ok", ESC, b"u")
        await _evaluates_to(session, "getline(1, '$')", ["xyza", "b"])
        await asyncio.sleep(0.3)
        while not events.empty():
            assert (await _next(events))["text"] in {"xyza\nb", "k\nxyza\nb"}
        assert await _cell_text(session) == "xyza\nb"


@requires_nvim
async def test_set_cell_replaces_the_text_without_an_echo_and_keeps_the_cursor(neovim_home):
    async with _cells(neovim_home) as (session, events):
        await session.open_cell("tab#1", "cell.py", "one\ntwo\nthree", "first")
        session.write_input(b"jll")
        await _evaluates_to(session, "[line('.'), col('.')]", [2, 3])

        await session.set_cell("tab#1", "ONE\ntwo\nthree\nfour")

        assert await _cell_text(session) == "ONE\ntwo\nthree\nfour"
        assert await _eval(session, "[line('.'), col('.')]") == [2, 3]
        assert await _eval(session, "&modified") == 0
        await _quiet(events)

        # Same text, hidden cell, unknown cell: nothing changes and nothing is said
        await session.set_cell("tab#1", "ONE\ntwo\nthree\nfour")
        await session.set_cell("tab#unknown", "x")
        await _quiet(events)

        # The guard does not swallow later typing
        session.write_input(b"Ax")
        assert (await _next(events))["text"] == "ONE\ntwo\nthree\nfour".replace("two", "twox")


@requires_nvim
async def test_set_cell_reaches_a_hidden_cell_and_its_text_is_there_when_shown_again(neovim_home):
    async with _cells(neovim_home) as (session, events):
        await session.open_cell("tab#1", "cell.py", "one", "first")
        await session.open_cell("tab#2", "cell.py", "two", "first")

        await session.set_cell("tab#1", "changed\n")
        await _quiet(events)
        await session.open_cell("tab#1", "cell.py", "changed\n", "last")

        assert await _cell_text(session) == "changed\n"
        assert await _eval(session, "line('.')") == 2
        await _quiet(events)


@requires_nvim
@pytest.mark.parametrize("mode", ["normal", "insert"])
async def test_shift_enter_runs_the_cell_after_flushing_the_edit_that_precedes_it(neovim_home, mode):
    async with _cells(neovim_home) as (session, events):
        await session.open_cell("tab#1", "cell.py", "print(1)", "first")

        await _type(session, *((b"A!", ESC, SHIFT_ENTER) if mode == "normal" else (b"A!", SHIFT_ENTER)))
        order = []
        while not order or order[-1]["type"] != "cell_run":
            order.append(await _next(events))

        assert [event["type"] for event in order] == ["cell_edit", "cell_run"]
        assert order[0]["text"] == order[1]["text"] == "print(1)!"
        assert order[1]["key"] == "tab#1"
        assert await _cell_text(session) == "print(1)!"  # no newline inserted
        await _quiet(events)

        # Without an edit the run is alone
        session.write_input(SHIFT_ENTER)
        assert await _next(events) == {"type": "cell_run", "key": "tab#1", "text": "print(1)!"}


@requires_nvim
@pytest.mark.parametrize(("down", "up"), [(b"j", b"k"), (DOWN, UP)])
async def test_moving_past_the_last_or_first_line_leaves_the_cell_with_its_text(neovim_home, down, up):
    async with _cells(neovim_home) as (session, events):
        await session.open_cell("tab#1", "cell.py", "a\nb", "first")

        session.write_input(up)  # first line: out
        assert await _next(events) == {"type": "cell_exit", "key": "tab#1", "direction": "prev", "text": "a\nb"}

        session.write_input(down)  # inside: moves, nothing said
        await _evaluates_to(session, "line('.')", 2)
        await _quiet(events)

        await _type(session, b"A!", ESC, down)  # last line: the edit first, then out
        assert (await _next(events))["type"] == "cell_edit"
        assert await _next(events) == {"type": "cell_exit", "key": "tab#1", "direction": "next", "text": "a\nb!"}
        await _quiet(events)


@requires_nvim
async def test_a_count_moves_as_far_as_it_can_and_only_a_stuck_cursor_leaves(neovim_home):
    async with _cells(neovim_home) as (session, events):
        await session.open_cell("tab#1", "cell.py", "a\nb\nc", "first")

        session.write_input(b"5j")
        await _evaluates_to(session, "line('.')", 3)
        await _quiet(events)
        session.write_input(b"3j")
        assert (await _next(events))["direction"] == "next"
        session.write_input(b"gg" + b"3k")
        assert (await _next(events))["direction"] == "prev"


@requires_nvim
async def test_insert_mode_arrows_leave_the_cell_at_its_edges_and_otherwise_move(neovim_home):
    async with _cells(neovim_home) as (session, events):
        await session.open_cell("tab#1", "cell.py", "a\nb", "first")

        session.write_input(b"i" + DOWN)
        await _evaluates_to(session, "line('.')", 2)
        await _quiet(events)
        session.write_input(DOWN)
        assert await _next(events) == {"type": "cell_exit", "key": "tab#1", "direction": "next", "text": "a\nb"}
        session.write_input(UP + UP)
        await _evaluates_to(session, "line('.')", 1)
        assert (await _next(events))["direction"] == "prev"
        assert await _eval(session, "mode()") == "i"


@requires_nvim
async def test_exit_then_open_of_the_neighbour_lands_on_its_edge(neovim_home):
    async with _cells(neovim_home) as (session, events):
        await session.open_cell("tab#1", "cell.py", "a", "first")
        await session.open_cell("tab#2", "cell.py", "x\ny", "first")

        session.write_input(UP)
        assert (await _next(events))["key"] == "tab#2"
        await session.open_cell("tab#1", "cell.py", "a", "last")
        session.write_input(DOWN)
        assert await _next(events) == {"type": "cell_exit", "key": "tab#1", "direction": "next", "text": "a"}
        await session.open_cell("tab#2", "cell.py", "x\ny", "first")
        assert await _eval(session, "line('.')") == 1


@requires_nvim
async def test_write_in_a_cell_requests_a_notebook_save_after_the_last_edit(neovim_home):
    async with _cells(neovim_home) as (session, events):
        await session.open_cell("tab#1", "cell.py", "a", "first")

        await _type(session, b"Ab", ESC, b":w\r")
        order = []
        while not order or order[-1]["type"] != "save_notebook":
            order.append(await _next(events))

        assert [event["type"] for event in order] == ["cell_edit", "save_notebook"]
        assert order[-1] == {"type": "save_notebook", "key": "tab#1", "text": "ab"}
        assert await _eval(session, "&modified") == 0
        # Nothing was written for the cell anywhere
        copy = session._documents["tab#1"].copy
        assert copy.read_text() == "a"
        await _quiet(events)


@requires_nvim
async def test_close_cell_wipes_the_buffer_and_the_key_opens_again_from_the_given_text(neovim_home):
    async with _cells(neovim_home) as (session, events):
        await session.open_cell("tab#1", "cell.py", "a", "first")
        await session.open_cell("tab#2", "cell.py", "b", "first")
        session.write_input(b"Ax")
        assert (await _next(events))["text"] == "bx"

        await session.close_cell("tab#2")
        await session.close_cell("tab#2")  # twice, and a cell never opened
        await session.close_cell("tab#nothing")
        await session.set_cell("tab#2", "ignored")
        assert await _eval(session, "len(getbufinfo({'bufloaded': 1}))") == 2  # tab#1 and the empty buffer left behind

        await session.open_cell("tab#2", "cell.py", "fresh", "first")
        assert await _cell_text(session) == "fresh"
        await _type(session, ESC, b"u")  # nothing of the old buffer's undo history
        await _quiet(events)
        assert await _cell_text(session) == "fresh"


@requires_nvim
async def test_a_cell_buffer_deleted_in_neovim_opens_again_with_the_given_text(neovim_home):
    async with _cells(neovim_home) as (session, events):
        await session.open_cell("tab#1", "cell.py", "a", "first")
        await session.open_cell("tab#2", "cell.py", "b", "first")
        session.write_input(b":bwipeout!\r")
        await _evaluates_to(session, "bufname()", "")

        await session.open_cell("tab#2", "cell.py", "again", "first")

        assert await _cell_text(session) == "again"
        session.write_input(b"Ax")
        assert await _next(events) == {"type": "cell_edit", "key": "tab#2", "text": "againx"}


@requires_nvim
async def test_cells_never_report_switches_as_documents(neovim_home):
    switched = []
    events: asyncio.Queue[dict] = asyncio.Queue()
    session = NeovimSession.spawn("placeholder.txt", "", rows=24, cols=80, cwd=neovim_home, path="placeholder", cell_session=True)
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(_save, lambda text: None, None, switched.append, None, events.put_nowait), 15)
        await session.open_cell("tab#1", "cell.py", "a", "first")
        await session.open_cell("tab#2", "cell.py", "b", "first")
        await session.open_cell("tab#1", "cell.py", "a", "first")
        await asyncio.sleep(0.2)
        assert switched == []
    finally:
        await session.close()
        await drain


def test_open_message_needs_a_document_unless_it_is_a_notebook_session():
    size = {"rows": 24, "cols": 80}
    assert neovim_route.OpenMessage.model_validate({"cell_mode": True, **size}).cell_mode
    assert neovim_route.OpenMessage.model_validate({"path": "p", "filename": "f", "content": "", **size}).cell_mode is False
    for incomplete in ({"path": "p", "filename": "f"}, {"path": "p", "content": ""}, {"filename": "f", "content": ""}, {}):
        with pytest.raises(ValueError):
            neovim_route.OpenMessage.model_validate({**incomplete, **size})


@contextlib.asynccontextmanager
async def _notebook_browser(monkeypatch, tmp_path):
    """The session endpoint serving a browser that opened a notebook, with Neovim connected once the body starts."""
    monkeypatch.setattr(websocket_origin, "CORS_ORIGINS", ["*"])
    browser = _Browser()
    serving = asyncio.create_task(neovim_route.neovim_session(browser, request_identity(), mounted_roots(tmp_path)))
    browser.send({"type": "open", "cell_mode": True, "path": str(tmp_path / "nb.ipynb"), "rows": 24, "cols": 80})
    try:
        browser.send({"type": "message", "text": "Connected"})
        await browser.shows("Connected")
        yield browser
    finally:
        browser.leave()
        await asyncio.wait_for(serving, 10)


async def _frames(browser: _Browser, count: int) -> list[dict]:
    """The next *count* JSON frames of the server, in order, ignoring none."""
    return [await asyncio.wait_for(browser._frames.get(), 10) for _ in range(count)]


@requires_nvim
async def test_every_cell_frame_over_the_socket(neovim_home, tmp_path, monkeypatch):
    async with _notebook_browser(monkeypatch, tmp_path) as browser:
        browser.send({"type": "open_cell", "key": "t#1", "filename": "cell.py", "text": "gamma", "cursor": "first"})
        browser.send({"type": "open_cell", "key": "t#2", "filename": "cell.py", "text": "alpha\nbeta", "cursor": "last"})
        browser.send({"type": "set_cell", "key": "t#2", "text": "ALPHA\nbeta"})
        await browser.shows("ALPHA")  # the cells are open
        browser.type(b"Ak")  # on t#2
        edit = await browser.next_frame("cell_edit")
        assert edit == {"type": "cell_edit", "key": "t#2", "text": "ALPHA\nbetak"}

        browser.type(ESC)
        await asyncio.sleep(0.1)
        browser.type(SHIFT_ENTER)
        assert await browser.next_frame("cell_run") == {"type": "cell_run", "key": "t#2", "text": "ALPHA\nbetak"}

        browser.type(UP)
        browser.type(UP)  # first line of t#2: out
        assert await browser.next_frame("cell_exit") == {
            "type": "cell_exit",
            "key": "t#2",
            "direction": "prev",
            "text": "ALPHA\nbetak",
        }

        browser.send({"type": "open_cell", "key": "t#1", "filename": "cell.py", "text": "gamma", "cursor": "last"})
        browser.send({"type": "set_cell", "key": "t#1", "text": "QRSTU"})
        await browser.shows("QRSTU")
        browser.type(b":w\r")
        assert await browser.next_frame("save_notebook") == {"type": "save_notebook", "key": "t#1", "text": "QRSTU"}

        browser.send({"type": "close_cell", "key": "t#2"})
        browser.send({"type": "set_cell", "key": "t#2", "text": "late"})
        browser.send({"type": "open_cell", "key": 5, "filename": "cell.py", "text": "", "cursor": "first"})
        browser.send({"type": "open_cell", "key": "bad", "filename": "cell.py", "text": "", "cursor": "middle"})
        browser.send({"type": "message", "text": "Still alive"})
        await browser.shows("Still alive")


@requires_nvim
async def test_set_cell_over_the_socket_has_no_echo_and_frames_keep_their_order(neovim_home, tmp_path, monkeypatch):
    async with _notebook_browser(monkeypatch, tmp_path) as browser:
        browser.send({"type": "open_cell", "key": "t#1", "filename": "cell.py", "text": "a", "cursor": "first"})
        browser.send({"type": "set_cell", "key": "t#1", "text": "a\nb"})
        browser.send({"type": "set_cell", "key": "t#1", "text": "a\nb\nc"})
        browser.send({"type": "message", "text": "Settled"})
        await browser.shows("Settled")
        await asyncio.sleep(0.3)
        assert browser._frames.empty()

        # Edits and the run of a quick Shift+Enter arrive in the order they happened, round after round
        digits = ""
        browser.type(ESC)  # normal mode, the cursor on the first line
        await asyncio.sleep(0.1)
        for round_ in range(50):
            browser.type(b"A" + str(round_ % 10).encode() + SHIFT_ENTER)
            digits += str(round_ % 10)
            text = f"a{digits}\nb\nc"
            assert await _frames(browser, 2) == [
                {"type": "cell_edit", "key": "t#1", "text": text},
                {"type": "cell_run", "key": "t#1", "text": text},
            ]
            browser.type(ESC)
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.3)
        assert browser._frames.empty()
