"""A notebook in Neovim: one buffer for all code cells, its window pinned to the active cell, and the frames the socket carries."""

import asyncio
import contextlib
from pathlib import Path
import re
import shutil

import pytest

from backend import websocket_origin
from backend.routes import neovim as neovim_route
from lib.neovim_render import parse_tohtml
from lib.neovim_session import NOTEBOOK_MARKER, NeovimSession
from tests.conftest import mounted_roots, request_identity
from tests.test_neovim import _Browser, _drain, _evaluates_to, neovim_home, requires_nvim  # noqa: F401

SHIFT_ENTER = b"\x1b[13;2u"  # what the browser sends for Shift+Enter, Neovim reads it as <S-CR>
DOWN, UP = b"\x1b[B", b"\x1b[A"
ESC, BACKSPACE, CTRL_E, CTRL_Y = b"\x1b", b"\x7f", b"\x05", b"\x19"
CELLS = [
    {"id": "a", "text": "import os\nx = 1"},
    {"id": "b", "text": "print(os.sep)\ny = 2\nz = 3"},
    {"id": "c", "text": "w = 4"},
]
# The script of CELLS has its markers on lines 1, 4 and 8, `FIRST` and `LAST` are the lines of each cell
SCRIPT = ["# %%", "import os", "x = 1", "# %%", "print(os.sep)", "y = 2", "z = 3", "# %%", "w = 4"]
MARKER_LINES = [1, 4, 8]
FIRST = {"a": 2, "b": 5, "c": 9}
LAST = {"a": 3, "b": 7, "c": 9}
MARKER_MESSAGE = "Cell marker lines cannot be changed"


def script_of(cells: list[dict[str, str]]) -> list[str]:
    return [line for cell in cells for line in (NOTEBOOK_MARKER, *cell["text"].split("\n"))] or [""]


def spans_of(cells: list[dict[str, str]]) -> dict[str, tuple[int, int]]:
    """First and last line (1-based) of every cell in the script of *cells*."""
    spans, line = {}, 0
    for cell in cells:
        count = len(cell["text"].split("\n"))
        spans[cell["id"]] = (line + 2, line + 1 + count)
        line += 1 + count
    return spans


async def _save(path: str, text: str) -> str:
    return text


class _Events(asyncio.Queue):
    """What a notebook session reports. The renders of the cells (`cell_render`) and the screen rows of the active cell (`cell_height`)
    are kept apart in `renders` and `heights`, they come on their own time.

    `log` holds every event of all three in the order they came.
    """

    def __init__(self) -> None:
        super().__init__()
        self.renders: asyncio.Queue[dict] = asyncio.Queue()
        self.heights: asyncio.Queue[dict] = asyncio.Queue()
        self.log: list[dict] = []

    def put_nowait(self, event: dict) -> None:
        self.log.append(event)
        aside = {"cell_render": self.renders, "cell_height": self.heights}.get(event["type"])
        if aside is None:
            super().put_nowait(event)
        else:
            aside.put_nowait(event)


@contextlib.asynccontextmanager
async def _notebook(home, cells=CELLS, *, rows=24, filename="nb.py", notebook_path=None, on_selection=None):
    """A connected notebook session showing *cells*, and the queue of the events it reports."""
    events = _Events()
    session = NeovimSession.spawn(
        "placeholder.txt", "", rows=rows, cols=80, cwd=home, path="placeholder", cell_session=True, notebook_path=notebook_path
    )
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(_save, lambda text: None, on_selection, on_cell=events.put_nowait), 15)
        await session.set_cells(filename, cells)
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


async def _settled(events: asyncio.Queue[dict], seconds: float = 0.4) -> list[dict]:
    """Everything reported until nothing came for *seconds*."""
    await asyncio.sleep(seconds)
    seen = []
    while not events.empty():
        seen.append(events.get_nowait())
    return seen


async def _eval(session: NeovimSession, expression: str) -> object:
    return await session._rpc.request("nvim_eval", expression)


async def _lines(session: NeovimSession) -> list[str]:
    return await _eval(session, "getline(1, '$')")


async def _marks(session: NeovimSession) -> list[int]:
    """The line (1-based) of every cell's extmark, in buffer order."""
    code = "local ns = vim.api.nvim_get_namespaces().gigachad_cells local rows = {} "
    code += "for _, m in ipairs(vim.api.nvim_buf_get_extmarks(0, ns, 0, -1, {})) do rows[#rows + 1] = m[2] + 1 end return rows"
    return await session._lua(code)


async def _hidden(session: NeovimSession) -> list[tuple[int, int]]:
    """The line ranges (1-based, first and last) the window does not draw."""
    code = "local ns = vim.api.nvim_get_namespaces().gigachad_cells_hidden local out = {} "
    code += "for _, m in ipairs(vim.api.nvim_buf_get_extmarks(0, ns, 0, -1, { details = true })) do "
    code += "out[#out + 1] = { m[2] + 1, m[4].end_row + 1 } end return out"
    return [tuple(pair) for pair in await session._lua(code)]


async def _messages(session: NeovimSession) -> str:
    return (await session._rpc.request("nvim_exec2", "messages", {"output": True}))["output"]


async def _reverted(session: NeovimSession) -> None:
    """Neovim said that it undid a change to a marker line."""
    for _ in range(200):
        if MARKER_MESSAGE in await _messages(session):
            return
        await asyncio.sleep(0.05)
    pytest.fail("no message about the marker lines")


async def _drawn(session: NeovimSession, count: int) -> list[str]:
    """The first *count* rows of the terminal screen, trailing blanks cut."""
    code = "local out = {} for row = 1, ... do local cells = {} for col = 1, vim.o.columns do "
    code += (
        "cells[#cells + 1] = vim.fn.screenstring(row, col) end out[#out + 1] = (table.concat(cells):gsub('%s+$', '')) end return out"
    )
    return await session._lua(code, count)


async def _pinned(session: NeovimSession, first: int, last: int) -> None:
    """The window starts at line *first*, the cursor is within [first, last], every line outside it is hidden and not drawn."""
    await _evaluates_to(session, "line('w0')", first)
    assert first <= await _eval(session, "line('.')") <= last

    total = await _eval(session, "line('$')")
    assert await _hidden(session) == ([(1, first - 1)] if first > 1 else []) + ([(last + 1, total)] if last < total else [])
    expected = [*await _eval(session, f"getline({first}, {last})"), "~"]
    for _ in range(100):
        if (drawn := await _drawn(session, len(expected))) == expected:
            return
        await asyncio.sleep(0.05)
    assert drawn == expected


async def _pinned_to(session: NeovimSession, cell: str) -> None:
    await _pinned(session, FIRST[cell], LAST[cell])


@requires_nvim
@pytest.mark.parametrize(
    "cells",
    [
        [{"id": "x", "text": "one"}],
        [{"id": "x", "text": ""}, {"id": "y", "text": "a\n"}, {"id": "z", "text": "\nb"}],
        [{"id": "x", "text": "é\tz\n  y"}, {"id": "y", "text": "a\n\n"}],
        [],
    ],
)
async def test_set_cells_builds_one_script_buffer_with_a_marker_before_each_cell(neovim_home, cells):
    async with _notebook(neovim_home, cells, filename="analysis.py") as (session, events):
        assert await _lines(session) == script_of(cells)
        assert await _eval(session, "&filetype") == "python"
        assert await _eval(session, "&buflisted") == 0
        assert await _eval(session, "&modified") == 0
        assert await _eval(session, "fnamemodify(bufname(), ':t')") == "analysis.py"
        assert len(await _marks(session)) == len(cells)
        # The placeholder is gone and the first text is no undo step
        assert await _eval(session, "len(getbufinfo())") == 1
        session.write_input(b"u")
        await _quiet(events)
        assert await _lines(session) == script_of(cells)


@requires_nvim
async def test_the_buffer_is_a_single_py_file_on_a_private_copy_that_nothing_writes_to(neovim_home):
    async with _notebook(neovim_home, filename="nb.py") as (session, _events):
        copy = Path(await _eval(session, "expand('%:p')"))
        assert copy.suffix == ".py"
        assert copy.parent.parent == session._workdir
        assert copy.read_text() == "\n".join(SCRIPT) + "\n"

        # Edits and `:w` leave the copy alone
        await _type(session, b"Ax", ESC, b":w\r")
        await _evaluates_to(session, "&modified", 0)
        assert copy.read_text() == "\n".join(SCRIPT) + "\n"
        assert list(neovim_home.iterdir()) == []

    assert not copy.exists()  # gone with the session


@requires_nvim
async def test_first_cell_is_active_with_the_cursor_on_its_first_line_and_the_window_pinned(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        assert await _eval(session, "[line('.'), col('.')]") == [2, 1]
        assert await _eval(session, "&scrolloff") == 0
        await _pinned_to(session, "a")
        await _quiet(events)


@requires_nvim
@pytest.mark.parametrize(
    ("cell", "cursor", "line"),
    [("b", "first", 5), ("b", "last", 7), ("c", "first", 9), ("a", "last", 3), ("a", "first", 2)],
)
async def test_focus_cell_pins_the_window_and_puts_the_cursor_without_an_echo(neovim_home, cell, cursor, line):
    async with _notebook(neovim_home) as (session, events):
        await session.focus_cell(cell, cursor)

        assert await _eval(session, "[line('.'), col('.')]") == [line, 1]
        await _pinned_to(session, cell)
        await _quiet(events)


@requires_nvim
async def test_focus_cell_keep_stays_inside_the_cell_and_an_unknown_cell_does_nothing(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await session.focus_cell("b", "last")
        session.write_input(b"0ll")
        await _evaluates_to(session, "col('.')", 3)

        await session.focus_cell("b", "keep")
        await session.focus_cell("nope", "first")
        assert await _eval(session, "[line('.'), col('.')]") == [7, 3]
        await _pinned_to(session, "b")

        await session.focus_cell("c", "keep")  # the cursor was in another cell: first line
        assert await _eval(session, "line('.')") == 9
        await _pinned_to(session, "c")
        await _quiet(events)


@requires_nvim
async def test_typing_reports_the_changed_cell_with_its_id_once_per_change(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await session.focus_cell("b", "first")
        await _quiet(events)  # showing a cell is no edit

        session.write_input(b"Ixyz")
        seen = [await _next(events)]
        while seen[-1]["text"] != "xyzprint(os.sep)\ny = 2\nz = 3":
            seen.append(await _next(events))
        assert all(event == {"type": "cell_edit", "id": "b", "text": event["text"]} for event in seen)
        assert [event["text"] for event in seen] == list(dict.fromkeys(event["text"] for event in seen))  # no two alike
        await _quiet(events)

        # Typing and undoing within the debounce leaves the text the caller knows, nothing to say
        await _type(session, ESC, b"Ok", ESC, b"u")
        await _evaluates_to(session, "getline(5, 7)", ["xyzprint(os.sep)", "y = 2", "z = 3"])
        settled = await _settled(events)
        assert all(event["id"] == "b" for event in settled)
        assert [event["text"] for event in settled] in (
            [],
            ["xyzprint(os.sep)\ny = 2\nz = 3"],
            ["k\nxyzprint(os.sep)\ny = 2\nz = 3", "xyzprint(os.sep)\ny = 2\nz = 3"],
        )
        assert await _lines(session) == [*SCRIPT[:4], "xyzprint(os.sep)", "y = 2", "z = 3", *SCRIPT[7:]]


@requires_nvim
async def test_a_change_across_several_cells_reports_every_cell_in_notebook_order_then_the_focus(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        session.write_input(b":%s/\\d/N/\r")

        assert [await _next(events) for _ in range(4)] == [
            {"type": "cell_edit", "id": "a", "text": "import os\nx = N"},
            {"type": "cell_edit", "id": "b", "text": "print(os.sep)\ny = N\nz = N"},
            {"type": "cell_edit", "id": "c", "text": "w = N"},
            {"type": "cell_focus", "id": "c"},  # the cursor ends on the last changed line
        ]
        await _quiet(events)


@requires_nvim
async def test_set_cells_with_the_same_structure_replaces_only_the_differing_text_without_an_echo(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await session.focus_cell("b", "first")
        session.write_input(b"jll")
        await _evaluates_to(session, "[line('.'), col('.')]", [6, 3])

        stamp = await _eval(session, "b:changedtick")
        await session.set_cells("nb.py", CELLS)  # nothing differs
        assert await _eval(session, "b:changedtick") == stamp

        changed = [CELLS[0], {"id": "b", "text": "print(os.sep)\nY = 2\nz = 3\nextra"}, {"id": "c", "text": "w = 5"}]
        await session.set_cells("nb.py", changed)

        assert await _lines(session) == script_of(changed)
        assert await _eval(session, "[line('.'), col('.')]") == [6, 3]
        assert await _eval(session, "&modified") == 0
        assert await _marks(session) == [1, 4, 9]
        await _pinned(session, 5, 8)
        await _quiet(events)

        # The same again is no event either, and typing is still reported afterwards
        await session.set_cells("nb.py", changed)
        await _quiet(events)
        session.write_input(b"A!")
        assert await _next(events) == {"type": "cell_edit", "id": "b", "text": "print(os.sep)\nY = 2!\nz = 3\nextra"}


@requires_nvim
async def test_set_cells_keeps_text_typed_since_the_caller_was_told_the_cell(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        # One call, so no 40 ms flush runs in between: "x" was typed and not reported when the old text comes back
        await session._lua(
            "vim.api.nvim_buf_set_text(0, 1, 0, 1, 0, { 'x' }) "
            "_G.gigachad_set_cells({ code = { marker = ..., cells = { { id = 'a', text = 'import os\\nx = 1' } } } }, { 'a' })",
            NOTEBOOK_MARKER,
        )
        assert (await _lines(session))[1] == "ximport os"
        # Nothing of cell b and c was sent: another structure, the typed text still stays
        await session.set_cells("nb.py", [{"id": "a", "text": "import os\nx = 1"}, CELLS[2]])
        assert (await _lines(session))[:3] == ["# %%", "ximport os", "x = 1"]
        assert await _next(events) == {"type": "cell_edit", "id": "a", "text": "ximport os\nx = 1"}


@requires_nvim
@pytest.mark.parametrize(
    ("cells", "active"),
    [
        ([CELLS[0], {"id": "new", "text": "n1\nn2"}, *CELLS[1:]], "b"),
        ([CELLS[0], CELLS[2]], "c"),
        ([CELLS[2], CELLS[1], CELLS[0]], "b"),
        ([*CELLS, {"id": "d", "text": "tail"}], "b"),
        ([{"id": "p", "text": "1"}, {"id": "q", "text": "2"}, {"id": "r", "text": "3"}], "q"),
        ([{"id": "p", "text": "1"}], "p"),
        ([], None),
    ],
    ids=["insert", "delete-middle", "reverse", "append", "new-ids", "fewer", "none"],
)
async def test_a_changed_list_of_cells_rebuilds_the_script_and_keeps_the_cursor_in_its_cell(neovim_home, cells, active):
    async with _notebook(neovim_home) as (session, events):
        await session.focus_cell("b", "first")
        session.write_input(b"jl")  # second line of the cell, second column
        await _evaluates_to(session, "[line('.'), col('.')]", [6, 2])
        await _quiet(events)

        await session.set_cells("nb.py", cells)

        assert await _lines(session) == script_of(cells)
        expected_marks = [line for line, text in enumerate(script_of(cells), 1) if text == NOTEBOOK_MARKER] if cells else []
        assert await _marks(session) == expected_marks
        assert await _eval(session, "&modified") == 0
        if active is not None:
            first, last = spans_of(cells)[active]
            await _pinned(session, first, last)
            if active == "b":  # the cell survived: same line within it, same column
                assert await _eval(session, "[line('.') - line('w0') + 1, col('.')]") == [2, 2]
        await _quiet(events)

        # The rebuilt buffer keeps reporting edits of its cells under their ids
        if cells:
            session.write_input(b"A.")
            event = await _next(events)
            assert event["type"] == "cell_edit"
            assert event["id"] in {cell["id"] for cell in cells}


@requires_nvim
async def test_the_cursor_is_in_the_active_cell_after_its_cell_went_away(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await session.focus_cell("c", "first")
        await session.set_cells("nb.py", CELLS[:2])  # the last cell is gone: its neighbor takes over

        await _pinned(session, *spans_of(CELLS[:2])["b"])
        await _quiet(events)


@requires_nvim
async def test_a_cell_that_reads_like_a_marker_is_text_and_markers_stay_protected(neovim_home):
    cells = [{"id": "a", "text": "x = 1\n# %%\ny = 2"}, {"id": "b", "text": "z = 3"}]
    async with _notebook(neovim_home, cells) as (session, events):
        assert await _lines(session) == ["# %%", "x = 1", "# %%", "y = 2", "# %%", "z = 3"]
        assert await _marks(session) == [1, 5]

        session.write_input(b"Ax")
        assert await _next(events) == {"type": "cell_edit", "id": "a", "text": "x = 1x\n# %%\ny = 2"}
        await _type(session, ESC, b":1d\r")
        await _reverted(session)
        assert await _lines(session) == ["# %%", "x = 1x", "# %%", "y = 2", "# %%", "z = 3"]
        assert await _marks(session) == [1, 5]

        # Rebuilding keeps counting the cell's own line apart from the markers
        await session.set_cells("nb.py", [*cells, {"id": "c", "text": "# %%"}])
        assert await _lines(session) == ["# %%", "x = 1", "# %%", "y = 2", "# %%", "z = 3", "# %%", "# %%"]
        assert await _marks(session) == [1, 5, 7]


async def _focus_and_wait(session: NeovimSession, cell: str, cursor: str) -> None:
    await session.focus_cell(cell, cursor)
    await _pinned_to(session, cell)


async def _marker_line_left_alone(session: NeovimSession, events: asyncio.Queue[dict]) -> None:
    """The script is what it was, the marker lines are still where the cells start, the cursor is inside a cell."""
    await _reverted(session)
    assert await _lines(session) == SCRIPT
    assert await _marks(session) == MARKER_LINES
    assert await _eval(session, "line('.')") not in MARKER_LINES
    assert await _eval(session, "&modified") == 0
    assert [event for event in await _settled(events) if event["type"] == "cell_edit"] == []


PROTECTED = {
    "delete-first-marker": (":1d\r", "a", "first"),
    "delete-middle-marker": (":4d\r", "a", "first"),
    "delete-last-marker": (":8d\r", "a", "first"),
    "delete-up-to-top": (b"dgg", "b", "first"),
    "delete-to-bottom": (b"dG", "a", "first"),
    "delete-down-onto-marker": (b"dj", "a", "last"),
    "delete-up-onto-marker": (b"dk", "b", "first"),
    "join-with-marker": (b"J", "a", "last"),
    "delete-everything": (":%d\r", "b", "first"),
    "alter-every-line": (":%s/^/x/\r", "a", "first"),
    "reverse-lines": (":g/^/m0\r", "a", "first"),
    "sort-lines": (":%!sort\r", "a", "first"),
    "put-a-marker": (":3put ='# %%'\r", "a", "first"),
    "backspace-into-marker": (b"i" + BACKSPACE, "b", "first"),
    "visual-delete-across-cells": (b"Vjjjd", "a", "first"),
    "change-marker-text": (":4s/%%/xx/\r", "b", "first"),
}


@requires_nvim
@pytest.mark.parametrize(("keys", "cell", "cursor"), PROTECTED.values(), ids=PROTECTED.keys())
async def test_a_change_to_a_marker_line_is_undone_with_a_message(neovim_home, keys, cell, cursor):
    async with _notebook(neovim_home) as (session, events):
        await _focus_and_wait(session, cell, cursor)
        session.write_input(keys if isinstance(keys, bytes) else keys.encode())

        await _marker_line_left_alone(session, events)


@requires_nvim
async def test_a_reverted_change_elsewhere_leaves_the_cursor_and_the_active_cell_alone(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await _focus_and_wait(session, "b", "last")
        session.write_input(b"$h")
        await _evaluates_to(session, "[line('.'), col('.')]", [7, 4])

        session.write_input(b":1d\r")  # a change in cell a, which the undo would put the cursor into

        await _reverted(session)
        assert await _lines(session) == SCRIPT
        assert await _eval(session, "[line('.'), col('.')]") == [7, 4]
        await _pinned_to(session, "b")
        await _quiet(events)


@requires_nvim
async def test_typing_a_marker_line_is_undone_even_in_insert_mode(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await _focus_and_wait(session, "b", "first")
        session.write_input(b"O# %%")

        await _reverted(session)
        assert await _lines(session) == SCRIPT
        session.write_input(ESC)
        edits = [event for event in await _settled(events) if event["type"] == "cell_edit"]
        assert edits == [] or edits[-1] == {"type": "cell_edit", "id": "b", "text": CELLS[1]["text"]}
        assert await _marks(session) == MARKER_LINES


@requires_nvim
async def test_undo_after_a_reverted_change_goes_back_before_the_change_before_it_and_stays_intact(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await _type(session, b"Ax", ESC)
        await _evaluates_to(session, "getline(2)", "import osx")
        await _type(session, b":4d\r")
        await _evaluates_to(session, "getline(1, '$')", [*SCRIPT[:1], "import osx", *SCRIPT[2:]])

        await _type(session, b"u")  # the reverted change left no step of its own behind
        await _evaluates_to(session, "getline(2)", "import os")
        assert await _lines(session) == SCRIPT
        await _type(session, b"u")
        assert await _lines(session) == SCRIPT
        assert await _marks(session) == MARKER_LINES
        assert [event["text"] for event in await _settled(events) if event["type"] == "cell_edit"][-1] == CELLS[0]["text"]


@requires_nvim
async def test_deleting_all_lines_of_a_cell_leaves_it_one_empty_line(neovim_home):
    cells = [{"id": "x", "text": "one"}, {"id": "y", "text": "two"}]
    async with _notebook(neovim_home, cells) as (session, events):
        await _type(session, b"dd")
        await _evaluates_to(session, "getline(1, '$')", ["# %%", "", "# %%", "two"])
        assert await _next(events) == {"type": "cell_edit", "id": "x", "text": ""}
        assert await _marks(session) == [1, 3]

        await session.focus_cell("y", "first")
        await _type(session, b"dd")
        await _evaluates_to(session, "getline(1, '$')", ["# %%", "", "# %%", ""])
        assert await _next(events) == {"type": "cell_edit", "id": "y", "text": ""}
        assert MARKER_MESSAGE not in await _messages(session)
        await _quiet(events)


@requires_nvim
async def test_moving_down_past_the_last_line_enters_the_next_cell_and_reports_the_focus(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await _type(session, b"jj")  # x = 1, then onto the marker: on into the next cell

        assert await _eval(session, "line('.')") == 5
        assert await _next(events) == {"type": "cell_focus", "id": "b"}
        await _pinned_to(session, "b")
        await _quiet(events)

        # Moving inside the cell is no news
        await _type(session, b"jj")
        await _pinned_to(session, "b")
        await _quiet(events)


@requires_nvim
async def test_moving_up_past_the_first_line_enters_the_previous_cell_on_its_last_line(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await _focus_and_wait(session, "b", "first")
        session.write_input(b"k")

        await _evaluates_to(session, "line('.')", 3)
        assert await _next(events) == {"type": "cell_focus", "id": "a"}
        await _pinned_to(session, "a")


@requires_nvim
async def test_the_cursor_keeps_its_column_when_it_skips_a_marker(neovim_home):
    cells = [{"id": "a", "text": "abcdefgh"}, {"id": "b", "text": "ij\nklmnopqr"}]
    async with _notebook(neovim_home, cells) as (session, events):
        await _type(session, b"5l", b"j")  # the marker in between is shorter than column 6

        assert await _eval(session, "[line('.'), col('.')]") == [4, 2]  # the first line of b is two columns wide
        assert await _next(events) == {"type": "cell_focus", "id": "b"}
        await _type(session, b"j")
        assert await _eval(session, "[line('.'), col('.')]") == [5, 6]  # the wanted column came through


@requires_nvim
async def test_at_the_first_and_last_cell_edges_nothing_happens(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await _type(session, b"k", b"k")  # up from the first line of the first cell: onto its marker, which is none to enter
        assert await _eval(session, "line('.')") == 2
        await _pinned_to(session, "a")

        await _focus_and_wait(session, "c", "last")
        await _type(session, b"j", b"j", DOWN)
        assert await _eval(session, "line('.')") == 9
        await _pinned_to(session, "c")
        await _quiet(events)


@requires_nvim
@pytest.mark.parametrize(
    ("start", "keys", "line", "cell"),
    [
        (("a", "first"), (b"G",), 9, "c"),
        (("c", "first"), (b"g", b"g"), 2, "a"),
        (("a", "first"), (b"/z = 3\r",), 7, "b"),
        (("c", "first"), (b"?import\r",), 2, "a"),
        (("a", "first"), (b":6\r",), 6, "b"),
        (("c", "first"), (b":4\r",), 5, "b"),  # a jump onto a marker lands on the first line of its cell
        (("c", "first"), (b":1\r",), 2, "a"),
        (("a", "first"), (b"3j",), 5, "b"),
        (("a", "first"), (b"2j",), 5, "b"),
        (("a", "first"), (b"5j",), 7, "b"),
        (("c", "first"), (b"4k",), 5, "b"),
        (("b", "first"), (b"}",), 9, "c"),
    ],
    ids=[
        "G",
        "gg",
        "search",
        "search-back",
        "line-jump",
        "jump-onto-marker",
        "jump-onto-first-marker",
        "3j",
        "2j",
        "5j",
        "4k",
        "paragraph",
    ],
)
async def test_a_cursor_that_leaves_the_cell_pins_the_cell_it_lands_in_and_reports_the_focus(neovim_home, start, keys, line, cell):
    async with _notebook(neovim_home) as (session, events):
        await _focus_and_wait(session, *start)
        for key in keys:
            session.write_input(key)

        await _evaluates_to(session, "line('.')", line)
        assert await _next(events) == {"type": "cell_focus", "id": cell}
        await _pinned_to(session, cell)
        await _quiet(events)


@requires_nvim
@pytest.mark.parametrize(
    ("start", "key", "line", "cell"),
    [(("a", "last"), DOWN, 5, "b"), (("b", "first"), UP, 3, "a")],
    ids=["down", "up"],
)
async def test_arrows_in_insert_mode_cross_cells_too(neovim_home, start, key, line, cell):
    async with _notebook(neovim_home) as (session, events):
        await _focus_and_wait(session, *start)
        await _type(session, b"i", key)

        await _evaluates_to(session, "line('.')", line)
        assert await _next(events) == {"type": "cell_focus", "id": cell}
        await _pinned_to(session, cell)
        assert (await _eval(session, "mode()")) == "i"


@requires_nvim
async def test_edits_of_the_cell_left_are_reported_before_the_focus_moves(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await _focus_and_wait(session, "a", "last")
        session.write_input(b"Ax")
        await _type(session, ESC)
        session.write_input(b"j")

        assert await _next(events) == {"type": "cell_edit", "id": "a", "text": "import os\nx = 1x"}
        assert await _next(events) == {"type": "cell_focus", "id": "b"}
        await _quiet(events)


@requires_nvim
@pytest.mark.parametrize(
    "keys",
    [(CTRL_E,), (CTRL_Y,), (b"zt",), (b"j", b"zt"), (b"zb",), (b"zz",), (b"j", b"zb", b"k", b"zt")],
    ids=["ctrl-e", "ctrl-y", "zt", "j-zt", "zb", "zz", "mixed"],
)
async def test_the_window_is_pinned_again_after_it_scrolled(neovim_home, keys):
    async with _notebook(neovim_home, rows=24) as (session, events):
        for key in keys:
            session.write_input(key)
            await asyncio.sleep(0.15)

        await _pinned_to(session, "a")
        await _quiet(events)


@requires_nvim
@pytest.mark.parametrize(
    "keys",
    [(b"\x04",), (b"\x06",), (b"\x02",), (b"L",), (b"H",), (b"M",), (b"10j",), (b"10k",), (b"\x05", b"\x05"), (b"\x19", b"\x19")],
    ids=["ctrl-d", "ctrl-f", "ctrl-b", "L", "H", "M", "10j", "10k", "ctrl-e-twice", "ctrl-y-twice"],
)
async def test_whatever_moves_the_cursor_the_window_ends_up_pinned_to_the_cell_it_is_in(neovim_home, keys):
    async with _notebook(neovim_home) as (session, events):
        await _focus_and_wait(session, "b", "first")
        for key in keys:
            session.write_input(key)
            await asyncio.sleep(0.15)
        await asyncio.sleep(0.3)

        line = await _eval(session, "line('.')")
        assert line not in MARKER_LINES
        cell = next(cell for cell, (first, last) in spans_of(CELLS).items() if first <= line <= last)
        await _pinned_to(session, cell)
        focused = [event for event in await _settled(events) if event["type"] == "cell_focus"]
        assert (focused[-1]["id"] if focused else "b") == cell


@requires_nvim
async def test_a_cell_taller_than_the_window_scrolls_with_its_cursor_but_never_above_its_first_line(neovim_home):
    tall = [{"id": "t", "text": "\n".join(f"l{index}" for index in range(60))}, {"id": "u", "text": "tail"}]
    async with _notebook(neovim_home, tall, rows=24) as (session, events):
        await _evaluates_to(session, "line('w0')", 2)
        session.write_input(b"59j")  # the last line of the cell, 61

        await _evaluates_to(session, "line('.')", 61)
        assert await _eval(session, "line('w0')") >= 2
        assert await _eval(session, "line('w$')") >= 61
        assert await _hidden(session) == [(1, 1), (62, 63)]

        session.write_input(b"gg")  # the marker above: back on the first line, window back on top
        await _evaluates_to(session, "line('w0')", 2)
        assert await _eval(session, "line('.')") == 2
        await _quiet(events)


@requires_nvim
async def test_shift_enter_runs_the_active_cell_after_reporting_what_was_typed(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await _focus_and_wait(session, "b", "first")

        session.write_input(b"Ax" + SHIFT_ENTER)  # insert mode, typed and run at once
        assert await _next(events) == {"type": "cell_edit", "id": "b", "text": "print(os.sep)x\ny = 2\nz = 3"}
        assert await _next(events) == {"type": "cell_run", "id": "b", "text": "print(os.sep)x\ny = 2\nz = 3", "advance": True}

        await _type(session, ESC)
        session.write_input(SHIFT_ENTER)  # normal mode, nothing new to report
        assert await _next(events) == {"type": "cell_run", "id": "b", "text": "print(os.sep)x\ny = 2\nz = 3", "advance": True}

        # After moving on, the new cell is the one that runs
        await session.focus_cell("c", "first")
        session.write_input(SHIFT_ENTER)
        assert await _next(events) == {"type": "cell_run", "id": "c", "text": "w = 4", "advance": True}
        await _quiet(events)


@requires_nvim
async def test_write_requests_a_notebook_save_after_the_last_edit_and_writes_nothing(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        copy = Path(await _eval(session, "expand('%:p')"))
        await _type(session, b"Ab", ESC, b":w\r")

        order = []
        while not order or order[-1]["type"] != "save_notebook":
            order.append(await _next(events))
        assert order == [{"type": "cell_edit", "id": "a", "text": "import osb\nx = 1"}, {"type": "save_notebook"}]
        assert await _eval(session, "&modified") == 0
        assert copy.read_text() == "\n".join(SCRIPT) + "\n"

        await _type(session, b":w\r")  # nothing changed since: the request alone
        assert await _next(events) == {"type": "save_notebook"}
        await _quiet(events)


@requires_nvim
async def test_a_buffer_wiped_in_neovim_is_built_again_by_the_next_set_cells(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await _type(session, b":bwipeout!\r")
        await _evaluates_to(session, "bufname()", "")

        await session.set_cells("again.py", CELLS[:2])

        assert await _lines(session) == script_of(CELLS[:2])
        assert await _eval(session, "&filetype") == "python"
        await _pinned(session, 2, 3)
        session.write_input(b"Ax")
        assert await _next(events) == {"type": "cell_edit", "id": "a", "text": "import osx\nx = 1"}


@requires_nvim
async def test_notebook_sessions_never_report_buffer_switches_as_documents(neovim_home):
    switched = []
    events: asyncio.Queue[dict] = asyncio.Queue()
    session = NeovimSession.spawn("placeholder.txt", "", rows=24, cols=80, cwd=neovim_home, path="placeholder", cell_session=True)
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(_save, lambda text: None, None, switched.append, None, events.put_nowait), 15)
        await session.set_cells("nb.py", CELLS)
        await session.focus_cell("b", "first")
        await session.set_cells("nb.py", CELLS[:2])
        await asyncio.sleep(0.2)
        assert switched == []
    finally:
        await session.close()
        await drain


def _python_language_server() -> str | None:
    """`basedpyright-langserver` on PATH or in the developer's mason install: tests needing a server skip without one."""
    mason = Path.home() / ".local/share/nvim/mason/bin/basedpyright-langserver"
    return shutil.which("basedpyright-langserver") or (str(mason) if mason.exists() else None)


@pytest.mark.skipif(_python_language_server() is None, reason="basedpyright is not installed")
@requires_nvim
async def test_a_language_server_sees_all_cells_as_one_script(neovim_home):
    """A name imported in cell 1 is defined in cell 2, and removing the import makes it undefined: the LSP reads one file."""
    config = f"""
vim.api.nvim_create_autocmd("FileType", {{
  pattern = "python",
  callback = function()
    local cmd = {{ "{_python_language_server()}", "--stdio" }}
    vim.lsp.start({{ name = "basedpyright", cmd = cmd, root_dir = vim.fn.getcwd() }})
  end,
}})
"""
    (neovim_home / "init.lua").write_text(config)
    cells = [
        {"id": "1", "text": "import os\nimport json"},
        {"id": "2", "text": "print(os.getcwd())\nprint(json.dumps({}), missing_name)"},
    ]

    async def undefined() -> list[str]:
        code = "return vim.tbl_map(function(d) return d.message end, vim.diagnostic.get(0))"
        return sorted(message for message in await session._lua(code) if "is not defined" in message)

    async def until(expected: list[str]) -> None:
        for _ in range(300):
            if await undefined() == expected:
                return
            await asyncio.sleep(0.1)
        assert await undefined() == expected

    async with _notebook(neovim_home, cells, filename="analysis.py") as (session, _events):
        await until(['"missing_name" is not defined'])  # `os` and `json` of cell 1 are known in cell 2

        await session.set_cells("analysis.py", [{"id": "1", "text": "import json"}, cells[1]])
        await until(['"missing_name" is not defined', '"os" is not defined'])  # the edit of cell 1 reached the server


@requires_nvim
async def test_a_notebook_path_names_the_buffer_beside_the_notebook_without_creating_anything(neovim_home, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    notebook = project / "analysis.ipynb"
    notebook.write_text("{}")
    async with _notebook(neovim_home, notebook_path=notebook) as (session, events):
        assert await _eval(session, "expand('%:p')") == f"{notebook}.py"
        assert await _eval(session, "&filetype") == "python"
        assert await _lines(session) == SCRIPT
        assert await _eval(session, "&modified") == 0

        await session.focus_cell("a", "last")
        await _type(session, b"A!", ESC)
        await session.set_cells("nb.py", [CELLS[0], {"id": "b", "text": "other"}, CELLS[2]])
        session.write_input(b":w\r")
        while (await _next(events))["type"] != "save_notebook":
            pass
        await asyncio.sleep(0.3)

        assert sorted(path.name for path in project.iterdir()) == ["analysis.ipynb"]


@requires_nvim
async def test_no_kind_of_write_creates_the_virtual_file_or_reads_one_that_is_there(neovim_home, tmp_path):
    """`:1,2w!` (a partial write), `:w!`, `:up` and `:sav!` all stop at the notebook save, and a file of that name stays as it was."""
    project = tmp_path / "project"
    project.mkdir()
    notebook = project / "analysis.ipynb"
    notebook.write_text("{}")
    stale = project / "analysis.ipynb.py"
    stale.write_text("stale = 1\n")
    async with _notebook(neovim_home, notebook_path=notebook) as (session, events):
        assert await _lines(session) == SCRIPT
        await _type(session, b"Ax", ESC)
        for command in (":1,2w!", ":w!", ":up", f":sav! {project / 'copy.py'}"):
            await _type(session, command.encode() + b"\r")
            await asyncio.sleep(0.2)

        saves = [event for event in await _settled(events) if event["type"] == "save_notebook"]
        assert len(saves) >= 2
        assert sorted(path.name for path in project.iterdir()) == ["analysis.ipynb", "analysis.ipynb.py"]
        assert stale.read_text() == "stale = 1\n"


@requires_nvim
async def test_a_renamed_notebook_names_its_buffer_beside_the_new_file_and_writes_stay_intercepted(neovim_home, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    notebook = project / "analysis.ipynb"
    notebook.write_text("{}")
    async with _notebook(neovim_home, notebook_path=notebook) as (session, events):
        await _type(session, b"Ax", ESC)
        renamed = project / "report.ipynb"
        notebook.rename(renamed)

        elsewhere = tmp_path / "moved.ipynb"
        await session.rename(str(tmp_path / "other.ipynb"), str(elsewhere), elsewhere)  # not this notebook
        await session.rename(str(notebook), "report", None)  # no validated file: nothing to name the buffer after
        assert await _eval(session, "expand('%:p')") == f"{notebook}.py"

        modified = await _eval(session, "&modified")
        await session.rename(str(notebook), str(renamed), renamed)
        assert await _eval(session, "[expand('%:p'), &modified, &filetype]") == [f"{renamed}.py", modified, "python"]
        assert await _lines(session) == ["# %%", "import osx", *SCRIPT[2:]]
        await session.focus_cell("b", "first")  # the cells go on
        assert await _eval(session, "line('.')") == FIRST["b"]

        session.write_input(b":w\r")
        while (await _next(events))["type"] != "save_notebook":
            pass
        await asyncio.sleep(0.3)
        assert sorted(path.name for path in project.iterdir()) == ["report.ipynb"]


@requires_nvim
async def test_without_a_notebook_path_the_buffer_is_a_private_copy_in_the_session_directory(neovim_home, tmp_path):
    async with _notebook(neovim_home, filename="nb.py") as (session, _events):
        assert Path(await _eval(session, "expand('%:p')")).is_relative_to(session._workdir)


@requires_nvim
async def test_a_relative_notebook_path_is_ignored(neovim_home):
    async with _notebook(neovim_home, filename="nb.py", notebook_path=Path("rel/analysis.ipynb")) as (session, _events):
        assert Path(await _eval(session, "expand('%:p')")).is_relative_to(session._workdir)


@pytest.mark.skipif(_python_language_server() is None, reason="basedpyright is not installed")
@requires_nvim
async def test_an_import_of_a_package_of_the_notebook_venv_resolves(neovim_home, tmp_path):
    """The buffer is named in the notebook's directory, so the project's pyproject and `.venv` apply."""
    project = tmp_path / "project"
    package = project / ".venv" / "lib" / "python3.12" / "site-packages" / "tinypkg_for_nb"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n")
    (project / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (project / "pyproject.toml").write_text('[tool.basedpyright]\nvenvPath = "."\nvenv = ".venv"\n')
    notebook = project / "analysis.ipynb"
    notebook.write_text('{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}')
    config = f"""
vim.api.nvim_create_autocmd("FileType", {{
  pattern = "python",
  callback = function(args)
    local root = vim.fs.root(args.buf, {{ "pyproject.toml", ".git" }}) or vim.fn.getcwd()
    vim.lsp.start({{ name = "basedpyright", cmd = {{ "{_python_language_server()}", "--stdio" }}, root_dir = root }})
  end,
}})
"""
    (neovim_home / "init.lua").write_text(config)
    cells = [{"id": "1", "text": "import tinypkg_for_nb\nimport not_a_real_package_xyz"}, {"id": "2", "text": "print(tinypkg_for_nb.VALUE)"}]

    async def unresolved() -> list[str]:
        code = "return vim.tbl_map(function(d) return d.message end, vim.diagnostic.get(0))"
        return sorted(message for message in await session._lua(code) if "could not be resolved" in message)

    async with _notebook(neovim_home, cells, notebook_path=notebook) as (session, _events):
        expected = ['Import "not_a_real_package_xyz" could not be resolved']
        for _ in range(300):
            if await unresolved() == expected:
                break
            await asyncio.sleep(0.1)
        assert await unresolved() == expected


class _NotebookBrowser(_Browser):
    """A browser of a notebook session: the renders of the cells (`cell_render`) and the active cell's screen rows (`cell_height`) are
    kept apart in `renders` and `heights`, they come on their own time."""

    def __init__(self) -> None:
        super().__init__()
        self.renders: asyncio.Queue[dict] = asyncio.Queue()
        self.heights: asyncio.Queue[dict] = asyncio.Queue()

    async def send_json(self, frame: dict) -> None:
        aside = {"cell_render": self.renders, "cell_height": self.heights}.get(frame["type"])
        await (self._frames if aside is None else aside).put(frame)


@contextlib.asynccontextmanager
async def _notebook_browser(monkeypatch, tmp_path, *, notebook: Path | None = None, root: Path | None = None):
    """The session endpoint serving a browser that opened a notebook, with Neovim connected once the body starts."""
    monkeypatch.setattr(websocket_origin, "CORS_ORIGINS", ["*"])
    browser = _NotebookBrowser()
    path = notebook or tmp_path / "nb.ipynb"
    serving = asyncio.create_task(neovim_route.neovim_session(browser, request_identity(), mounted_roots(root or tmp_path)))
    browser.send({"type": "open", "cell_mode": True, "path": str(path), "rows": 24, "cols": 80})
    try:
        browser.send({"type": "message", "text": "Connected"})
        await browser.shows("Connected")
        yield browser
    finally:
        browser.leave()
        await asyncio.wait_for(serving, 10)


def test_open_message_needs_a_document_unless_it_is_a_notebook_session():
    size = {"rows": 24, "cols": 80}
    assert neovim_route.OpenMessage.model_validate({"cell_mode": True, **size}).cell_mode
    assert neovim_route.OpenMessage.model_validate({"path": "p", "filename": "f", "content": "", **size}).cell_mode is False
    for incomplete in ({"path": "p", "filename": "f"}, {"path": "p", "content": ""}, {"filename": "f", "content": ""}, {}):
        with pytest.raises(ValueError):
            neovim_route.OpenMessage.model_validate({**incomplete, **size})


@requires_nvim
async def test_every_notebook_frame_over_the_socket(neovim_home, tmp_path, monkeypatch):
    async with _notebook_browser(monkeypatch, tmp_path) as browser:
        browser.send({"type": "set_cells", "filename": "nb.py", "cells": CELLS})
        browser.send({"type": "focus_cell", "id": "b", "cursor": "last"})
        browser.send({"type": "message", "text": "Settled"})
        await browser.shows("Settled")
        await browser.silent()  # no echo for what the browser sent
        render = await asyncio.wait_for(browser.renders.get(), 10)  # every cell is drawn as Neovim draws it
        assert render["id"] in ("a", "b", "c") and render["lines"]
        heights = [await asyncio.wait_for(browser.heights.get(), 10) for _ in range(2)]  # the screen rows of the pinned cell
        assert heights == [{"type": "cell_height", "id": "a", "rows": 2}, {"type": "cell_height", "id": "b", "rows": 3}]

        browser.type(b"Ax")  # on the last line of cell b
        assert await browser.next_frame("cell_edit") == {"type": "cell_edit", "id": "b", "text": "print(os.sep)\ny = 2\nz = 3x"}

        browser.type(ESC)
        await asyncio.sleep(0.1)
        browser.type(SHIFT_ENTER)
        run = {"type": "cell_run", "id": "b", "text": "print(os.sep)\ny = 2\nz = 3x", "advance": True}
        assert await browser.next_frame("cell_run") == run

        browser.type(DOWN)  # past the last line of b: into c
        assert await browser.next_frame("cell_focus") == {"type": "cell_focus", "id": "c"}

        browser.send(
            {"type": "set_cells", "filename": "nb.py", "cells": [*CELLS[:2], {"id": "c", "text": "QRSTU"}, {"id": "d", "text": "new"}]}
        )
        browser.send({"type": "focus_cell", "id": "d", "cursor": "first"})
        await browser.shows("new")
        browser.send({"type": "focus_cell", "id": "b", "cursor": {"line": 1}})  # a click on the second line of b's snapshot
        browser.send({"type": "message", "text": "On b"})
        await browser.shows("On b")
        browser.type(b"iY\x1b")
        assert await browser.next_frame("cell_edit") == {"type": "cell_edit", "id": "b", "text": "print(os.sep)\nYy = 2\nz = 3"}
        browser.type(b":w\r")
        assert await browser.next_frame("save_notebook") == {"type": "save_notebook"}


@requires_nvim
async def test_malformed_notebook_frames_are_ignored_and_the_session_lives_on(neovim_home, tmp_path, monkeypatch):
    async with _notebook_browser(monkeypatch, tmp_path) as browser:
        browser.send({"type": "set_cells", "filename": "nb.py", "cells": CELLS})
        for bad in (
            {"type": "set_cells", "filename": "nb.py", "cells": "nope"},
            {"type": "set_cells", "filename": "nb.py", "cells": [{"id": "r", "text": "x", "kind": "raw"}]},
            {"type": "set_cells", "filename": "nb.py", "cells": [{"id": "r", "text": "x", "kind": 3}]},
            {"type": "set_cells", "filename": "nb.py", "cells": [{"id": "a"}]},
            {"type": "set_cells", "filename": "nb.py", "cells": [{"id": 1, "text": "x"}]},
            {"type": "set_cells", "filename": "nb.py", "cells": [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]},
            {"type": "set_cells", "filename": "nb.py", "cells": ["a"]},
            {"type": "set_cells", "cells": CELLS},
            {"type": "focus_cell", "id": "b", "cursor": "middle"},
            {"type": "focus_cell", "id": 5, "cursor": "first"},
            {"type": "focus_cell", "id": "ghost", "cursor": "first"},
            {"type": "unknown_frame"},
        ):
            browser.send(bad)
        browser.send({"type": "message", "text": "Still alive"})
        await browser.shows("Still alive")
        await browser.silent()


@requires_nvim
async def test_frames_keep_their_order_through_one_queue(neovim_home, tmp_path, monkeypatch):
    async with _notebook_browser(monkeypatch, tmp_path) as browser:
        browser.send({"type": "set_cells", "filename": "nb.py", "cells": [CELLS[0]]})
        browser.send({"type": "set_cells", "filename": "nb.py", "cells": [CELLS[0], CELLS[1]]})
        browser.send({"type": "set_cells", "filename": "nb.py", "cells": CELLS})
        browser.send({"type": "focus_cell", "id": "a", "cursor": "last"})
        browser.send({"type": "message", "text": "Settled"})
        await browser.shows("Settled")
        await browser.silent()

        # Edits and the run of a quick Shift+Enter arrive in the order they happened, round after round
        digits = ""
        browser.type(ESC)
        assert await browser.frames(1) == [{"type": "leave"}]  # Esc in normal mode gives focus back to the app
        for round_ in range(50):
            browser.type(b"A" + str(round_ % 10).encode() + SHIFT_ENTER)
            digits += str(round_ % 10)
            text = f"import os\nx = 1{digits}"
            assert await browser.frames(2) == [
                {"type": "cell_edit", "id": "a", "text": text},
                {"type": "cell_run", "id": "a", "text": text, "advance": True},
            ]
            browser.type(ESC)
            await asyncio.sleep(0.1)
        await browser.silent()


@requires_nvim
async def test_a_frame_with_a_repeated_cell_id_is_dropped_with_a_warning(neovim_home, tmp_path, monkeypatch, caplog):
    async with _notebook_browser(monkeypatch, tmp_path) as browser:
        browser.send({"type": "set_cells", "filename": "nb.py", "cells": CELLS})
        browser.send({"type": "set_cells", "filename": "nb.py", "cells": [CELLS[0], {"id": "a", "text": "again"}]})
        browser.send({"type": "focus_cell", "id": "b", "cursor": "last"})
        browser.send({"type": "message", "text": "Still alive"})
        await browser.shows("Still alive")

        assert any("repeats a cell id" in record.getMessage() for record in caplog.records)
        browser.type(b"Ax")  # the cells of the first frame are still there: b is the active one
        assert await browser.next_frame("cell_edit") == {"type": "cell_edit", "id": "b", "text": "print(os.sep)\ny = 2\nz = 3x"}


@requires_nvim
@pytest.mark.parametrize(
    ("filename", "suffix", "filetype"),
    [
        ("notebook.py", ".py", "python"),
        ("notebook.jl", ".jl", "julia"),
        ("notebook.r", ".r", "r"),
        ("notebook", ".py", "python"),
        ("n.we ird", ".py", "python"),
    ],
)
async def test_the_virtual_name_takes_the_extension_of_the_notebook_language(neovim_home, tmp_path, filename, suffix, filetype):
    project = tmp_path / "project"
    project.mkdir()
    notebook = project / "analysis.ipynb"
    notebook.write_text("{}")
    async with _notebook(neovim_home, filename=filename, notebook_path=notebook) as (session, _events):
        assert await _eval(session, "expand('%:p')") == f"{notebook}{suffix}"
        assert await _eval(session, "&filetype") == filetype
        assert sorted(path.name for path in project.iterdir()) == ["analysis.ipynb"]


@requires_nvim
@pytest.mark.parametrize(
    ("notebook", "named"),
    [
        ("project/nb.ipynb", True),
        ("project/../project/nb.ipynb", True),
        ("project/missing.ipynb", False),
        ("project/sub/../../outside/nb.ipynb", False),
    ],
)
async def test_only_a_notebook_file_under_the_file_roots_names_the_buffer(neovim_home, tmp_path, monkeypatch, notebook, named):
    (tmp_path / "project").mkdir()
    (tmp_path / "outside").mkdir()
    (tmp_path / "project" / "nb.ipynb").write_text("{}")
    (tmp_path / "outside" / "nb.ipynb").write_text("{}")
    path = Path(str(tmp_path / notebook))  # keeps the `..` segments
    async with _notebook_browser(monkeypatch, tmp_path, notebook=path, root=tmp_path / "project") as browser:
        browser.send({"type": "set_cells", "filename": "nb.py", "cells": CELLS})
        await browser.shows("nb.ipynb.py" if named else "nb.py")  # the status line names the buffer


@requires_nvim
async def test_u_after_a_rebuild_never_runs_into_the_marker_protection(neovim_home):
    """Shift+Enter on the last cell appends a cell: typing there is undone, older typing is out of reach and `u` is harmless."""
    async with _notebook(neovim_home) as (session, events):
        await session.focus_cell("c", "last")
        await _type(session, b"Ax", ESC)
        await _evaluates_to(session, "getline(9)", "w = 4x")
        await _settled(events)
        cells = [CELLS[0], CELLS[1], {"id": "c", "text": "w = 4x"}, {"id": "d", "text": "new"}]
        await session.set_cells("nb.py", cells)
        await session.focus_cell("d", "last")
        await _type(session, b"Ay", ESC)
        await _evaluates_to(session, "getline(11)", "newy")

        await _type(session, b"u")
        await _evaluates_to(session, "getline(11)", "new")
        await _type(session, b"u", b"u", b"u")

        assert await _lines(session) == script_of(cells)
        assert await _marks(session) == [1, 4, 8, 10]
        assert MARKER_MESSAGE not in await _messages(session)

        await _type(session, b"Az", ESC)  # still an editable notebook
        await _evaluates_to(session, "getline(11)", "newz")
        await _type(session, b"u")
        await _evaluates_to(session, "getline(11)", "new")


@requires_nvim
async def test_u_after_a_rebuild_that_moved_a_cell_keeps_the_markers(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await _type(session, b"Ax", ESC)
        await _settled(events)
        moved = [{"id": "a", "text": "import osx\nx = 1"}, CELLS[2], CELLS[1]]
        await session.set_cells("nb.py", moved)
        await _type(session, b"u", b"u")

        assert await _lines(session) == script_of(moved)
        assert await _marks(session) == [1, 4, 6]
        assert MARKER_MESSAGE not in await _messages(session)


@requires_nvim
async def test_a_new_text_of_a_cell_from_the_notebook_is_one_undo_step_that_leaves_the_markers_alone(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await session.set_cells("nb.py", [{"id": "a", "text": "changed"}, *CELLS[1:]])
        assert await _lines(session) == script_of([{"id": "a", "text": "changed"}, *CELLS[1:]])
        await _settled(events)

        await _type(session, b"u")
        assert await _lines(session) == SCRIPT
        assert [event["text"] for event in await _settled(events) if event["type"] == "cell_edit"] == [CELLS[0]["text"]]
        assert await _marks(session) == MARKER_LINES
        assert MARKER_MESSAGE not in await _messages(session)


@requires_nvim
async def test_a_message_longer_than_the_screen_does_not_stop_neovim_answering(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await session._lua("vim.api.nvim_echo({ { string.rep('a long message\\n', 60) } }, true, {})")

        await asyncio.wait_for(session.set_cells("nb.py", CELLS[:2]), 5)
        await asyncio.wait_for(session.focus_cell("b", "first"), 5)
        assert await _eval(session, "mode()") == "n"
        assert await _lines(session) == script_of(CELLS[:2])


@requires_nvim
@pytest.mark.parametrize("through_lsp", [True, False], ids=["lsp-edit", "put-above"])
async def test_lines_put_above_the_first_marker_join_the_first_cell(neovim_home, through_lsp):
    async with _notebook(neovim_home) as (session, events):
        if through_lsp:  # what an LSP quick fix does: an edit at the very start of the script
            await session._lua("vim.api.nvim_buf_set_text(0, 0, 0, 0, 0, { 'import json', '' })")
        else:
            await _type(session, b":0put =['import json']\r")

        await _evaluates_to(session, "getline(1, '$')", ["# %%", "import json", *SCRIPT[1:]])
        assert await _marks(session) == [1, 5, 9]
        assert [event for event in await _settled(events) if event["type"] == "cell_edit"] == [
            {"type": "cell_edit", "id": "a", "text": "import json\nimport os\nx = 1"}
        ]
        assert MARKER_MESSAGE not in await _messages(session)

        await _type(session, b"u")  # one step: the head is gone and the markers are where they were
        assert await _lines(session) == SCRIPT
        assert await _marks(session) == MARKER_LINES
        assert MARKER_MESSAGE not in await _messages(session)


# --- Every cell drawn as Neovim draws it (`cell_render`) ---

HEX = re.compile(r"#[0-9a-f]{6}")
STYLE_KEYS = {"fg", "bg", "sp", "bold", "italic", "strike", "underline"}


def _texts(render: dict) -> list[str]:
    """The lines of a render as plain text, gutter included."""
    return ["".join(text for text, _style in line) for line in render["lines"]]


def _well_formed(render: dict) -> None:
    """A render holds data only: strings, style indexes in range, styles of known keys with `#rrggbb` colors."""
    assert render["styles"][0] == {}
    for style in render["styles"]:
        assert set(style) <= STYLE_KEYS
        assert all(HEX.fullmatch(style[key]) for key in ("fg", "bg", "sp") if key in style)
        assert all(style[key] is True for key in ("bold", "italic", "strike") if key in style)
        assert style.get("underline", "line") in ("line", "curl")
    for line in render["lines"]:
        for text, style in line:
            assert isinstance(text, str) and text and 0 <= style < len(render["styles"])
    assert set(render["base"]) <= {"fg", "bg"}
    assert isinstance(render["height"], int) and render["height"] >= 1


async def _renders(events: _Events, seconds: float = 0.8) -> dict[str, dict]:
    """The renders that came until none came for *seconds*, the last one of every cell."""
    seen: dict[str, dict] = {}
    while True:
        try:
            render = await asyncio.wait_for(events.renders.get(), seconds)
        except TimeoutError:
            return seen
        _well_formed(render)
        seen[render["id"]] = render


@requires_nvim
async def test_every_cell_is_rendered_with_its_exact_text(neovim_home):
    async with _notebook(neovim_home) as (session, events):  # cell a is active: it gets its first render too
        renders = await _renders(events)
        assert set(renders) == {"a", "b", "c"}
        assert _texts(renders["a"]) == ["import os", "x = 1"]
        assert _texts(renders["b"]) == ["print(os.sep)", "y = 2", "z = 3"]
        assert _texts(renders["c"]) == ["w = 4"]
        assert HEX.fullmatch(renders["b"]["base"]["fg"])


@requires_nvim
async def test_markup_in_a_cell_stays_text(neovim_home):
    hostile = '<img src=x onerror="alert(1)"> &amp; </span><span class="Normal">&lt;'
    cells = [CELLS[0], {"id": "h", "text": f"s = '{hostile}'\n# </pre><script>alert(2)</script>"}]
    async with _notebook(neovim_home, cells) as (session, events):
        renders = await _renders(events)
        assert _texts(renders["h"]) == cells[1]["text"].split("\n")


@requires_nvim
async def test_only_the_cells_that_changed_are_rendered_again(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        assert set(await _renders(events)) == {"a", "b", "c"}

        await session.set_cells("nb.py", [*CELLS[:2], {"id": "c", "text": "w = 5\nv = 6"}])
        renders = await _renders(events)
        assert set(renders) == {"c"}
        assert _texts(renders["c"]) == ["w = 5", "v = 6"]

        await session.set_cells("nb.py", [*CELLS[:2], {"id": "c", "text": "w = 5\nv = 6"}])
        assert await _renders(events) == {}


@requires_nvim
async def test_the_cell_being_typed_in_is_rendered_when_it_is_left_before_its_neighbor_takes_the_focus(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await _renders(events)
        await _type(session, b"Ax", ESC)  # cell a: "import osx"
        assert await _renders(events) == {}  # the active cell is Neovim's own window, it is not rendered

        await _type(session, b"jj")  # past the last line of a: into b
        assert await _next(events) == {"type": "cell_edit", "id": "a", "text": "import osx\nx = 1"}
        assert await _next(events) == {"type": "cell_focus", "id": "b"}
        kinds = [(event["type"], event["id"]) for event in events.log if event["type"] in ("cell_render", "cell_focus")]
        assert kinds[-2:] == [("cell_render", "a"), ("cell_focus", "b")]
        left = [event for event in events.log if event["type"] == "cell_render" and event["id"] == "a"][-1]
        assert _texts(left)[0] == "import osx"


@requires_nvim
async def test_focus_cell_puts_the_cursor_on_the_clicked_line_and_the_left_cell_keeps_its_current_render(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await _renders(events)
        await session.focus_cell("b", 1)  # a click on the second line of b's render
        assert await _eval(session, "line('.')") == FIRST["b"] + 1
        assert await _renders(events) == {}  # a did not change since its first render
        await session.focus_cell("c", 7)  # past the end: the last line
        assert await _eval(session, "line('.')") == LAST["c"]


@requires_nvim
async def test_diagnostics_signs_and_the_number_column_are_rendered(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await _renders(events)
        await session._lua("vim.wo.number = true")
        renders = await _renders(events)
        assert set(renders) == {"b", "c"}  # the number column changes every cell
        assert _texts(renders["b"]) == ["  5 print(os.sep)", "  6 y = 2", "  7 z = 3"]

        await session._lua(
            "vim.diagnostic.config({ virtual_text = true, signs = true })"
            " vim.diagnostic.set(vim.api.nvim_create_namespace('test'), 0,"
            " { { lnum = 4, col = 0, end_col = 5, message = 'Undefined name', severity = vim.diagnostic.severity.ERROR } })"
        )
        renders = await _renders(events)
        assert set(renders) == {"b", "c"}  # the cell with the diagnostic, and the sign column it opens in every cell
        first, *rest = _texts(renders["b"])
        assert re.fullmatch(r"\S+\s+5 print\(os\.sep\)\s+\S+ Undefined name", first)
        assert rest == ["    6 y = 2", "    7 z = 3"]
        assert _texts(renders["c"]) == ["    9 w = 4"]
        message = next(run for run in renders["b"]["lines"][0] if "Undefined name" in run[0])
        assert "fg" in renders["b"]["styles"][message[1]]


@requires_nvim
async def test_a_new_colorscheme_renders_every_cell_again(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        before = await _renders(events)
        await session._lua("vim.cmd.colorscheme('vim')")
        after = await _renders(events)
        assert set(after) == {"b", "c"}
        assert after["b"]["base"] != before["b"]["base"]


@requires_nvim
async def test_with_absolute_numbers_a_line_added_above_renders_the_cells_below_again(neovim_home):
    grown = [{"id": "a", "text": "import os\nx = 1\nv = 0"}, *CELLS[1:]]
    async with _notebook(neovim_home) as (session, events):
        await _renders(events)
        await session._lua("vim.wo.number = true")
        await _renders(events)

        await session.set_cells("nb.py", grown)
        renders = await _renders(events)
        assert set(renders) == {"b", "c"}  # a is the active cell, Neovim's own window shows it
        assert _texts(renders["b"]) == ["  6 print(os.sep)", "  7 y = 2", "  8 z = 3"]
        assert _texts(renders["c"]) == [" 10 w = 4"]

        # Relative numbers read the same wherever the cell starts
        await session._lua("vim.wo.number = false vim.wo.relativenumber = true")
        await _renders(events)
        await session.set_cells("nb.py", CELLS)
        assert await _renders(events) == {}


# A language server's semantic tokens as Neovim keeps them, faked: `version` is the text version the result belongs to
FAKE_TOKENS = """
local buffer = vim.api.nvim_get_current_buf()
vim.api.nvim_set_hl(0, "@lsp.type.function.python", { fg = "#123456" })
vim.lsp.util.buf_versions[buffer] = ...
vim.lsp.semantic_tokens.__STHighlighter.active[buffer] = {
  on_win = function() end,
  client_state = { { current_result = { version = select(2, ...), highlights = {
    { line = 4, start_col = 0, end_col = 5, type = "function", modifiers = {} },
  } } } },
}
local request = { type = "complete", method = "textDocument/semanticTokens/full" }
vim.api.nvim_exec_autocmds("LspRequest", { buffer = buffer, data = { request = request } })
"""


def _token_colored(render: dict) -> bool:
    """Whether `print` of cell b is drawn in the color of its (fake) semantic token."""
    return any(text == "print" and render["styles"][style].get("fg") == "#123456" for text, style in render["lines"][0])


@requires_nvim
async def test_a_render_made_before_the_tokens_of_its_text_arrived_is_made_again_when_they_do(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await _renders(events)
        await session._lua(FAKE_TOKENS, 1, 1)
        assert _token_colored((await _renders(events))["b"])

        # An edit that moves no token: the result for the new text is still out when b is rendered
        await session._lua(FAKE_TOKENS, 2, 1)
        await session.set_cells("nb.py", [CELLS[0], {"id": "b", "text": "print(os.sep)\ny = 3\nz = 3"}, CELLS[2]])
        renders = await _renders(events)
        assert set(renders) == {"b"}
        assert not _token_colored(renders["b"])

        # The same tokens arrive for it: the signature is what it was, the render gets them all the same
        await session._lua(FAKE_TOKENS, 2, 2)
        renders = await _renders(events)
        assert set(renders) == {"b"}
        assert _token_colored(renders["b"])
        await session._lua(FAKE_TOKENS, 2, 2)
        assert await _renders(events) == {}  # and once only


async def _render_frames(events: _Events, seconds: float = 0.8) -> list[dict]:
    """Every `cell_render` frame (renders and the ones that take a render back) until none came for *seconds*."""
    frames = []
    while True:
        try:
            frames.append(await asyncio.wait_for(events.renders.get(), seconds))
        except TimeoutError:
            return frames


@requires_nvim
async def test_a_notebook_grown_past_the_render_limit_takes_every_render_back_and_renders_again_below_it(neovim_home):
    long = {"id": "c", "text": "\n".join(f"w{index} = {index}" for index in range(4100))}
    async with _notebook(neovim_home) as (session, events):
        assert set(await _renders(events)) == {"a", "b", "c"}

        await session.set_cells("nb.py", [*CELLS[:2], long])
        assert await _render_frames(events) == [{"type": "cell_render", "id": cell} for cell in ("a", "b", "c")]
        await session.set_cells("nb.py", [*CELLS[:2], {**long, "text": long["text"] + "\nmore = 1"}])
        assert await _render_frames(events) == []  # taken back once, nothing renders past the limit

        await session.set_cells("nb.py", CELLS)
        renders = await _renders(events)
        assert set(renders) == {"a", "b", "c"}
        assert _texts(renders["c"]) == ["w = 4"]


@requires_nvim
async def test_a_failed_render_takes_the_cell_render_back_until_the_cell_renders_again(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await _renders(events)
        await session._lua("_G.real_open_win = vim.api.nvim_open_win vim.api.nvim_open_win = function() error('x') end")
        await session.set_cells("nb.py", [CELLS[0], {"id": "b", "text": "changed"}, CELLS[2]])
        assert await _render_frames(events) == [{"type": "cell_render", "id": "b"}]
        assert ":1: x" in (await session._lua("return _G.gigachad_render_state()"))["failure"]  # the test's error, kept

        await session._lua("vim.api.nvim_open_win = _G.real_open_win")
        await session.set_cells("nb.py", [CELLS[0], {"id": "b", "text": "changed again"}, CELLS[2]])
        renders = await _renders(events)
        assert set(renders) == {"b"}
        assert _texts(renders["b"]) == ["changed again"]


# What a notifier such as noice.nvim watches: whether `vim.notify` is still its own. The 'statuscolumn' runs inside every render (on
# its float) and notes what it sees there, an extmark `tohtml` does not support makes `tohtml` warn on every render of cell b
WATCH_RENDERS = """
_G.notices, _G.during, _G.window_events = {}, {}, {}
_G.own_notify = function(message) table.insert(_G.notices, message) end
vim.notify = _G.own_notify
function _G.watch_render()
  table.insert(_G.during, { vim.notify == _G.own_notify, vim.o.eventignore, vim.fn.win_gettype() })
  return ""
end
vim.o.eventignore = "FocusGained"
vim.api.nvim_create_autocmd({ "WinClosed", "OptionSet" }, {
  callback = function(event) table.insert(_G.window_events, event.event .. " " .. event.match) end,
})
local namespace = vim.api.nvim_create_namespace("test_right_align")
vim.api.nvim_buf_set_extmark(0, namespace, 4, 0, { virt_text = { { "!", "Comment" } }, virt_text_pos = "right_align" })
vim.wo.statuscolumn = "%{v:lua.watch_render()}%l "
"""


@requires_nvim
async def test_renders_leave_vim_notify_and_eventignore_alone_and_say_nothing(neovim_home):
    async with _notebook(neovim_home) as (session, events):
        await _renders(events)
        await session._lua(WATCH_RENDERS)
        assert "b" in await _renders(events)  # the new 'statuscolumn' renders the cells again
        for colorscheme in ("vim", "default"):
            await session._lua(f"vim.cmd.colorscheme('{colorscheme}')")
            assert set(await _renders(events)) == {"b", "c"}
        await session.set_cells("nb.py", [CELLS[0], {"id": "b", "text": "print(os.sep)\ny = 3"}, CELLS[2]])
        assert "b" in await _renders(events)
        await asyncio.sleep(0.3)  # past the warnings `tohtml` schedules

        state = await session._lua(
            "return { vim.notify == _G.own_notify, _G.notices, _G.during, _G.window_events, vim.o.eventignore }"
        )
        same, notices, during, window_events, ignored = state
        assert same and notices == [] and ignored == "FocusGained"
        assert any(kind == "popup" for _same, _ignored, kind in during), "the statuscolumn never ran in a render"
        assert all(same and ignored == "FocusGained" for same, ignored, _kind in during)
        assert not [event for event in window_events if event.startswith("WinClosed") or event == "OptionSet signcolumn"]

@requires_nvim
async def test_rendering_gives_way_to_typing_and_goes_on_after_it(neovim_home):
    cells = [{"id": f"k{index}", "text": f"v{index} = {index}"} for index in range(60)]
    async with _notebook(neovim_home, cells) as (session, events):
        assert len(await _renders(events)) == 60
        await session._lua("vim.cmd.colorscheme('vim')")  # every cell but the active one renders again
        first = await asyncio.wait_for(events.renders.get(), 10)

        session.write_input(b"A")
        for _ in range(15):  # faster than the 150 ms pause a render waits for
            session.write_input(b"x")
            await asyncio.sleep(0.05)
        during = events.renders.qsize()
        session.write_input(ESC)

        assert during <= 10, f"{during} cells rendered while typing"
        rest = await _renders(events)
        assert {first["id"], *rest} == {cell["id"] for cell in cells[1:]}



def test_parse_tohtml_keeps_text_and_valid_styles_only():
    html = (
        "<style>\n.Keyword {color: #c61365; font-style: italic}\n.Bad {color: red; background-color: url(x)}\n"
        ".Warn {text-decoration-color: #bb4f04; text-decoration-style: wavy; text-decoration-line: underline}\n</style>\n"
        '<pre>\n<span class="Keyword">import</span> <span class="Bad">x</span>\n'
        '<span class="Warn">&lt;b&gt; &amp;</span> <img src=x onerror=alert(1)></span>\n\n</pre>'
    )
    gutter = [[["1 ", {"fg": "#a1968a"}]], [["2 ", {"fg": "red", "bold": "yes", "underline": "wavy", "onclick": "x"}]]]
    rendering = parse_tohtml(html, gutter, {"fg": "#ded7d0", "bg": "javascript:x", "bold": True})
    assert _texts(rendering) == ["1 import x", "2 <b> & <img src=x onerror=alert(1)>"]
    styles = rendering["styles"]
    assert [styles[style] for _text, style in rendering["lines"][0]] == [{"fg": "#a1968a"}, {"fg": "#c61365", "italic": True}, {}]
    assert [styles[style] for _text, style in rendering["lines"][1]] == [{}, {"sp": "#bb4f04", "underline": "curl"}, {}]
    assert rendering["base"] == {"fg": "#ded7d0"}


def test_parse_tohtml_starts_the_range_without_what_the_rows_above_left_open():
    html = (
        "<style>\n.Warn {text-decoration-line: underline}\n.Keyword {color: #c61365}\n</style>\n"
        '<pre><span class="Warn"></span></span><span class="Warn">\n<span class="Keyword">x</span> = 1\n\n</pre>'
    )
    rendering = parse_tohtml(html)
    assert [(text, rendering["styles"][style]) for text, style in rendering["lines"][0]] == [("x", {"fg": "#c61365"}), (" = 1", {})]


def test_parse_tohtml_takes_the_screen_rows_and_the_wrap_column_only_when_they_are_valid():
    html = "<style>\n</style>\n<pre>\nx = 1\ny = 2\n\n</pre>"
    assert parse_tohtml(html, None, None, None, 5, 4) | {"styles": None} == {
        "styles": None, "lines": [[["x = 1", 0]], [["y = 2", 0]]], "base": {}, "rows": [0, 1], "height": 5, "wrap": 4
    }
    for height, wrap in ((0, -1), (True, False), ("3", "4"), (None, None)):
        rendering = parse_tohtml(html, None, None, None, height, wrap)
        assert rendering["height"] == 2 and "wrap" not in rendering  # the lines' count, and a window that does not wrap


@requires_nvim
async def test_a_render_takes_the_screen_rows_of_its_cell_and_the_wrap_column(neovim_home):
    long = " ".join(["word"] * 40)  # 199 columns: three screen lines in 80
    cells = [CELLS[0], {"id": "w", "text": f"{long}\nx = 1"}]
    async with _notebook(neovim_home, cells) as (session, events):
        render = (await _renders(events))["w"]
        assert (_texts(render), render["height"], render["wrap"]) == ([long, "x = 1"], 4, 0)

        await session._lua("vim.wo.number = true")  # wrapped lines go on after the number column
        render = (await _renders(events))["w"]
        assert (render["height"], render["wrap"]) == (4, 4)

        await session._lua("vim.wo.wrap = false")
        render = (await _renders(events))["w"]
        assert render["height"] == 2 and "wrap" not in render


# --- Markdown cells in a second buffer, and leaving a buffer at a cell of the other kind ---

CTRL_ENTER = b"\x1b[13;5u"  # what the browser sends for Ctrl+Enter, Neovim reads it as <C-CR>
MIXED = [
    {"id": "m0", "kind": "markdown", "text": "# Title\nIntro"},
    {"id": "a", "kind": "code", "text": "import os\nx = 1"},
    {"id": "b", "text": "print(os.sep)"},  # no kind: a code cell
    {"id": "m1", "kind": "markdown", "text": "Some *text*"},
    {"id": "m2", "kind": "markdown", "text": "More\ntext"},
    {"id": "c", "kind": "code", "text": "w = 4"},
]
CODE_SCRIPT = ["# %%", "import os", "x = 1", "# %%", "print(os.sep)", "# %%", "w = 4"]
MARKDOWN_SCRIPT = ["<!-- %% -->", "# Title", "Intro", "<!-- %% -->", "Some *text*", "<!-- %% -->", "More", "text"]
# First and last line of every cell of MIXED in its own buffer
MIXED_SPANS = {"a": (2, 3), "b": (5, 5), "c": (7, 7), "m0": (2, 3), "m1": (5, 5), "m2": (7, 8)}


async def _markdown_buffer(session: NeovimSession) -> dict | None:
    """Name, lines, filetype, listed and modified flags of the markdown buffer, None when there is none."""
    code = "for _, b in ipairs(vim.api.nvim_list_bufs()) do local name = vim.api.nvim_buf_get_name(b) "
    code += "if name:match('%.md$') then return { name = name, lines = vim.api.nvim_buf_get_lines(b, 0, -1, false), "
    code += "filetype = vim.bo[b].filetype, listed = vim.bo[b].buflisted, modified = vim.bo[b].modified } end end return vim.NIL"
    return await session._lua(code)


async def _focus_mixed(session: NeovimSession, cell: str, cursor: str) -> None:
    await session.focus_cell(cell, cursor)
    await _pinned(session, *MIXED_SPANS[cell])


@requires_nvim
async def test_markdown_cells_are_a_second_buffer_with_their_own_markers(neovim_home):
    async with _notebook(neovim_home, CELLS) as (session, events):
        assert await _markdown_buffer(session) is None  # no markdown cells, no buffer for them

    async with _notebook(neovim_home, MIXED) as (session, events):
        assert await _lines(session) == CODE_SCRIPT  # the window shows the code cells, the first one active
        await _pinned(session, 2, 3)
        markdown = await _markdown_buffer(session)
        assert markdown["lines"] == MARKDOWN_SCRIPT
        assert [markdown["filetype"], markdown["listed"], markdown["modified"]] == ["markdown", False, False]
        assert Path(markdown["name"]).name == "nb.md" and Path(markdown["name"]).is_relative_to(session._workdir)
        await _quiet(events)


@requires_nvim
async def test_focus_cell_on_a_markdown_cell_puts_the_markdown_buffer_in_the_window_pinned_to_it(neovim_home):
    async with _notebook(neovim_home, MIXED) as (session, events):
        await session.focus_cell("m2", "last")
        assert await _lines(session) == MARKDOWN_SCRIPT
        assert await _eval(session, "&filetype") == "markdown"
        await _pinned(session, 7, 8)
        assert await _eval(session, "line('.')") == 8

        await _focus_mixed(session, "b", "first")  # back to the code
        assert await _lines(session) == CODE_SCRIPT
        await _focus_mixed(session, "m0", "keep")
        assert await _eval(session, "line('.')") == 2
        await _quiet(events)


@requires_nvim
async def test_markdown_cells_report_edits_cross_into_an_adjacent_markdown_cell_and_keep_their_markers(neovim_home):
    async with _notebook(neovim_home, MIXED) as (session, events):
        await _focus_mixed(session, "m1", "first")
        await _type(session, b"A!", ESC)  # an Escape that ends an insert leaves nothing
        assert await _next(events) == {"type": "cell_edit", "id": "m1", "text": "Some *text*!"}

        session.write_input(b"j")  # m2 comes right after m1 in the notebook
        assert await _next(events) == {"type": "cell_focus", "id": "m2"}
        await _pinned(session, 7, 8)
        session.write_input(b"k")
        assert await _next(events) == {"type": "cell_focus", "id": "m1"}
        await _pinned(session, 5, 5)

        await _type(session, b":4d\r")  # the marker line of m1
        await _reverted(session)
        assert await _lines(session) == [*MARKDOWN_SCRIPT[:4], "Some *text*!", *MARKDOWN_SCRIPT[5:]]
        assert [event for event in await _settled(events) if event["type"] == "cell_edit"] == []


EXITS = {
    "k-above-code": ("a", "first", b"k", {"type": "cell_exit", "id": "a", "direction": "up"}),
    "up-above-code": ("a", "first", UP, {"type": "cell_exit", "id": "a", "direction": "up"}),
    "j-below-code": ("b", "last", b"j", {"type": "cell_exit", "id": "b", "direction": "down"}),
    "down-in-insert-mode": ("b", "last", b"A" + DOWN, {"type": "cell_exit", "id": "b", "direction": "down"}),
    "k-above-markdown": ("m1", "first", b"k", {"type": "cell_exit", "id": "m1", "direction": "up"}),
    "j-below-markdown": ("m2", "last", b"j", {"type": "cell_exit", "id": "m2", "direction": "down"}),
    "down-below-markdown": ("m2", "last", DOWN, {"type": "cell_exit", "id": "m2", "direction": "down"}),
    "up-in-insert-mode": ("m1", "first", b"i" + UP, {"type": "cell_exit", "id": "m1", "direction": "up"}),
    "j-into-the-next-code-cell": ("a", "last", b"j", {"type": "cell_focus", "id": "b"}),
    "j-past-the-last-cell": ("c", "last", b"j", None),
    "k-past-the-first-cell": ("m0", "first", b"k", None),
}


@requires_nvim
@pytest.mark.parametrize(("cell", "cursor", "keys", "expected"), EXITS.values(), ids=EXITS.keys())
async def test_an_up_or_down_key_at_the_edge_toward_a_cell_of_the_other_kind_leaves_the_buffer(
    neovim_home, cell, cursor, keys, expected
):
    async with _notebook(neovim_home, MIXED) as (session, events):
        await _focus_mixed(session, cell, cursor)
        line = await _eval(session, "line('.')")
        await _type(session, keys)
        if expected is not None:
            assert await _next(events) == expected
        await _quiet(events)
        if expected is not None and expected["type"] == "cell_exit" and cell.startswith("m"):
            await _pinned(session, *MIXED_SPANS["a"])  # the markdown editing ended: the code is back in the window
        elif expected is None or expected["type"] == "cell_exit":
            assert await _eval(session, "line('.')") == line  # the cursor stays, the caller activates the neighbor
            await _pinned(session, *MIXED_SPANS[cell])


@requires_nvim
async def test_leaving_a_markdown_cell_by_its_edge_puts_the_code_back_in_the_window(neovim_home):
    async with _notebook(neovim_home, MIXED) as (session, events):
        await _focus_mixed(session, "m2", "last")
        await _type(session, b"A" + DOWN)
        assert await _next(events) == {"type": "cell_exit", "id": "m2", "direction": "down"}
        await _evaluates_to(session, "[&filetype, mode()]", ["python", "n"])
        await _pinned(session, *MIXED_SPANS["a"])  # the code book's active cell
        await _quiet(events)


@requires_nvim
async def test_a_code_cell_left_by_its_edge_is_rendered_before_the_exit(neovim_home):
    async with _notebook(neovim_home, MIXED) as (session, events):
        await _renders(events)
        await _focus_mixed(session, "b", "last")
        await _type(session, b"Ax", ESC)
        assert await _next(events) == {"type": "cell_edit", "id": "b", "text": "print(os.sep)x"}
        session.write_input(b"j")
        assert await _next(events) == {"type": "cell_exit", "id": "b", "direction": "down"}
        exit_at = next(index for index, event in enumerate(events.log) if event["type"] == "cell_exit")
        renders = [event for event in events.log[:exit_at] if event["type"] == "cell_render" and event["id"] == "b"]
        assert _texts(renders[-1]) == ["print(os.sep)x"]


@requires_nvim
async def test_the_edits_of_a_cell_are_reported_before_it_is_left(neovim_home):
    async with _notebook(neovim_home, MIXED) as (session, events):
        await _focus_mixed(session, "b", "last")
        session.write_input(b"Ax" + DOWN)  # typed and gone at once
        assert await _next(events) == {"type": "cell_edit", "id": "b", "text": "print(os.sep)x"}
        assert await _next(events) == {"type": "cell_exit", "id": "b", "direction": "down"}
        await _quiet(events)


@requires_nvim
async def test_a_screen_line_mapping_of_j_goes_through_a_wrapped_last_line_before_it_leaves(neovim_home):
    (neovim_home / "init.lua").write_text("vim.keymap.set({ 'n', 'x' }, 'j', \"v:count == 0 ? 'gj' : 'j'\", { expr = true })\n")
    long = " ".join(["word"] * 40)  # 199 columns: three screen lines in 80
    cells = [{"id": "m", "kind": "markdown", "text": long}, {"id": "c", "text": "x = 1"}]
    async with _notebook(neovim_home, cells) as (session, events):
        await session.focus_cell("m", "first")
        await _evaluates_to(session, "&filetype", "markdown")
        for row in (2, 3):
            await _type(session, b"j")
            assert await _eval(session, "[line('.'), winline()]") == [2, row]
        await _quiet(events)
        await _type(session, b"j")
        assert await _next(events) == {"type": "cell_exit", "id": "m", "direction": "down"}
        await session.focus_cell("m", "first")  # the exit gave the window back to the code
        await _evaluates_to(session, "&filetype", "markdown")
        await _type(session, b"2j")  # with a count the mapping gives plain `j`: the last line is the edge wherever the cursor is on it
        assert await _next(events) == {"type": "cell_exit", "id": "m", "direction": "down"}
        assert await _eval(session, "[line('.'), mode()]") == [2, "n"]  # the count is dropped, Neovim answers again


async def _heights(events: _Events, seconds: float = 0.4) -> list[tuple[str, int]]:
    """The `cell_height`s that came until none came for *seconds*, as (id, rows)."""
    seen = []
    while True:
        try:
            event = await asyncio.wait_for(events.heights.get(), seconds)
        except TimeoutError:
            return seen
        seen.append((event["id"], event["rows"]))


@requires_nvim
async def test_the_active_cell_tells_its_screen_rows_when_it_is_pinned_and_whenever_they_change(neovim_home):
    long = " ".join(["word"] * 40)  # 199 columns: three screen lines in 80
    cells = [{"id": "a", "text": "x = 1\ny = 2"}, {"id": "m", "kind": "markdown", "text": f"# Title\n{long}"}, {"id": "b", "text": long}]
    async with _notebook(neovim_home, cells) as (session, events):
        assert await _heights(events) == [("a", 2)]
        await session.focus_cell("m", "first")
        assert await _heights(events) == [("m", 4)]  # a title and a line wrapped into three
        await session.focus_cell("b", "first")
        assert await _heights(events) == [("b", 3)]

        session.resize(24, 40)  # narrower: the long line takes five rows
        assert await _heights(events) == [("b", 5)]
        await _type(session, b"o", b"z", ESC)
        assert await _heights(events) == [("b", 6)]
        await _type(session, b"k", b"j")  # moving inside the cell changes nothing
        assert await _heights(events) == []
        await session._lua("vim.wo.wrap = false")
        assert await _heights(events) == [("b", 2)]

        await session.set_cells("nb.py", [*cells[:2], {"id": "b", "text": "a\nb\nc"}])  # a new text from the notebook
        assert await _heights(events) == [("b", 3)]
        await session.focus_cell("a", "first")
        await session.focus_cell("b", "first")  # told again: the caller may have sized another cell meanwhile
        assert await _heights(events) == [("a", 2), ("b", 3)]


@requires_nvim
async def test_ctrl_enter_runs_a_cell_in_place_and_shift_enter_runs_and_advances_in_both_buffers(neovim_home):
    async with _notebook(neovim_home, MIXED) as (session, events):
        await _focus_mixed(session, "a", "first")
        session.write_input(b"Ay" + CTRL_ENTER)
        assert await _next(events) == {"type": "cell_edit", "id": "a", "text": "import osy\nx = 1"}
        assert await _next(events) == {"type": "cell_run", "id": "a", "text": "import osy\nx = 1", "advance": False}
        await _type(session, ESC)
        session.write_input(CTRL_ENTER)  # normal mode too
        assert await _next(events) == {"type": "cell_run", "id": "a", "text": "import osy\nx = 1", "advance": False}

        await _focus_mixed(session, "m1", "first")
        session.write_input(CTRL_ENTER)
        assert await _next(events) == {"type": "cell_run", "id": "m1", "text": "Some *text*", "advance": False}
        await _evaluates_to(session, "&filetype", "python")  # the run ended the editing: the code is back in the window
        await _focus_mixed(session, "m1", "first")
        session.write_input(b"A." + SHIFT_ENTER)
        assert await _next(events) == {"type": "cell_edit", "id": "m1", "text": "Some *text*."}
        assert await _next(events) == {"type": "cell_run", "id": "m1", "text": "Some *text*.", "advance": True}
        await _quiet(events)


@requires_nvim
async def test_the_ask_keymap_quotes_lines_of_the_active_cell_counted_from_its_first_line(neovim_home, tmp_path):
    selections: asyncio.Queue[dict] = asyncio.Queue()
    notebook = tmp_path / "nb.ipynb"
    notebook.write_text("{}")
    async with _notebook(neovim_home, MIXED, notebook_path=notebook, on_selection=selections.put_nowait) as (session, events):
        await _focus_mixed(session, "a", "first")
        await _type(session, b"Vj\\a")
        quoted = await asyncio.wait_for(selections.get(), 10)
        assert quoted == {"path": f"{notebook}.py", "start_line": 1, "end_line": 2, "text": "import os\nx = 1"}

        await _focus_mixed(session, "m2", "last")
        await _type(session, b"V\\a")
        quoted = await asyncio.wait_for(selections.get(), 10)
        assert quoted == {"path": f"{notebook}.md", "start_line": 2, "end_line": 2, "text": "text"}
        assert await _eval(session, "mode()") == "n"


@requires_nvim
async def test_escape_in_normal_mode_leaves_a_markdown_cell_after_its_edits(neovim_home):
    async with _notebook(neovim_home, MIXED) as (session, events):
        await _focus_mixed(session, "m2", "first")
        await _type(session, b"iZ", ESC)  # ends the insert only
        assert await _next(events) == {"type": "cell_edit", "id": "m2", "text": "ZMore\ntext"}
        await _quiet(events)
        await _type(session, ESC)
        assert await _next(events) == {"type": "cell_leave", "id": "m2"}
        await _evaluates_to(session, "&filetype", "python")  # the code is back in the window

        await _focus_mixed(session, "m2", "first")
        session.write_input(b"x")
        await asyncio.sleep(0.01)  # within the 40 ms the edit waits
        session.write_input(ESC)
        assert await _next(events) == {"type": "cell_edit", "id": "m2", "text": "More\ntext"}
        assert await _next(events) == {"type": "cell_leave", "id": "m2"}

        await _focus_mixed(session, "a", "first")  # code Escape sends no cell event, the global `leave` is not wired here
        await _type(session, ESC)
        await _quiet(events)


@requires_nvim
async def test_the_users_own_escape_runs_before_a_markdown_cell_is_left(neovim_home):
    (neovim_home / "init.lua").write_text("vim.keymap.set('n', '<Esc>', function() vim.g.escaped = (vim.g.escaped or 0) + 1 end)\n")
    async with _notebook(neovim_home, MIXED) as (session, events):
        await _focus_mixed(session, "m1", "first")
        await _type(session, ESC)
        assert await _next(events) == {"type": "cell_leave", "id": "m1"}
        assert await _eval(session, "g:escaped") == 1


@requires_nvim
async def test_code_cells_render_again_once_a_markdown_cell_is_left(neovim_home):
    async with _notebook(neovim_home, MIXED) as (session, events):
        await _renders(events)
        await _focus_mixed(session, "m1", "first")
        await _type(session, ESC)
        assert await _next(events) == {"type": "cell_leave", "id": "m1"}
        await session.set_cells("nb.py", [*MIXED[:2], {"id": "b", "text": "print(2)"}, *MIXED[3:]])  # changed on disk
        renders = await _renders(events)
        assert list(renders) == ["b"] and _texts(renders["b"]) == ["print(2)"]


@requires_nvim
async def test_only_code_cells_render_and_the_code_cell_left_for_markdown_is_rendered_before_it(neovim_home):
    async with _notebook(neovim_home, MIXED) as (session, events):
        assert set(await _renders(events)) == {"a", "b", "c"}
        await _type(session, b"Az", ESC)
        assert await _next(events) == {"type": "cell_edit", "id": "a", "text": "import osz\nx = 1"}
        assert await _renders(events) == {}  # the active cell is drawn by Neovim itself

        await session.focus_cell("m1", "first")
        render = events.renders.get_nowait()  # sent before the markdown took the window
        assert render["id"] == "a" and _texts(render) == ["import osz", "x = 1"]

        # While the markdown is in the window, renders wait: they follow the window's options
        a = {"id": "a", "kind": "code", "text": "import osz\nx = 1"}  # as Neovim told it
        await session.set_cells("nb.py", [MIXED[0], a, {"id": "b", "text": "print(1)"}, *MIXED[3:]])
        await _type(session, b"Aq", ESC)
        assert await _next(events) == {"type": "cell_edit", "id": "m1", "text": "Some *text*q"}
        assert await _renders(events) == {}

        await session.focus_cell("c", "first")
        renders = await _renders(events)
        assert list(renders) == ["b"] and _texts(renders["b"]) == ["print(1)"]


@requires_nvim
async def test_the_markdown_buffer_is_named_beside_the_notebook_never_written_and_follows_a_rename(neovim_home, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    notebook = project / "analysis.ipynb"
    notebook.write_text("{}")
    async with _notebook(neovim_home, MIXED, notebook_path=notebook) as (session, events):
        await _focus_mixed(session, "m1", "first")
        assert await _eval(session, "[expand('%:p'), &filetype, &swapfile, &undofile]") == [f"{notebook}.md", "markdown", 0, 0]
        assert await _lines(session) == MARKDOWN_SCRIPT
        await _type(session, b"Ax", ESC)
        for command in (":w", ":1,2w!", ":up"):
            await _type(session, command.encode() + b"\r")
            await asyncio.sleep(0.2)
        assert [event["type"] for event in await _settled(events)][:2] == ["cell_edit", "save_notebook"]

        renamed = project / "report.ipynb"
        notebook.rename(renamed)
        await session.rename(str(notebook), str(renamed), renamed)
        assert await _eval(session, "[expand('%:p'), &filetype]") == [f"{renamed}.md", "markdown"]
        await _focus_mixed(session, "a", "first")
        assert await _eval(session, "expand('%:p')") == f"{renamed}.py"
        await _type(session, b":w\r")
        assert await _next(events) == {"type": "save_notebook"}
        await asyncio.sleep(0.3)
        assert sorted(path.name for path in project.iterdir()) == ["report.ipynb"]


@requires_nvim
async def test_the_users_markdown_settings_reach_the_window_once_it_shows_the_markdown(neovim_home):
    (neovim_home / "init.lua").write_text(
        "vim.api.nvim_create_autocmd('FileType', { pattern = 'markdown', callback = function() vim.opt_local.spell = true end })\n"
    )
    async with _notebook(neovim_home, MIXED) as (session, events):
        assert await _eval(session, "&spell") == 0
        await _focus_mixed(session, "m0", "first")
        assert await _eval(session, "&spell") == 1
        await _focus_mixed(session, "a", "first")
        assert await _eval(session, "&spell") == 0
        await _focus_mixed(session, "m1", "first")
        assert await _eval(session, "&spell") == 1
        await _quiet(events)


@requires_nvim
async def test_set_cells_changes_the_markdown_buffer_as_it_does_the_code_buffer(neovim_home):
    async with _notebook(neovim_home, MIXED) as (session, events):
        await _focus_mixed(session, "m2", "first")
        changed = [*MIXED[:4], {"id": "m2", "kind": "markdown", "text": "Changed\ntext"}, MIXED[5]]
        await session.set_cells("nb.py", changed)  # one cell's text: replaced in place
        assert await _lines(session) == [*MARKDOWN_SCRIPT[:6], "Changed", "text"]
        await _pinned(session, 7, 8)

        # b turns into a markdown cell: it leaves the script and joins the markdown between m0 and m1
        turned = [MIXED[0], MIXED[1], {"id": "b", "kind": "markdown", "text": "print(os.sep)"}, *changed[3:]]
        await session.set_cells("nb.py", turned)
        expected = [*MARKDOWN_SCRIPT[:3], "<!-- %% -->", "print(os.sep)", *MARKDOWN_SCRIPT[3:6], "Changed", "text"]
        assert await _lines(session) == expected
        await _focus_mixed(session, "a", "first")
        assert await _lines(session) == ["# %%", "import os", "x = 1", "# %%", "w = 4"]
        await _quiet(events)

        await session.focus_cell("m0", "first")
        await _type(session, b":bwipeout!\r")  # Neovim lost the markdown buffer: the next set_cells makes it again
        await _evaluates_to(session, "bufname() =~ '\\.md$'", 0)
        await session.set_cells("nb.py", MIXED)
        await _focus_mixed(session, "m1", "first")
        assert await _lines(session) == MARKDOWN_SCRIPT
        await _settled(events)
        session.write_input(b"Ax")
        assert await _next(events) == {"type": "cell_edit", "id": "m1", "text": "Some *text*x"}


@requires_nvim
async def test_markdown_frames_over_the_socket(neovim_home, tmp_path, monkeypatch):
    async with _notebook_browser(monkeypatch, tmp_path) as browser:
        browser.send({"type": "set_cells", "filename": "nb.py", "cells": MIXED})
        browser.send({"type": "focus_cell", "id": "m1", "cursor": "first"})
        browser.send({"type": "message", "text": "Settled"})
        await browser.shows("Settled")
        await browser.silent()

        browser.type(b"A!")
        assert await browser.next_frame("cell_edit") == {"type": "cell_edit", "id": "m1", "text": "Some *text*!"}
        browser.type(CTRL_ENTER)
        assert await browser.next_frame("cell_run") == {"type": "cell_run", "id": "m1", "text": "Some *text*!", "advance": False}
        browser.send({"type": "focus_cell", "id": "m1", "cursor": "first"})  # the run ended the editing, the user edits again
        await asyncio.sleep(0.2)
        browser.type(ESC)
        assert await browser.next_frame("cell_leave") == {"type": "cell_leave", "id": "m1"}
        browser.send({"type": "focus_cell", "id": "m1", "cursor": "first"})
        await asyncio.sleep(0.2)
        browser.type(b"k")
        assert await browser.next_frame("cell_exit") == {"type": "cell_exit", "id": "m1", "direction": "up"}
        await browser.silent()

        renders = set()
        while not browser.renders.empty():
            renders.add(browser.renders.get_nowait()["id"])
        assert renders and renders <= {"a", "b", "c"}
