"""A notebook in Neovim: one buffer for all code cells, its window pinned to the active cell, and the frames the socket carries."""

import asyncio
import contextlib
from pathlib import Path
import shutil

import pytest

from backend import websocket_origin
from backend.routes import neovim as neovim_route
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


@contextlib.asynccontextmanager
async def _notebook(home, cells=CELLS, *, rows=24, filename="nb.py", notebook_path=None):
    """A connected notebook session showing *cells*, and the queue of the events it reports."""
    events: asyncio.Queue[dict] = asyncio.Queue()
    session = NeovimSession.spawn(
        "placeholder.txt", "", rows=rows, cols=80, cwd=home, path="placeholder", cell_session=True, notebook_path=notebook_path
    )
    drain = asyncio.create_task(_drain(session))
    try:
        await asyncio.wait_for(session.connect(_save, lambda text: None, on_cell=events.put_nowait), 15)
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
            "_G.gigachad_set_cells(nil, ..., { { id = 'a', text = 'import os\\nx = 1' } })",
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
        assert await _next(events) == {"type": "cell_run", "id": "b", "text": "print(os.sep)x\ny = 2\nz = 3"}

        await _type(session, ESC)
        session.write_input(SHIFT_ENTER)  # normal mode, nothing new to report
        assert await _next(events) == {"type": "cell_run", "id": "b", "text": "print(os.sep)x\ny = 2\nz = 3"}

        # After moving on, the new cell is the one that runs
        await session.focus_cell("c", "first")
        session.write_input(SHIFT_ENTER)
        assert await _next(events) == {"type": "cell_run", "id": "c", "text": "w = 4"}
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


@contextlib.asynccontextmanager
async def _notebook_browser(monkeypatch, tmp_path, *, notebook: Path | None = None, root: Path | None = None):
    """The session endpoint serving a browser that opened a notebook, with Neovim connected once the body starts."""
    monkeypatch.setattr(websocket_origin, "CORS_ORIGINS", ["*"])
    browser = _Browser()
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

        browser.type(b"Ax")  # on the last line of cell b
        assert await browser.next_frame("cell_edit") == {"type": "cell_edit", "id": "b", "text": "print(os.sep)\ny = 2\nz = 3x"}

        browser.type(ESC)
        await asyncio.sleep(0.1)
        browser.type(SHIFT_ENTER)
        assert await browser.next_frame("cell_run") == {"type": "cell_run", "id": "b", "text": "print(os.sep)\ny = 2\nz = 3x"}

        browser.type(DOWN)  # past the last line of b: into c
        assert await browser.next_frame("cell_focus") == {"type": "cell_focus", "id": "c"}

        browser.send(
            {"type": "set_cells", "filename": "nb.py", "cells": [*CELLS[:2], {"id": "c", "text": "QRSTU"}, {"id": "d", "text": "new"}]}
        )
        browser.send({"type": "focus_cell", "id": "d", "cursor": "first"})
        await browser.shows("new")
        browser.type(b":w\r")
        assert await browser.next_frame("save_notebook") == {"type": "save_notebook"}


@requires_nvim
async def test_malformed_notebook_frames_are_ignored_and_the_session_lives_on(neovim_home, tmp_path, monkeypatch):
    async with _notebook_browser(monkeypatch, tmp_path) as browser:
        browser.send({"type": "set_cells", "filename": "nb.py", "cells": CELLS})
        for bad in (
            {"type": "set_cells", "filename": "nb.py", "cells": "nope"},
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
                {"type": "cell_run", "id": "a", "text": text},
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
