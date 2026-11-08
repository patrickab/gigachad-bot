import pytest

from lib.tex_source import fill_placeholder, frame_insertion

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
COMMENTED = [*SLIDES[:5], r"% \begin{frame}{Old}", *SLIDES[6:]]
INLINE = [*SLIDES[:3], r"a % \end{frame}", *SLIDES[4:]]
OPTIONS = [*SLIDES[:2], r"\begin{frame}<2->[plain]{One}", *SLIDES[3:]]
ONE_LINE = [*SLIDES[:2], r"\begin{frame}[plain]\statement{a}{b}\end{frame}", SLIDES[-1]]
TRAILING = [*SLIDES, "notes"]
UNENDED = [*SLIDES[:2], "a"]


def _inserted(lines: list[str], cursor: int, frame: list[str]) -> tuple[list[str], int]:
    """*lines* with *frame* inserted for a cursor on line *cursor*, and the line the cursor lands on."""
    insertion = frame_insertion(lines, cursor, frame)
    return [*lines[: insertion.row], *insertion.lines, *lines[insertion.row :]], insertion.cursor_row


@pytest.mark.parametrize(
    ("lines", "cursor", "expected", "landed"),
    [
        # Inside a frame: after its end, a blank line apart
        (SLIDES, 4, [*SLIDES[:5], "", *NEW_FRAME, *SLIDES[5:]], 7),
        # Between two frames: at the cursor
        (SLIDES, 6, [*SLIDES[:6], *NEW_FRAME, "", *SLIDES[6:]], 7),
        # On `\end{document}`: right before it
        (SLIDES, 10, [*SLIDES[:9], "", *NEW_FRAME, "", *SLIDES[9:]], 11),
        # After `\end{document}`: right before it
        (TRAILING, 11, [*TRAILING[:9], "", *NEW_FRAME, "", *TRAILING[9:]], 11),
        # A commented `\begin{frame}` is no frame
        (COMMENTED, 6, [*COMMENTED[:6], "", *NEW_FRAME, "", *COMMENTED[6:]], 8),
        # In the preamble: right before `\end{document}`
        (SLIDES, 1, [*SLIDES[:9], "", *NEW_FRAME, "", *SLIDES[9:]], 11),
        # In the preamble without `\end{document}`: right after `\begin{document}`
        (UNENDED, 1, [*UNENDED[:2], "", *NEW_FRAME, "", *UNENDED[2:]], 4),
        # No document environment at all: at the cursor
        (["abc"], 1, ["abc", "", *NEW_FRAME], 3),
        # An inline `% \end{frame}` comment ends no frame
        (INLINE, 4, [*INLINE[:5], "", *NEW_FRAME, *INLINE[5:]], 7),
        # A one-line frame under the cursor: after it
        (ONE_LINE, 3, [*ONE_LINE[:3], "", *NEW_FRAME, "", *ONE_LINE[3:]], 5),
        # On `\end{frame}`: after it
        (SLIDES, 5, [*SLIDES[:5], "", *NEW_FRAME, *SLIDES[5:]], 7),
        # A frame with overlay and options
        (OPTIONS, 4, [*OPTIONS[:5], "", *NEW_FRAME, *OPTIONS[5:]], 7),
    ],
)
def test_a_frame_goes_after_the_enclosing_frame_never_in_the_preamble_nor_after_the_document(lines, cursor, expected, landed):
    assert _inserted(lines, cursor, NEW_FRAME) == (expected, landed)


@pytest.mark.parametrize(
    ("frame", "landed"),
    [
        ([r"% \begin{frame}{Old}", r"\begin{frame}{New}", r"\end{frame}"], 8),
        (["x"], 7),
    ],
)
def test_the_cursor_lands_on_the_frames_begin_line_else_on_its_first_line(frame, landed):
    assert _inserted(SLIDES, 6, frame) == ([*SLIDES[:6], *frame, "", *SLIDES[6:]], landed)


@pytest.mark.parametrize(
    ("line", "filled"),
    [
        (r"\roundimage{example-image-16x9}", r"\roundimage{imgs/100%.png}"),
        # Several placeholders: the first only
        (r"\figrow{example-image-a}{example-image-b}{example-image}", r"\figrow{imgs/100%.png}{example-image-b}{example-image}"),
        # ASCII letters, digits and `-` only, as the old Lua `[%w%-]`
        (r"\roundimage{example-image_x}", r"\roundimage{imgs/100%.png_x}"),
        ("text", None),
    ],
)
def test_fill_placeholder_puts_the_path_into_the_first_placeholder_only(line, filled):
    assert fill_placeholder(line, "imgs/100%.png") == filled
