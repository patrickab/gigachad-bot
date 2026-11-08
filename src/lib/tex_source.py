r"""Plain-text helpers for LaTeX sources.

Line numbers are 1-based like Neovim's cursor, only `frame_begin` answers a list index.

- `plain_lines` strips the comments of source lines
- `frame_begin` and `frame_end` find the beamer frame around a source line
- `frame_insertion` places a new beamer frame for a cursor line: after the frame around it, else below it, never in the
  preamble nor after `\end{document}`
- `fill_placeholder` puts an image path into a line's first `example-image` placeholder (`PLACEHOLDER`)
- `whole_word` matches a word standing alone, not inside a longer word or a `\command` name
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import string

COMMENT = re.compile(r"(?<!\\)%.*")
# An mwe placeholder image the templates show until the user picks an image: `example-image`, `example-image-16x9`, ...
PLACEHOLDER = re.compile(r"example-image[A-Za-z0-9-]*")


@dataclass(frozen=True)
class FrameInsertion:
    """Where a new frame goes: its padded *lines* below line *row* (0: above the first line), the cursor then on *cursor_row*."""

    row: int
    lines: list[str]
    cursor_row: int


def plain_lines(source: str | list[str]) -> list[str]:
    """Strip the comment of each line of *source*, a text (split keeping line ends) or its lines."""
    lines = source.splitlines(keepends=True) if isinstance(source, str) else source
    return [COMMENT.sub("", line) for line in lines]


def frame_begin(lines: list[str]) -> int | None:
    r"""The index of the `\begin{frame}` line enclosing the last of comment-free *lines*, None outside a frame."""
    last = len(lines) - 1
    for index in range(last, -1, -1):
        opened = lines[index].rfind(r"\begin{frame}")
        # The last line is the frame's own `\end{frame}`, an earlier one closes the previous frame
        closed = lines[index].rfind(r"\end{frame}") if index < last else -1
        if opened > closed:
            return index
        if closed >= 0:
            return None
    return None


def frame_end(lines: list[str], line: int) -> int:
    r"""The `\end{frame}` line of the beamer frame enclosing *line* of *lines*, else *line*."""
    plain = plain_lines(lines)
    if frame_begin(plain[:line]) is None:
        return line
    return next((number for number in range(line, len(plain) + 1) if r"\end{frame}" in plain[number - 1]), line)


def frame_insertion(lines: list[str], cursor_row: int, frame: list[str]) -> FrameInsertion:
    r"""Place the beamer *frame* in the source *lines* whose cursor is on *cursor_row*.

    It goes after the frame enclosing the cursor, else below the cursor. From the preamble it goes before `\end{document}`
    (below `\begin{document}` without one), from `\end{document}` or after it right before it. A blank line parts it from a
    non-blank neighbour, the cursor lands on its `\begin{frame}` line (its first without one).
    """
    plain = plain_lines(lines)
    row = frame_end(lines, cursor_row)
    ending = next((number for number, text in enumerate(plain, 1) if r"\end{document}" in text), None)
    opening = next((number for number, text in enumerate(plain[:ending], 1) if r"\begin{document}" in text), None)
    if opening is not None and row < opening:
        row = opening if ending is None else ending - 1
    if ending is not None and row >= ending:
        row = ending - 1
    begin = next((number for number, text in enumerate(plain_lines(frame), 1) if r"\begin{frame}" in text), 1)
    gap_above = [""] if row > 0 and not _is_blank(lines[row - 1]) else []
    gap_below = [""] if row < len(lines) and not _is_blank(lines[row]) else []
    return FrameInsertion(row, [*gap_above, *frame, *gap_below], row + len(gap_above) + begin)


def fill_placeholder(line: str, path: str) -> str | None:
    """*line* with its first `PLACEHOLDER` replaced by *path* taken literally, None without one."""
    placeholder = PLACEHOLDER.search(line)
    return None if placeholder is None else line[: placeholder.start()] + path + line[placeholder.end() :]


def whole_word(word: str) -> re.Pattern[str]:
    r"""*word* standing alone, not inside a longer word or a `\command` name."""
    return re.compile(rf"(?<![\w\\]){re.escape(word)}(?!\w)")


def _is_blank(line: str) -> bool:
    """Whether *line* holds nothing but ASCII whitespace."""
    return not line.strip(string.whitespace)
