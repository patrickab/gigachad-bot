r"""SyncTeX lookups between a built PDF and its LaTeX sources.

- `synctex_source` maps a point of a built PDF back to the source file, line and column that typeset it, refined by the clicked word
- `synctex_view` maps a source line forward to its page, the top-left corner of the box it typeset and the words to find there
- `slide_end` finds the source line after the slide on a page of a beamer PDF, where a new frame goes
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import gzip
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import unicodedata
import zlib

from lib.latex import BUILD_DIRECTORY, build_output
from lib.tex_source import COMMENT, frame_begin, frame_end, plain_lines, whole_word

_SYNCTEX_TIMEOUT_SECONDS = 10
_VIEW_WORDS = 8
# Arguments that typeset no text of their own: environment names, files, labels, colors, links
_UNTYPESET_ARGUMENT = re.compile(
    r"\\(?:begin|end|includegraphics|input|include|label|ref|eqref|cref|Cref|autoref|cite[pt]?|usepackage|usetheme"
    r"|documentclass|textcolor|color|colorbox|href|url|hspace|vspace)\*?\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}"
)
_INLINE_MATH = re.compile(r"(?<!\\)\$[^$]*(?<!\\)\$")
_OPTIONS = re.compile(r"\[[^\]]*\]|<[^<>]*>")
_COMMAND = re.compile(r"\\(?:[A-Za-z@]+\*?|.)")
_WORD = re.compile(r"[^\W\d_]{4,}")
# A SyncTeX record of a box, kern, glue, math node or rule: `<kind><input tag>,<line>[,<column>]:...`
_SYNCTEX_RECORD = re.compile(r"[\[(vhxkg$r](\d+),(\d+)")
_HEADING = re.compile(r"\\(?P<sub>(?:sub)*)section\b")

log = logging.getLogger(__name__)


class SynctexUnavailable(RuntimeError):
    """`synctex` is not installed."""


@dataclass(frozen=True)
class SynctexSpot:
    """The 1-based *page* and top-left corner of what a source line typeset, in PDF points from the page's top-left, and its *words*."""

    page: int
    x: float
    y: float
    words: list[str] = field(default_factory=list)


def synctex_source(
    root: Path, page: int, x: float, y: float, word: str | None = None, offset: int = 0, occurrence: int = 0,
) -> tuple[Path, int, int] | None:
    """The source file, 1-based line and 0-based UTF-8 byte column that typeset the point (*x*, *y*) of *page* in the last build of *root*.

    The point is in PDF points from the top-left corner of the 1-based *page*. The clicked *word* refines the line with
    `refine_line`, then character *offset* of its *occurrence* on its rendered line gives the column (`word_column`), else
    column 0. None without SyncTeX data or a match. Raises `SynctexUnavailable` when `synctex` is missing.
    """
    # synctex finds `<stem>.synctex.gz` beside the PDF it is given, which need not exist
    result = _run_synctex(root, ["edit", "-o", f"{page}:{x}:{y}:{build_output(root, '.pdf')}"])
    if result is None:
        return None
    match = _first_synctex_match(result, root.parent)
    if match is None:
        return None
    source, line = match
    if not word:
        return source, line, 0
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return source, line, 0
    # The PDF may hold a ligature glyph (`ﬁ`) or a decomposed accent where the source spells plain letters
    typeset = unicodedata.normalize("NFKC", word)
    offset = len(unicodedata.normalize("NFKC", word[:max(offset, 0)]))
    line = refine_line(text, line, typeset)
    lines = text.splitlines()
    return source, line, word_column(lines[line - 1], typeset, offset, occurrence) if line <= len(lines) else 0


def synctex_view(root: Path, file: Path, line: int) -> SynctexSpot | None:
    """The 1-based page and top-left corner, in PDF points from the page's top-left, of what 1-based *line* of *file* typeset.

    A line inside a beamer frame asks for the frame's `\\end{frame}` line, which SyncTeX attributes the whole frame to, so the
    corner is the frame's and `words` of the line tell where on the page it is. Reads the last build of *root*. None without
    SyncTeX data or a match. Raises `SynctexUnavailable` when `synctex` is missing.
    """
    try:
        lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    pdf = build_output(root, ".pdf")
    result = _run_synctex(root, ["view", "-i", f"{frame_end(lines, line)}:0:{file}", "-o", str(pdf)])
    box = None if result is None else _first_synctex_box(result)
    if box is None:
        return None
    page, x, y = box
    return SynctexSpot(page=page, x=x, y=y, words=line_words(lines[line - 1]) if line <= len(lines) else [])


def line_words(text: str) -> list[str]:
    r"""The first distinct words of 4 letters or more the source line *text* typesets, without comments, math, `\commands` and options."""
    plain = COMMENT.sub("", text)
    for pattern in (_UNTYPESET_ARGUMENT, _INLINE_MATH, _OPTIONS):
        plain = pattern.sub(" ", plain)
    words = dict.fromkeys(_WORD.findall(_COMMAND.sub(" ", plain)))
    return list(words)[:_VIEW_WORDS]


def _run_synctex(root: Path, arguments: list[str]) -> str | None:
    """The output of `synctex <arguments>` run beside *root*, None without SyncTeX data of *root*, on failure or timeout."""
    if shutil.which("synctex") is None:
        raise SynctexUnavailable("synctex is not installed on the server, install a TeX distribution such as TeX Live")
    if not build_output(root, ".synctex.gz").is_file():
        return None
    try:
        result = subprocess.run(
            ["synctex", *arguments], cwd=root.parent, stdin=subprocess.DEVNULL, capture_output=True,
            timeout=_SYNCTEX_TIMEOUT_SECONDS, check=False,
        )
    except subprocess.TimeoutExpired:
        log.warning("synctex timed out for %s", root)
        return None
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", errors="replace")


def refine_line(text: str, line: int, word: str) -> int:
    r"""The first line from the `\begin{frame}` enclosing 1-based *line* of *text* to *line* that holds *word*, else *line*."""
    # beamer reads a frame body as a macro argument, so SyncTeX puts all of it on the frame's `\end{frame}` line.
    # A whole word that is not a `\command` name first ("the" is not in `\theme` or "Other"), exact case first
    lines = plain_lines(text.splitlines()[:line])
    begin = frame_begin(lines)
    if begin is None:
        return line
    frame = list(enumerate(lines[begin:], start=begin + 1))
    whole, folded = whole_word(word), word.casefold()
    holds = (
        whole.search, re.compile(whole.pattern, re.IGNORECASE).search,
        lambda entry: word in entry, lambda entry: folded in entry.casefold(),
    )
    return next((number for test in holds for number, entry in frame if test(entry)), line)


def word_column(text: str, word: str, offset: int = 0, occurrence: int = 0) -> int:
    r"""The 0-based UTF-8 byte column of character *offset* of the *occurrence*-th whole *word* (from 0, clamped) of the source line *text*, exact case first, else 0."""
    # A `\command` name or a comment holds no word
    plain = COMMENT.sub("", text)
    whole = whole_word(word)
    matches = list(whole.finditer(plain)) or list(re.compile(whole.pattern, re.IGNORECASE).finditer(plain))
    if not matches:
        return 0
    match = matches[min(max(occurrence, 0), len(matches) - 1)]
    character = match.start() + min(max(offset, 0), len(match.group()) - 1)
    return len(text[:character].encode("utf-8"))


def _first_synctex_match(output: str, folder: Path) -> tuple[Path, int] | None:
    """The first `Input:` file, resolved against *folder*, and its following `Line:` of `synctex edit` *output*."""
    source: Path | None = None
    for line in output.splitlines():
        key, _, value = line.partition(":")
        if key == "Input":
            source = Path(os.path.normpath(folder / value))
        elif key == "Line" and source is not None:
            number = int(value) if value.strip().isdigit() else 0
            return (source, number) if number >= 1 else None
    return None


def _first_synctex_box(output: str) -> tuple[int, float, float] | None:
    """The page and top-left corner of the first record of `synctex view` *output*.

    A record's box has its left edge at `h` and its baseline at `v`, both from the page's top-left, and rises `H` above it.
    """
    fields: dict[str, str] = {}
    for line in output.splitlines():
        key, _, value = line.partition(":")
        if key == "Page" and fields:
            break  # the next record
        if key in ("Page", "h", "v", "H"):
            fields[key] = value.strip()
    try:
        page, left, baseline, height = int(fields["Page"]), float(fields["h"]), float(fields["v"]), float(fields["H"])
    except (KeyError, ValueError):
        return None
    return (page, left, max(0.0, baseline - height)) if page >= 1 else None


def slide_end(root: Path, page: int) -> tuple[Path, int] | None:
    r"""The `.tex` file of *root*'s folder and 1-based line after the slide on 1-based *page* of its last build, where a new frame goes, None without one."""
    folder = root.parent.resolve()
    sources: dict[Path, list[str]] = {}
    ends: Counter[tuple[Path, int]] = Counter()
    for (name, line), count in _page_lines(build_output(root, ".synctex.gz"), page).items():
        path = Path(os.path.normpath(root.parent / name))
        if path not in sources:
            sources[path] = _source_lines(path, folder)
        if sources[path]:
            ends[path, frame_end(sources[path], line)] += count
    # SyncTeX puts a whole frame on its `\end{frame}` line. A page without one is drawn by a command, e.g. a theme's chapter slide by `\section`
    framed = Counter({(path, line): count for (path, line), count in ends.items() if r"\end{frame}" in _plain_line(sources[path], line)})
    best = (framed or ends).most_common(1)
    if not best:
        return None
    (path, line), _ = best[0]
    return path, line if framed else _after_headings(sources[path], line)


def _page_lines(synctex: Path, page: int) -> Counter[tuple[str, int]]:
    """How many records of 1-based *page* the gzipped SyncTeX file *synctex* puts on each (input file as named there, line)."""
    inputs: dict[str, str] = {}
    tags: Counter[tuple[str, int]] = Counter()
    try:
        with gzip.open(synctex, "rt", encoding="utf-8", errors="replace") as file:
            on_page = False
            for entry in file:
                if entry.startswith("Input:"):
                    tag, _, name = entry[len("Input:"):].rstrip("\n").partition(":")
                    inputs[tag] = name
                elif entry.startswith("{"):
                    on_page = entry[1:].strip() == str(page)
                elif entry.startswith("}") and on_page:
                    break
                elif on_page and (record := _SYNCTEX_RECORD.match(entry)):
                    tags[record[1], int(record[2])] += 1
    except (OSError, EOFError, zlib.error):
        return Counter()
    lines: Counter[tuple[str, int]] = Counter()
    for (tag, line), count in tags.items():
        if tag in inputs:
            lines[inputs[tag], line] += count
    return lines


def _source_lines(path: Path, folder: Path) -> list[str]:
    """The lines of *path* when it is a readable `.tex` source in the resolved *folder* (build outputs excluded), else none."""
    resolved = path.resolve()
    if path.suffix != ".tex" or not resolved.is_relative_to(folder) or BUILD_DIRECTORY in resolved.relative_to(folder).parts:
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _plain_line(lines: list[str], line: int) -> str:
    """1-based *line* of *lines* without its comment, empty past the end."""
    return COMMENT.sub("", lines[line - 1]) if 1 <= line <= len(lines) else ""


def _after_headings(lines: list[str], line: int) -> int:
    r"""The last of the deeper headings right below the heading on 1-based *line* of *lines* (`\subsection` below `\section`), else *line*."""
    heading = _HEADING.search(_plain_line(lines, line))
    if heading is None:
        return line
    depth = len(heading["sub"])
    end = line
    for number in range(line + 1, len(lines) + 1):
        text = _plain_line(lines, number)
        below = _HEADING.search(text)
        if below is not None and len(below["sub"]) > depth:
            end = number
        elif text.strip():
            break
    return end
