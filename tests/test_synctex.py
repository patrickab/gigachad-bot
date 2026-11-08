from pathlib import Path
import shutil

import pytest

from lib import synctex
from tests.test_latex import write_files

SYNCTEX_DOCUMENT = (
    "\\documentclass{article}\n"
    "\\begin{document}\n"
    "First page text.\n"
    "\\newpage\n"
    "\\input{sections/chapter}\n"
    "\\newpage\n"
    "\\input{../../elsewhere/aside}\n"
    "\\end{document}\n"
)
# A point on the first text line of a page of an article
TOP_LINE = {"x": 150, "y": 130}
needs_synctex = pytest.mark.skipif(
    not all(shutil.which(name) for name in ("latexmk", "pdflatex", "synctex")), reason="needs latexmk, pdflatex and synctex",
)


@needs_synctex
def test_synctex_maps_a_pdf_point_to_the_source_line_of_the_root_or_an_input_file(tex_client, vault, tmp_path):
    write_files(vault, {"paper/main.tex": SYNCTEX_DOCUMENT, "paper/sections/chapter.tex": "Second page text.\n"})
    write_files(tmp_path, {"elsewhere/aside.tex": "Third page text.\n"})
    root = vault / "paper" / "main.tex"
    assert tex_client.post("/api/latex/build", json={"path": str(root)}).json()["ok"] is True

    def source_of(page: int):
        return tex_client.post("/api/latex/synctex", json={"path": str(root), "page": page, **TOP_LINE})

    assert source_of(1).json() == {"path": str(root), "line": 3, "column": 0}
    assert source_of(2).json() == {"path": str(vault / "paper" / "sections" / "chapter.tex"), "line": 1, "column": 0}
    assert source_of(3).status_code == 403
    assert source_of(4).status_code == 404


@needs_synctex
def test_synctex_view_maps_a_source_line_to_the_top_left_of_its_text_on_its_page(tex_client, vault, tmp_path):
    write_files(vault, {"paper/main.tex": SYNCTEX_DOCUMENT, "paper/sections/chapter.tex": "Second page text.\n"})
    write_files(tmp_path, {"elsewhere/aside.tex": "Third page text.\n"})
    root = vault / "paper" / "main.tex"
    assert tex_client.post("/api/latex/build", json={"path": str(root)}).json()["ok"] is True

    def spot_of(file: Path, line: int):
        return tex_client.post("/api/latex/synctex/view", json={"path": str(root), "file": str(file), "line": line})

    second = spot_of(vault / "paper" / "sections" / "chapter.tex", 1).json()
    assert second["page"] == 2
    # The first text line of an article page: the 1in margin plus the header, above the line that `TOP_LINE` clicks
    assert 100 < second["x"] < 160 and 100 < second["y"] < TOP_LINE["y"]
    assert spot_of(root, 3).json()["page"] == 1
    assert spot_of(tmp_path / "elsewhere" / "aside.tex", 1).status_code == 403


def test_synctex_view_answers_404_before_a_build_and_503_without_synctex(tex_client, document, monkeypatch):
    def spot():
        return tex_client.post("/api/latex/synctex/view", json={"path": str(document), "file": str(document), "line": 1})

    monkeypatch.setattr(synctex.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert spot().status_code == 404
    monkeypatch.setattr(synctex.shutil, "which", lambda name: None)
    assert spot().status_code == 503
    assert "synctex" in spot().json()["detail"]


SLIDES = (
    "\\documentclass{beamer}\n"
    "\\begin{document}\n"
    "\\begin{frame}{Title}\n"
    "First body line.\n"
    "\n"
    "Second \\the\\year{} café and the body of the slide.\n"
    "\n"
    "Third body paragraph.\n"
    "\\end{frame}\n"
    "\\end{document}\n"
)
# A point on "Second", which SyncTeX puts on `\end{frame}` like the rest of the frame body
SECOND_WORD = {"page": 1, "x": 40, "y": 128}


@needs_synctex
def test_synctex_finds_the_clicked_word_in_a_beamer_frame(tex_client, vault):
    write_files(vault, {"slides/main.tex": SLIDES})
    root = vault / "slides" / "main.tex"
    assert tex_client.post("/api/latex/build", json={"path": str(root)}).json()["ok"] is True

    def source_of(**word: str | int) -> dict:
        return tex_client.post("/api/latex/synctex", json={"path": str(root), **SECOND_WORD, **word}).json()

    assert source_of()["line"] == 9
    assert source_of(word="Second")["line"] == 6
    # The "h" of the second "the" the slide shows: `\the` is a command, "é" two bytes
    text = SLIDES.splitlines()[5]
    clicked = source_of(word="the", offset=1, occurrence=1)
    assert (clicked["line"], clicked["column"]) == (6, len(text[:text.index("the slide") + 1].encode()))
    assert source_of()["column"] == 0


WORDS_LINE = "\\point{Other notes}{What to see in the first state, the theme.} % the end"


@pytest.mark.parametrize(("word", "offset", "occurrence", "column"), [
    ("the", 1, 0, WORDS_LINE.index("the first") + 1),  # not in "Other"
    ("the", 2, 1, WORDS_LINE.index("the theme") + 2),
    ("the", 0, 7, WORDS_LINE.index("the theme")),  # past the last occurrence, not in the comment
    ("theme", 9, 0, WORDS_LINE.index("theme.") + 4),  # an offset past the word stays on its last letter
    ("point", 0, 0, 0),  # a command name holds no word
    ("what", 0, 0, WORDS_LINE.index("What")),  # in another case
    ("absent", 3, 0, 0),
])
def test_the_clicked_character_gives_the_column_of_its_word_on_the_source_line(word, offset, occurrence, column):
    assert synctex.word_column(WORDS_LINE, word, offset, occurrence) == column


def test_the_column_counts_utf8_bytes_and_skips_a_command_of_the_same_name():
    line = "\\the\\year{} Café — the state"
    assert synctex.word_column(line, "the", 1) == len(line[:line.index("the state") + 1].encode())
    assert synctex.word_column(line, "state") == len(line.encode()) - len("state")


def test_the_clicked_word_prefers_a_line_holding_it_as_a_whole_word():
    text = "\\begin{frame}{Other notes}\n\\theoremstyle{plain}\nWhat to see in the first state.\n\\end{frame}\n"
    assert synctex.refine_line(text, 4, "the") == 3
    assert synctex.refine_line(text, 4, "oth") == 1


FRAMES = (
    "\\begin{frame}{Results}\n"
    "The plot shows growth.\n"
    "\\end{frame}\n"
    "\\begin{frame}{Method}\n"
    "We measure growth weekly.\n"
    "% \\begin{frame} left over\n"
    "Then we plot it.\n"
    "\\end{frame}\n"
    "Closing words.\n"
)


@pytest.mark.parametrize(("word", "line"), [
    ("measure", 5),  # past the commented-out `\begin{frame}`
    ("plot", 7),  # also in the frame before
    ("then", 7),  # in another case
    ("shows", 8),  # only in the frame before
    ("absent", 8),
])
def test_the_clicked_word_picks_its_own_line_of_the_enclosing_frame(word, line):
    # SyncTeX answers the frame's `\end{frame}` line for all of its body
    assert synctex.refine_line(FRAMES, 8, word) == line


def test_the_clicked_word_keeps_a_line_outside_a_frame():
    assert synctex.refine_line(FRAMES, 9, "plot") == 9
    assert synctex.refine_line("\\section{Intro}\nFirst words.\nMore words.\n", 3, "First") == 3


TWO_FRAMES = (
    "\\documentclass{beamer}\n"
    "\\newcommand{\\hl}[1]{#1}\n"
    "\\begin{document}\n"
    "\\begin{frame}{Results}\n"
    "Growth doubled this quarter.\n"
    "\\end{frame}\n"
    "\\begin{frame}{Method}\n"
    "We measure \\hl{weekly} samples. % \\begin{frame} commented\n"
    "Then we plot them.\n"
    "\\end{frame}\n"
    "\\end{document}\n"
)


@needs_synctex
def test_synctex_view_puts_a_line_inside_a_beamer_frame_on_the_page_of_its_frame(tex_client, vault):
    write_files(vault, {"slides/main.tex": TWO_FRAMES})
    root = vault / "slides" / "main.tex"
    assert tex_client.post("/api/latex/build", json={"path": str(root)}).json()["ok"] is True

    def spot_of(line: int):
        return tex_client.post("/api/latex/synctex/view", json={"path": str(root), "file": str(root), "line": line}).json()

    def page_and_words(line: int) -> tuple[int, list[str]]:
        spot = spot_of(line)
        return spot["page"], spot["words"]

    # SyncTeX alone answers the frame before for a line of a frame body, which beamer reads as an argument
    assert page_and_words(5) == (1, ["Growth", "doubled", "this", "quarter"])
    assert page_and_words(8) == (2, ["measure", "weekly", "samples"])
    assert page_and_words(9) == (2, ["Then", "plot", "them"])
    assert page_and_words(10) == (2, [])


@pytest.mark.parametrize(("line", "words"), [
    ("We measure \\hl{weekly} samples. % plot later", ["measure", "weekly", "samples"]),
    ("\\item[\\textbf{Note}] \\textcolor{red}{Growth} beats \\emph{last} year", ["Growth", "beats", "last", "year"]),
    ("\\begin{frame}[fragile]{Results} see \\ref{tab:main}", ["Results"]),
    ("\\includegraphics[width=0.5\\textwidth]{figures/plot.png} where $area = width$", ["where"]),
    ("data data again, \\only<2>{again} 50\\% done", ["data", "again", "done"]),
    ("one two three four five six seven eight nine eleven twelve thirteen", ["three", "four", "five", "seven", "eight", "nine", "eleven", "twelve"]),
])
def test_a_line_typesets_its_plain_words_without_commands_options_or_comments(line, words):
    assert synctex.line_words(line) == words


def test_synctex_answers_404_before_a_build(tex_client, document, monkeypatch):
    monkeypatch.setattr(synctex.shutil, "which", lambda name: f"/usr/bin/{name}")
    response = tex_client.post("/api/latex/synctex", json={"path": str(document), "page": 1, "x": 10, "y": 10})
    assert response.status_code == 404


def test_synctex_answers_503_when_synctex_is_missing(tex_client, document, monkeypatch):
    monkeypatch.setattr(synctex.shutil, "which", lambda name: None)
    response = tex_client.post("/api/latex/synctex", json={"path": str(document), "page": 1, "x": 10, "y": 10})
    assert response.status_code == 503
    assert "synctex" in response.json()["detail"]



DECK = (
    "\\documentclass{beamer}\n"
    "\\AtBeginSection[]{\\begin{frame}\\sectionpage\\end{frame}}\n"
    "\\begin{document}\n"
    "\\begin{frame}{Agenda}\\tableofcontents\\end{frame}\n"
    "\\begin{frame}{Steps}\n"
    "First \\pause second\n"
    "\\end{frame}\n"
    "\n"
    "\\section{Chapter}\n"
    "% the first step\n"
    "\\subsection{Step}\n"
    "\n"
    "\\input{slides/more}\n"
    "\\end{document}\n"
)
MORE_SLIDES = "\\begin{frame}{More}\nBody\n\\end{frame}\n"


@needs_synctex
def test_the_slide_end_of_a_page_is_its_frames_end_or_the_headings_of_a_chapter_slide(tex_client, vault):
    write_files(vault, {"deck/main.tex": DECK, "deck/slides/more.tex": MORE_SLIDES})
    root = vault / "deck" / "main.tex"
    assert tex_client.post("/api/latex/build", json={"path": str(root)}).json()["ok"] is True

    def end_of(page: int):
        response = tex_client.post("/api/latex/synctex/slide", json={"path": str(root), "page": page})
        return response.json() if response.status_code == 200 else response.status_code

    assert end_of(1) == {"path": str(root), "line": 4}  # the agenda, not the `.toc` it reads
    assert end_of(2) == end_of(3) == {"path": str(root), "line": 7}  # both overlays of one frame
    assert end_of(4) == {"path": str(root), "line": 11}  # the chapter slide, past its subsection
    assert end_of(5) == {"path": str(vault / "deck" / "slides" / "more.tex"), "line": 3}
    assert end_of(6) == 404


def test_the_slide_end_answers_404_before_a_build(tex_client, document):
    assert tex_client.post("/api/latex/synctex/slide", json={"path": str(document), "page": 1}).status_code == 404


@pytest.mark.parametrize(("source", "line", "end"), [
    ("\\section{A}\n\\subsection{B}\n\n% note\n\\subsubsection{C}\nText\n", 1, 5),
    ("\\section{A}\n\\section{B}\n", 1, 1),
    ("\\subsection{A}\n\\section{B}\n\\subsection{C}\n", 1, 1),
    ("\\section{A}\nText\n\\subsection{B}\n", 1, 1),
    ("Text\n\\subsection{B}\n", 1, 1),
])
def test_a_chapter_slide_ends_after_the_deeper_headings_right_below_its_heading(source, line, end):
    assert synctex._after_headings(source.splitlines(), line) == end
