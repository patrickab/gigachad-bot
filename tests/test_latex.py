from pathlib import Path
import stat

import pytest

from backend.identity import get_request_identity
from backend.routes import latex as latex_route
from backend.routes.deps import get_file_roots
from lib import latex
from tests.conftest import mounted_roots, request_identity, route_client

PDFLATEX_FAILURE = r"""This is pdfTeX, Version 3.141592653-2.6-1.40.25 (TeX Live 2023) (preloaded format=pdflatex)
 restricted \write18 enabled.
entering extended mode
(./main.tex
LaTeX2e <2023-11-01> patch level 1
(/usr/share/texlive/texmf-dist/tex/latex/base/article.cls
Document Class: article 2023/05/17 v1.4n Standard LaTeX document class
) (./main.aux)
./main.tex:42: Undefined control sequence.
l.42 \sectoin
             {Intro}
The control sequence at the end of the top line
of your error message was never \def'ed.

./chapter.tex:7: Missing $ inserted.
<inserted text>
                $
l.7 a_b

! LaTeX Error: File `foo.sty' not found.

Type X to quit or <RETURN> to proceed,
or enter new name. (Default extension: sty)

Enter file name:
l.3 \usepackage{foo}

./main.tex:42: Undefined control sequence.
"""


def fake_latexmk(directory, monkeypatch, script: str) -> None:
    """Put a `latexmk` shell script first on PATH."""
    executable = directory / "latexmk"
    executable.write_text(f"#!/bin/sh\n{script}\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{directory}:/usr/bin:/bin")


def test_errors_come_from_file_line_messages_and_bare_bang_lines_with_their_context_line():
    errors = latex.parse_errors(PDFLATEX_FAILURE, Path("/work/paper/main.tex"))
    assert errors == [
        latex.LatexIssue(42, "Undefined control sequence.", "/work/paper/main.tex"),
        latex.LatexIssue(7, "Missing $ inserted.", "/work/paper/chapter.tex"),
        latex.LatexIssue(3, "LaTeX Error: File `foo.sty' not found."),
    ]


def test_errors_name_the_input_file_they_are_in_resolved_against_the_roots_folder():
    log = (
        "./sections/intro.tex:3: Undefined control sequence.\n"
        "l.3 \\foo\n"
        "../shared/macros.tex:9: Missing } inserted.\n"
        "/work/paper/sections/../refs.bib:1: Illegal, another \\bibdata command.\n"
    )
    assert latex.parse_errors(log, Path("/work/paper/main.tex")) == [
        latex.LatexIssue(3, "Undefined control sequence.", "/work/paper/sections/intro.tex"),
        latex.LatexIssue(None, "macros.tex:9: Missing } inserted."),
        latex.LatexIssue(1, "Illegal, another \\bibdata command.", "/work/paper/refs.bib"),
    ]


@pytest.mark.parametrize(
    ("head", "flag"),
    [
        ("% !TEX program = xelatex\n", "-pdfxe"),
        ("%!TEX program=LuaLaTeX\n", "-pdflua"),
        ("% !TEX program = pdflatex\n", "-pdf"),
        ("\\documentclass{article}\n", "-pdf"),
        ("\n" * 5 + "% !TEX program = xelatex\n", "-pdf"),
    ],
)
def test_engine_comes_from_the_magic_comment_in_the_first_lines(head, flag):
    assert latex.engine_flag(head) == flag


def test_paths_outside_the_mounts_and_non_tex_files_are_refused(tex_client, tmp_path, document):
    outside = tmp_path / "outside.tex"
    outside.write_text("x", encoding="utf-8")
    notes = document.parent / "notes.md"
    notes.write_text("x", encoding="utf-8")

    assert tex_client.post("/api/latex/build", json={"path": str(outside)}).status_code == 403
    assert tex_client.post("/api/latex/build", json={"path": str(notes)}).status_code == 400
    assert tex_client.post("/api/latex/build", json={"path": str(document.parent / "missing.tex")}).status_code == 404


def test_logins_without_host_access_are_refused(tmp_path, document):
    client = route_client(
        latex_route.router,
        overrides={get_request_identity: request_identity(False), get_file_roots: mounted_roots(tmp_path / "vault")},
    )
    assert client.post("/api/latex/build", json={"path": str(document)}).status_code == 403


def test_build_answers_503_when_latexmk_is_missing(tex_client, document, monkeypatch):
    monkeypatch.setattr(latex.shutil, "which", lambda name: None)
    response = tex_client.post("/api/latex/build", json={"path": str(document)})
    assert response.status_code == 503
    assert "latexmk" in response.json()["detail"]


def test_failed_build_reports_errors_and_keeps_serving_the_last_good_pdf(tex_client, document, tmp_path, monkeypatch):
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    failure = tmp_path / "failure.log"
    failure.write_text(PDFLATEX_FAILURE, encoding="utf-8")
    # TeX truncates its output PDF while it runs, the viewer must keep reading the last good one
    fake_latexmk(bin_directory, monkeypatch, f': > .latex-build/main.pdf\ncat "{failure}"\nexit 12')
    previous = document.parent / ".latex-build" / "main.view.pdf"
    previous.parent.mkdir()
    previous.write_bytes(b"%PDF-old")

    body = tex_client.post("/api/latex/build", json={"path": str(document)}).json()

    assert body["ok"] is False
    assert body["pdf_path"] == str(previous)
    assert previous.read_bytes() == b"%PDF-old"
    assert body["errors"][0] == {"line": 42, "message": "Undefined control sequence.", "file": str(document)}
    assert body["log_tail"].splitlines()[-1] == "./main.tex:42: Undefined control sequence."
    assert len(body["log_tail"].splitlines()) <= 40


def test_build_runs_latexmk_in_the_files_directory_with_the_engine_from_the_magic_comment(tex_client, document, tmp_path, monkeypatch):
    document.write_text("% !TEX program = xelatex\n\\documentclass{article}\n", encoding="utf-8")
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    invocation = tmp_path / "invocation"
    fake_latexmk(
        bin_directory,
        monkeypatch,
        f'pwd > "{invocation}"\necho "$@" >> "{invocation}"\nmkdir -p .latex-build && echo pdf > .latex-build/main.pdf',
    )

    body = tex_client.post("/api/latex/build", json={"path": str(document)}).json()

    assert body == {
        "ok": True,
        "pdf_path": str(document.parent / ".latex-build" / "main.view.pdf"),
        "errors": [],
        "log_tail": "",
        "duration_ms": body["duration_ms"],
    }
    directory, arguments = invocation.read_text(encoding="utf-8").splitlines()
    assert directory == str(document.parent)
    assert arguments == (
        f"-g -pdfxe -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1 -outdir={document.parent}/.latex-build main.tex"
    )



def test_errors_raised_inside_a_package_are_reported_with_the_package_file():
    log = "/usr/share/texmf-dist/tex/generic/babel/babel.sty:4330: Package babel Error: Unknown option 'ngerman'.\n"
    assert latex.parse_errors(log, Path("/work/paper/main.tex")) == [
        latex.LatexIssue(None, "babel.sty:4330: Package babel Error: Unknown option 'ngerman'."),
    ]


def test_discarding_a_build_removes_the_outputs_of_that_stem_only(tmp_path):
    for name in ("paper.tex", "paper.v2.tex", "other.tex"):
        (tmp_path / name).write_text("", encoding="utf-8")
    outputs = tmp_path / latex.BUILD_DIRECTORY
    outputs.mkdir()
    for name in ("paper.pdf", "paper.view.pdf", "paper.aux", "paper.v2.pdf", "paper.v2.view.pdf", "other.pdf", "papers.pdf"):
        (outputs / name).write_bytes(b"x")

    latex.discard_build(tmp_path / "paper.tex")

    assert sorted(path.name for path in outputs.iterdir()) == ["other.pdf", "paper.v2.pdf", "paper.v2.view.pdf", "papers.pdf"]


def test_discarding_a_build_of_a_folder_without_one_does_nothing(tmp_path):
    latex.discard_build(tmp_path / "paper.tex")

    assert not (tmp_path / latex.BUILD_DIRECTORY).exists()


ROOT_SOURCE = "\\documentclass{article}\n\\begin{document}\n\\end{document}\n"
BODY_SOURCE = "\\section{Intro}\n"


def write_files(base: Path, files: dict[str, str]) -> None:
    for name, content in files.items():
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def project(base: Path, name: str, ceiling: Path | None = None) -> latex.TexProject:
    return latex.project_of(base / name, ceiling=ceiling or base)


def test_root_resolution_prefers_the_magic_comment_then_the_file_itself_then_main_tex(tmp_path):
    write_files(tmp_path, {
        "main.tex": ROOT_SOURCE,
        "other.tex": ROOT_SOURCE,
        "pointing.tex": "% !TEX root = other.tex\n" + ROOT_SOURCE,
        "chapter.tex": BODY_SOURCE,
        "solo/paper.tex": ROOT_SOURCE,
        "solo/chapter.tex": BODY_SOURCE,
    })

    assert project(tmp_path, "pointing.tex") == latex.TexProject(tmp_path, tmp_path / "other.tex", [])
    assert project(tmp_path, "other.tex").root == tmp_path / "other.tex"
    assert project(tmp_path, "chapter.tex").root == tmp_path / "main.tex"
    assert project(tmp_path, "solo/chapter.tex").root == tmp_path / "solo" / "paper.tex"


@pytest.mark.parametrize(
    ("target", "expected"),
    [("../main.tex", "main.tex"), ("../nothing.tex", "sections/only.tex")],
)
def test_a_magic_comment_names_the_root_in_another_folder_and_a_missing_one_is_skipped(tmp_path, target, expected):
    write_files(tmp_path, {
        "main.tex": ROOT_SOURCE,
        "sections/x.tex": f"% !TEX root = {target}\n" + BODY_SOURCE,
        "sections/only.tex": ROOT_SOURCE,
    })

    assert project(tmp_path, "sections/x.tex").root == tmp_path / expected


@pytest.mark.parametrize(
    "source",
    [
        "% \\documentclass{article}\n\\begin{document}\n",
        "\\begin{document}\n\\documentclass{article}\n",
        "\\section{Intro}\n",
        "",
    ],
)
def test_only_a_documentclass_before_the_document_body_outside_comments_makes_a_root_document(tmp_path, source):
    write_files(tmp_path, {"fake.tex": source})

    assert project(tmp_path, "fake.tex").root is None


def test_candidates_reach_two_folders_up_and_never_above_the_ceiling(tmp_path):
    write_files(tmp_path, {
        "top.tex": ROOT_SOURCE,
        "a/two.tex": ROOT_SOURCE,
        "a/b/one.tex": ROOT_SOURCE,
        "a/b/c/chapter.tex": BODY_SOURCE,
        "flat/a.tex": ROOT_SOURCE,
        "flat/b.tex": ROOT_SOURCE,
        "flat/chapter.tex": BODY_SOURCE,
    })
    a, b, flat = tmp_path / "a", tmp_path / "a" / "b", tmp_path / "flat"

    assert project(tmp_path, "a/b/c/chapter.tex").candidates == [b / "one.tex", a / "two.tex"]
    assert project(tmp_path, "a/b/c/chapter.tex", ceiling=b).candidates == [b / "one.tex"]
    assert project(tmp_path, "a/b/c/chapter.tex", ceiling=b / "c").candidates == []
    assert project(tmp_path, "flat/chapter.tex", ceiling=flat) == latex.TexProject(flat, None, [flat / "a.tex", flat / "b.tex"])


def test_a_magic_comment_pointing_above_the_ceiling_is_ignored(tmp_path):
    write_files(tmp_path, {"main.tex": ROOT_SOURCE, "mount/chapter.tex": "% !TEX root = ../main.tex\n" + BODY_SOURCE})

    assert project(tmp_path, "mount/chapter.tex", ceiling=tmp_path / "mount") == latex.TexProject(tmp_path / "mount", None, [])


def test_non_tex_files_take_the_root_of_the_nearest_folder_that_has_one_and_have_no_candidates(tmp_path):
    write_files(tmp_path, {
        "main.tex": ROOT_SOURCE,
        "style.sty": ROOT_SOURCE,
        "paper/main.tex": ROOT_SOURCE,
        "paper/sections/deep/x.sty": "",
        "loose/notes.bib": "",
    })

    assert project(tmp_path, "style.sty") == latex.TexProject(tmp_path, tmp_path / "main.tex", [])
    assert project(tmp_path, "paper/sections/deep/x.sty").root == tmp_path / "paper" / "main.tex"
    (tmp_path / "main.tex").unlink()
    assert project(tmp_path, "loose/notes.bib") == latex.TexProject(tmp_path / "loose", None, [])


def test_setting_the_root_replaces_an_existing_magic_root_line_and_keeps_the_rest(tmp_path):
    write_files(tmp_path, {
        "main.tex": ROOT_SOURCE,
        "sections/chapter.tex": "% !TEX program = xelatex\n% !TEX Root = old.tex\n\\section{Intro}\r\ntext\n",
        "sections/bare.tex": "\\section{Intro}\n",
    })

    latex.set_root(tmp_path / "sections" / "chapter.tex", tmp_path / "main.tex")
    latex.set_root(tmp_path / "sections" / "bare.tex", tmp_path / "main.tex")

    assert (tmp_path / "sections" / "chapter.tex").read_bytes() == (
        b"% !TEX root = ../main.tex\n% !TEX program = xelatex\n\\section{Intro}\r\ntext\n"
    )
    assert (tmp_path / "sections" / "bare.tex").read_text(encoding="utf-8") == "% !TEX root = ../main.tex\n\\section{Intro}\n"
    assert project(tmp_path, "sections/chapter.tex").root == tmp_path / "main.tex"


def test_new_files_point_at_the_root_or_start_a_document_or_stay_empty(tmp_path):
    assert latex.new_file_content(tmp_path / "sections" / "new.tex", tmp_path / "main.tex") == "% !TEX root = ../main.tex\n\n"
    assert latex.new_file_content(tmp_path / "new.tex", tmp_path / "main.tex") == "% !TEX root = main.tex\n\n"
    assert latex.new_file_content(tmp_path / "new.tex", None) == latex.NEW_DOCUMENT
    assert latex.new_file_content(tmp_path / "refs.bib", tmp_path / "main.tex") == ""


def test_project_route_answers_absolute_paths_and_null_root(tex_client, vault, tmp_path):
    write_files(vault, {"paper/a.tex": ROOT_SOURCE, "paper/b.tex": ROOT_SOURCE, "paper/chapter.tex": BODY_SOURCE})
    paper = vault / "paper"

    body = tex_client.get("/api/latex/project", params={"path": str(paper / "chapter.tex")}).json()

    assert body == {"folder": str(paper), "root": None, "candidates": [str(paper / "a.tex"), str(paper / "b.tex")]}
    assert tex_client.get("/api/latex/project", params={"path": str(tmp_path / "x.tex")}).status_code == 403
    assert tex_client.get("/api/latex/project", params={"path": str(paper / "none.tex")}).status_code == 404


def test_choosing_a_root_writes_the_magic_comment_and_records_the_change(tex_client, vault, scope):
    write_files(vault, {"paper/a.tex": ROOT_SOURCE, "paper/b.tex": ROOT_SOURCE, "paper/chapter.tex": BODY_SOURCE})
    paper = vault / "paper"

    response = tex_client.post("/api/latex/root", json={"path": str(paper / "chapter.tex"), "root": str(paper / "b.tex")})

    assert response.json() == {"folder": str(paper), "root": str(paper / "b.tex"), "candidates": []}
    assert scope.changes == [{"kind": "document", "key": str(paper / "chapter.tex"), "operation": "write"}]


def test_choosing_a_root_refuses_files_that_are_not_tex_or_not_root_documents(tex_client, vault, scope):
    write_files(vault, {"paper/a.tex": ROOT_SOURCE, "paper/chapter.tex": BODY_SOURCE, "paper/refs.bib": "", "paper/other.tex": BODY_SOURCE})
    paper = vault / "paper"

    def choose(path: str, root: str) -> int:
        return tex_client.post("/api/latex/root", json={"path": str(paper / path), "root": str(paper / root)}).status_code

    assert choose("refs.bib", "a.tex") == 400
    assert choose("chapter.tex", "refs.bib") == 400
    assert choose("chapter.tex", "other.tex") == 400
    assert choose("chapter.tex", "gone.tex") == 404
    assert (paper / "chapter.tex").read_text(encoding="utf-8") == BODY_SOURCE
    assert scope.changes == []


def test_new_files_start_with_the_folders_root_and_record_a_change(tex_client, vault, scope):
    write_files(vault, {"paper/main.tex": ROOT_SOURCE, "bare/readme.md": ""})
    paper = vault / "paper"

    created = [tex_client.post("/api/latex/file", json={"folder": str(paper), "name": name}) for name in ("intro.tex", "refs.bib")]
    tex_client.post("/api/latex/file", json={"folder": str(vault / "bare"), "name": "first.tex"})

    assert created[0].json() == {"path": str(paper / "intro.tex")}
    assert (paper / "intro.tex").read_text(encoding="utf-8") == "% !TEX root = main.tex\n\n"
    assert (paper / "refs.bib").read_text(encoding="utf-8") == ""
    assert (vault / "bare" / "first.tex").read_text(encoding="utf-8") == latex.NEW_DOCUMENT
    assert scope.changes == [
        {"kind": "document", "key": str(path), "operation": "write"}
        for path in (paper / "intro.tex", paper / "refs.bib", vault / "bare" / "first.tex")
    ]


@pytest.mark.parametrize("name", ["", "../escape.tex", ".hidden.tex", "notes.md"])
def test_new_file_names_must_be_one_visible_tex_bib_sty_or_cls_segment(tex_client, vault, scope, name):
    write_files(vault, {"paper/main.tex": ROOT_SOURCE})

    response = tex_client.post("/api/latex/file", json={"folder": str(vault / "paper"), "name": name})

    assert response.status_code == 400
    assert sorted(path.name for path in (vault / "paper").iterdir()) == ["main.tex"]
    assert not (vault / "escape.tex").exists()
    assert scope.changes == []


def test_creating_an_existing_file_answers_409_and_keeps_it(tex_client, vault, scope):
    write_files(vault, {"paper/main.tex": ROOT_SOURCE})

    response = tex_client.post("/api/latex/file", json={"folder": str(vault / "paper"), "name": "main.tex"})

    assert response.status_code == 409
    assert (vault / "paper" / "main.tex").read_text(encoding="utf-8") == ROOT_SOURCE
    assert scope.changes == []


@pytest.mark.parametrize(("folder", "status"), [("", 403), ("gone", 400)])
def test_creating_a_file_needs_an_existing_folder_inside_the_mounts(tex_client, vault, tmp_path, folder, status):
    write_files(vault, {"paper/main.tex": ROOT_SOURCE})
    target = tmp_path if folder == "" else vault / folder

    assert tex_client.post("/api/latex/file", json={"folder": str(target), "name": "a.tex"}).status_code == status
    assert not (tmp_path / "a.tex").exists()


BEGIN = b"\\begin{document}\n\\end{document}\n"


@pytest.mark.parametrize(("preamble", "inserted"), [
    (b"\\documentclass{article}\n", b"\\documentclass{article}\n\\usepackage{graphicx}\n"),
    (b"\\documentclass[a4paper,\n 11pt]{article}\n", b"\\documentclass[a4paper,\n 11pt]{article}\n\\usepackage{graphicx}\n"),
    (b"\\documentclass{article}\n% \\usepackage{graphicx}\n", b"\\documentclass{article}\n% \\usepackage{graphicx}\n\\usepackage{graphicx}\n"),
    (b"\\documentclass{article}\n\\usepackage{my-graphicx}\n", b"\\documentclass{article}\n\\usepackage{my-graphicx}\n\\usepackage{graphicx}\n"),
    (b"\\documentclass{article}\r\n", b"\\documentclass{article}\r\n\\usepackage{graphicx}\r\n"),
    (b"\\documentclass{article}\n\\usepackage[draft]{xcolor,graphicx}\n", None),
    (b"\\documentclass{article}\n\\usepackage[\n draft]{xcolor,\n graphicx}\n", None),
    (b"\\documentclass{article}\n\\RequirePackage{graphicx}\n", None),
])
def test_a_root_gets_a_package_before_begin_document_only_when_it_does_not_load_it(tmp_path, preamble, inserted):
    root = tmp_path / "main.tex"
    body = BEGIN.replace(b"\n", b"\r\n") if b"\r\n" in preamble else BEGIN
    root.write_bytes(preamble + body)

    wrote = latex.ensure_package(root, "graphicx")

    assert wrote == (inserted is not None)
    assert root.read_bytes() == (inserted or preamble) + body


def test_a_root_that_is_not_utf8_is_left_alone(tmp_path):
    root = tmp_path / "main.tex"
    source = "\\documentclass{article}\n% caf\xe9\n".encode("latin-1") + BEGIN
    root.write_bytes(source)

    assert latex.ensure_package(root, "graphicx") is False
    assert root.read_bytes() == source
