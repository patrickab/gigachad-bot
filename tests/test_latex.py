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


@pytest.fixture
def tex_client(tmp_path):
    """A host-access client whose only mount is `tmp_path / "vault"`."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return route_client(
        latex_route.router,
        overrides={get_request_identity: request_identity(), get_file_roots: mounted_roots(vault)},
    )


@pytest.fixture
def document(tmp_path):
    source = tmp_path / "vault" / "paper" / "main.tex"
    source.parent.mkdir(parents=True)
    source.write_text("\\documentclass{article}\n", encoding="utf-8")
    return source


def fake_latexmk(directory, monkeypatch, script: str) -> None:
    """Put a `latexmk` shell script first on PATH."""
    executable = directory / "latexmk"
    executable.write_text(f"#!/bin/sh\n{script}\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{directory}:/usr/bin:/bin")


def test_errors_come_from_file_line_messages_and_bare_bang_lines_with_their_context_line():
    errors = latex.parse_errors(PDFLATEX_FAILURE, "main.tex")
    assert errors == [
        latex.LatexIssue(42, "Undefined control sequence."),
        latex.LatexIssue(None, "chapter.tex:7: Missing $ inserted."),
        latex.LatexIssue(3, "LaTeX Error: File `foo.sty' not found."),
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
    assert body["errors"][0] == {"line": 42, "message": "Undefined control sequence."}
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
    assert latex.parse_errors(log, "main.tex") == [
        latex.LatexIssue(None, "babel.sty:4330: Package babel Error: Unknown option 'ngerman'."),
    ]
