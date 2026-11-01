"""Rewriting the references of a LaTeX project to a renamed file."""

from pathlib import Path

import pytest

from lib.latex_references import rewrite_references


def project(tmp_path, files: dict[str, str | bytes]):
    """A project folder holding *files* (relative name -> content), returned."""
    folder = tmp_path / "paper"
    for name, content in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return folder


def rename(folder, old: str, new: str) -> list[str]:
    """Rewrite the references to *old* (renamed to *new*), returning the rewritten files relative to *folder*."""
    return [path.relative_to(folder).as_posix() for path in rewrite_references(folder, folder / old, folder / new)]


def text(folder, name="main.tex") -> str:
    return (folder / name).read_bytes().decode("utf-8")


@pytest.mark.parametrize(
    ("before", "old", "new", "after"),
    [
        (r"\includegraphics{imgs/a.png}", "imgs/a.png", "imgs/b.png", r"\includegraphics{imgs/b.png}"),
        (r"\includegraphics[width=\linewidth]{imgs/a.png}", "imgs/a.png", "imgs/b.png", r"\includegraphics[width=\linewidth]{imgs/b.png}"),
        (r"\includegraphics*[width=3cm] {imgs/a.png}", "imgs/a.png", "imgs/b.png", r"\includegraphics*[width=3cm] {imgs/b.png}"),
        (r"\input{ch/one.tex}", "ch/one.tex", "ch/two.tex", r"\input{ch/two.tex}"),
        (r"\input{ch/one}", "ch/one.tex", "ch/two.tex", r"\input{ch/two}"),
        (r"\input{ch/a.b}", "ch/a.b.tex", "ch/c.tex", r"\input{ch/c}"),
        (r"\input{one}", "one.tex", "two.tex", r"\input{two}"),
        (r"\include{ch/one}", "ch/one.tex", "ch/two.tex", r"\include{ch/two}"),
        (r"\includeonly{ch/one,ch/other}", "ch/one.tex", "ch/two.tex", r"\includeonly{ch/two,ch/other}"),
        (r"\includeonly{ch/other,ch/one}", "ch/one.tex", "ch/two.tex", r"\includeonly{ch/other,ch/two}"),
        (r"\bibliography{refs}", "refs.bib", "literature.bib", r"\bibliography{literature}"),
        (r"\bibliography{other, refs ,more}", "refs.bib", "literature.bib", r"\bibliography{other, literature ,more}"),
        (r"\addbibresource{refs.bib}", "refs.bib", "literature.bib", r"\addbibresource{literature.bib}"),
        (r"\addbibresource[label=x]{refs.bib}", "refs.bib", "literature.bib", r"\addbibresource[label=x]{literature.bib}"),
        (r"\addbibresource{refs}", "refs.bib", "literature.bib", r"\addbibresource{refs}"),
        (r"\includegraphics{imgs/a}", "imgs/a.png", "imgs/b.png", r"\includegraphics{imgs/b}"),
        (r"\includegraphics{imgs/a}", "imgs/a.pdf", "imgs/b.pdf", r"\includegraphics{imgs/b}"),
        (r"\includegraphics{imgs/a.png}", "imgs/a.png", "imgs/b.svg", r"\includegraphics{imgs/b.svg}"),
        (r"\includegraphics{imgs/a}", "imgs/a.png", "imgs/b.jpeg", r"\includegraphics{imgs/a}"),
        (r"\includegraphics{imgs/a}", "imgs/a.png", "imgs/b.svg", r"\includegraphics{imgs/a}"),
        (r"\input{ch/one}", "ch/one.tex", "ch/one.txt", r"\input{ch/one}"),
    ],
)
def test_each_command_follows_the_renamed_file_in_its_extension_style(tmp_path, before, old, new, after):
    folder = project(tmp_path, {"main.tex": f"{before}\n"})

    rewrite_references(folder, folder / old, folder / new)

    assert text(folder) == f"{after}\n"


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (r"\input{./ch/one}", r"\input{./ch/two}"),
        (r"\includegraphics{./ch/one.tex}", r"\includegraphics{./ch/two.tex}"),
        (r"\bibliography{./ch/one,ch/one}", r"\bibliography{./ch/one,ch/one}"),
        (r"\includeonly{./ch/one,ch/one}", r"\includeonly{./ch/two,ch/two}"),
    ],
)
def test_a_dot_slash_prefix_stays(tmp_path, before, after):
    folder = project(tmp_path, {"main.tex": before})

    rewrite_references(folder, folder / "ch/one.tex", folder / "ch/two.tex")

    assert text(folder) == after


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("% \\input{ch/one}\n", "% \\input{ch/one}\n"),
        ("   %\\input{ch/one}\n", "   %\\input{ch/one}\n"),
        ("\\input{ch/one} % was \\input{ch/one}\n", "\\input{ch/two} % was \\input{ch/one}\n"),
        ("100\\% sure \\input{ch/one}\n", "100\\% sure \\input{ch/two}\n"),
        ("\\input{ch/one}%\\input{ch/one}\n", "\\input{ch/two}%\\input{ch/one}\n"),
    ],
)
def test_commented_text_is_left_alone(tmp_path, before, after):
    folder = project(tmp_path, {"main.tex": before})

    rewrite_references(folder, folder / "ch/one.tex", folder / "ch/two.tex")

    assert text(folder) == after


@pytest.mark.parametrize(
    ("line", "old", "new", "after"),
    [
        ("% !TEX root = ../main.tex", "main.tex", "paper.tex", "% !TEX root = ../paper.tex"),
        ("%!tex ROOT=../main.tex  ", "main.tex", "paper.tex", "%!tex ROOT=../paper.tex  "),
        ("% !TEX root = ./main.tex", "sub/main.tex", "sub/paper.tex", "% !TEX root = ./paper.tex"),
        ("% !TEX root = other.tex", "main.tex", "paper.tex", "% !TEX root = other.tex"),
    ],
)
def test_the_tex_root_line_is_rewritten_relative_to_its_own_folder(tmp_path, line, old, new, after):
    folder = project(tmp_path, {"sub/chapter.tex": f"{line}\nbody\n", "sub/main.tex": "", "main.tex": ""})

    rewrite_references(folder, folder / old, folder / new)

    assert text(folder, "sub/chapter.tex") == f"{after}\nbody\n"


def test_the_tex_root_line_counts_only_in_the_first_lines(tmp_path):
    late = "a\nb\nc\nd\ne\n% !TEX root = main.tex\n"
    folder = project(tmp_path, {"chapter.tex": late, "main.tex": ""})

    assert rename(folder, "main.tex", "paper.tex") == []

    assert text(folder, "chapter.tex") == late


def test_commands_resolve_against_the_project_folder_and_the_root_line_against_the_files_own(tmp_path):
    folder = project(
        tmp_path,
        {"sub/chapter.tex": "% !TEX root = ../main.tex\n\\input{other}\n\\input{sub/other}\n", "main.tex": "", "sub/other.tex": ""},
    )

    assert rename(folder, "sub/other.tex", "sub/renamed.tex") == ["sub/chapter.tex"]

    assert text(folder, "sub/chapter.tex") == "% !TEX root = ../main.tex\n\\input{other}\n\\input{sub/renamed}\n"


@pytest.mark.parametrize(
    "reference",
    [
        r"\input{ch/one2}",
        r"\input{xch/one}",
        r"\input{other/ch/one}",
        r"\input{ch/ONE}",
        r"\input{ch/one.tex.bak}",
        r"\input{ch/one.bib}",
        r"\inputminted{ch/one}",
        r"\includegraphics{ch/one}",
        r"\bibliography{ch/one}",
        r"\usepackage{ch/one}",
        r"\input{\chapters/one}",
        r"\input{one}",
        r"\input{}",
    ],
)
def test_only_exact_matches_change(tmp_path, reference):
    folder = project(tmp_path, {"main.tex": reference})

    assert rename(folder, "ch/one.tex", "ch/two.tex") == []

    assert text(folder) == reference


def test_an_extension_less_graphic_that_another_format_shadows_stays(tmp_path):
    reference = r"\includegraphics{imgs/a}"
    folder = project(tmp_path, {"main.tex": reference, "imgs/a.pdf": b"%PDF"})

    assert rename(folder, "imgs/a.png", "imgs/b.png") == []

    assert text(folder) == reference


def test_an_extension_less_graphic_follows_when_only_a_lower_format_exists(tmp_path):
    folder = project(tmp_path, {"main.tex": r"\includegraphics{imgs/a}", "imgs/a.jpg": b"jpg"})

    rename(folder, "imgs/a.png", "imgs/b.png")

    assert text(folder) == r"\includegraphics{imgs/b}"


@pytest.mark.parametrize(
    "content",
    [
        b"\\input{ch/one}\r\nline\r\n\\input{ch/one}\r\n",
        b"\\input{ch/one}\rold mac\r\\input{ch/one}",
        b"\\input{ch/one}\n\r\n\n\\input{ch/one}",
        b"\xef\xbb\xbf\\input{ch/one} caf\xc3\xa9 \xe2\x82\xac\n\x0c\n",
        b"\\input{ch/one}  \t\n\n\n",
    ],
)
def test_every_other_byte_stays(tmp_path, content):
    folder = project(tmp_path, {"main.tex": content})

    rewrite_references(folder, folder / "ch/one.tex", folder / "ch/two.tex")

    assert (folder / "main.tex").read_bytes() == content.replace(b"ch/one", b"ch/two")


def test_only_files_that_changed_are_written_and_returned(tmp_path):
    folder = project(
        tmp_path,
        {"main.tex": "\\input{ch/one}", "ch/two.tex": "text", "ch/three.tex": "\\input{ch/other}", "refs.bib": "@book{one}"},
    )
    untouched = {name: (folder / name).stat().st_mtime_ns for name in ("ch/two.tex", "ch/three.tex", "refs.bib")}

    assert rename(folder, "ch/one.tex", "ch/two.tex") == ["main.tex"]

    assert {name: (folder / name).stat().st_mtime_ns for name in untouched} == untouched


def test_tex_and_bib_files_in_any_subfolder_are_rewritten_in_order(tmp_path):
    folder = project(
        tmp_path,
        {"main.tex": r"\input{ch/one}", "a/deep/x.tex": r"\input{ch/one}", "z.bib": r"\input{ch/one}", "notes.md": r"\input{ch/one}"},
    )

    assert rename(folder, "ch/one.tex", "ch/two.tex") == ["main.tex", "z.bib", "a/deep/x.tex"]

    assert text(folder, "notes.md") == r"\input{ch/one}"


def test_build_output_git_and_dependency_folders_are_not_touched(tmp_path):
    content = r"\input{ch/one}"
    folder = project(
        tmp_path,
        {
            "main.tex": content,
            ".latex-build/main.tex": content,
            ".git/hooks/x.tex": content,
            "node_modules/pkg/x.tex": content,
            "main.aux": content,
        },
    )

    assert rename(folder, "ch/one.tex", "ch/two.tex") == ["main.tex"]

    assert [text(folder, name) for name in (".latex-build/main.tex", ".git/hooks/x.tex", "node_modules/pkg/x.tex", "main.aux")] == [content] * 4


def test_a_file_that_is_not_utf8_is_skipped_and_the_others_still_follow(tmp_path):
    latin = b"\\input{ch/one} caf\xe9"
    folder = project(tmp_path, {"a.bib": latin, "main.tex": r"\input{ch/one}"})

    assert rename(folder, "ch/one.tex", "ch/two.tex") == ["main.tex"]

    assert (folder / "a.bib").read_bytes() == latin
    assert text(folder) == r"\input{ch/two}"


def test_an_extension_less_graphic_names_the_new_file_in_full_when_another_format_would_shadow_it(tmp_path):
    folder = project(tmp_path, {"main.tex": r"\includegraphics{imgs/a}", "imgs/b.pdf": b"%PDF"})

    rename(folder, "imgs/a.png", "imgs/b.png")

    assert text(folder) == r"\includegraphics{imgs/b.png}"


@pytest.mark.parametrize("old", ["notes.md", "script.py", "imgs/a.svg", "ch"])
def test_a_renamed_file_no_command_can_point_at_rewrites_nothing(tmp_path, old):
    content = f"\\input{{{Path(old).stem}}}\n"
    folder = project(tmp_path, {"main.tex": content})

    assert rename(folder, old, "renamed.txt") == []

    assert text(folder) == content


def test_dot_folders_are_not_searched(tmp_path):
    content = r"\input{ch/one}"
    folder = project(tmp_path, {"main.tex": content, ".venv/lib/x.tex": content, ".hidden/x.tex": content})

    assert rename(folder, "ch/one.tex", "ch/two.tex") == ["main.tex"]

    assert [text(folder, name) for name in (".venv/lib/x.tex", ".hidden/x.tex")] == [content] * 2
