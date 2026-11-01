r"""LaTeX builds with `latexmk` and tex projects.

- `build` compiles one file into `<directory>/.latex-build`, one build at a time per file, without shell escape
- `discard_build` deletes the outputs of a file that was renamed or replaced, so a build never shows or reuses another file's PDF
- `project_of` finds the root document of any file of a tex project, `set_root` writes the `% !TEX root` comment that picks it
- `is_build_output` tells the files that are not sources
- `ensure_package` loads a package a pasted reference needs (`graphicx` for `\includegraphics`)
- `NEW_DOCUMENT` is the content of a new `.tex` file
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import signal
import time

BUILD_DIRECTORY = ".latex-build"
_AUX_SUFFIXES = (
    ".aux", ".log", ".synctex.gz", ".fls", ".fdb_latexmk", ".out", ".toc", ".bbl", ".blg", ".bcf", ".run.xml",
)
# A first section, an empty body builds no PDF ("No pages of output")
NEW_DOCUMENT = """\
\\documentclass[11pt]{article}
\\usepackage{graphicx}

\\begin{document}

\\section{Introduction}

\\end{document}
"""

_BUILD_TIMEOUT_SECONDS = 120
_LOG_TAIL_LINES = 40
MAGIC_COMMENT_LINES = 5
_ANCESTOR_LEVELS = 2
ROOT_COMMENT = re.compile(r"^(?P<head>\s*%\s*!TEX\s+root\s*=\s*)(?P<root>\S.*?)(?P<tail>\s*)$", re.IGNORECASE)
COMMENT = re.compile(r"(?<!\\)%.*")
_PACKAGE_LOAD = re.compile(r"\\(?:usepackage|RequirePackage)\s*(?:\[[^\]]*\])?\s*\{(?P<names>[^}]*)\}")
_ENGINE_FLAGS = {"pdflatex": "-pdf", "xelatex": "-pdfxe", "lualatex": "-pdflua"}
_ENGINE_COMMENT = re.compile(r"^\s*%\s*!TEX\s+program\s*=\s*(\w+)", re.IGNORECASE)
_FILE_LINE_ERROR = re.compile(r"^(?P<file>[^\s:][^:\n]*\.\w+):(?P<line>\d+): (?P<message>.+)$")
_BARE_ERROR = re.compile(r"^! (?P<message>.+)$")
_ERROR_LINE = re.compile(r"^l\.(?P<line>\d+)\b")
_BARE_ERROR_LOOKAHEAD = 10

_build_locks: dict[Path, asyncio.Lock] = {}


class LatexUnavailable(RuntimeError):
    """`latexmk` is not installed."""


@dataclass(frozen=True)
class LatexIssue:
    """One TeX error, `line` is None when the log does not name a line of a project file, `file` None when it does not name the file."""

    line: int | None
    message: str
    file: str | None = None


@dataclass(frozen=True)
class TexProject:
    """A tex project: its *folder*, its *root* document (None when unknown) and, without one, the root documents to choose from."""

    folder: Path
    root: Path | None
    candidates: list[Path]


@dataclass(frozen=True)
class LatexBuild:
    """Outcome of one build, `pdf_path` is the PDF of the last successful build (possibly an earlier one) or None."""

    ok: bool
    pdf_path: str | None
    errors: list[LatexIssue] = field(default_factory=list)
    log_tail: str = ""
    duration_ms: int = 0


def is_latexmk_installed() -> bool:
    return shutil.which("latexmk") is not None


def engine_flag(source: str) -> str:
    """The latexmk flag for the `% !TEX program = ...` magic comment in the first lines of *source*, `-pdf` without one."""
    for line in source.splitlines()[:MAGIC_COMMENT_LINES]:
        if match := _ENGINE_COMMENT.match(line):
            return _ENGINE_FLAGS.get(match.group(1).lower(), "-pdf")
    return "-pdf"


def parse_errors(output: str, source: Path) -> list[LatexIssue]:
    """TeX errors from latexmk or `.log` text: `file:line: message` lines, else `! message` lines with their `l.N` line.

    A `file:line` error names the absolute path of its file when that lies in the folder of the built *source*,
    else (a package) the file and line move into the message and the line is None.
    """
    lines = output.splitlines()
    errors: list[LatexIssue] = []
    for index, text in enumerate(lines):
        if match := _FILE_LINE_ERROR.match(text):
            issue = _issue_in(match["file"], int(match["line"]), match["message"], source)
        elif match := _BARE_ERROR.match(text):
            issue = LatexIssue(_context_line(lines[index + 1 : index + 1 + _BARE_ERROR_LOOKAHEAD]), match["message"])
        else:
            continue
        if issue not in errors:
            errors.append(issue)
    return errors


def _issue_in(file: str, line: int, message: str, source: Path) -> LatexIssue:
    path = Path(os.path.normpath(source.parent / file))
    if path.is_relative_to(source.parent):
        return LatexIssue(line, message, str(path))
    return LatexIssue(None, f"{path.name}:{line}: {message}")


def _context_line(following: list[str]) -> int | None:
    for text in following:
        if match := _ERROR_LINE.match(text):
            return int(match["line"])
    return None


def _tail(output: str) -> str:
    return "\n".join(output.splitlines()[-_LOG_TAIL_LINES:])


def _command(source: Path, output_directory: Path, engine: str) -> list[str]:
    return [
        "latexmk",
        "-g",  # always run TeX: latexmk otherwise replays a cached failure without its errors, even after a package install
        engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "-synctex=1",
        f"-outdir={output_directory}",
        source.name,
    ]


def _kill_group(process: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)


async def _run_latexmk(command: list[str], directory: Path) -> tuple[int | None, str]:
    """Run *command*, returning (exit code, combined output), or (None, output so far) after killing it on timeout."""
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=directory,
        env={**os.environ, "max_print_line": "10000"},  # TeX wraps log lines at 79 characters, cutting error messages
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,  # latexmk spawns the TeX engine, a timeout must stop the whole group
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), _BUILD_TIMEOUT_SECONDS)
    except TimeoutError:
        _kill_group(process)
        output, _ = await process.communicate()
        return None, output.decode("utf-8", errors="replace")
    except asyncio.CancelledError:
        _kill_group(process)
        raise
    return process.returncode, output.decode("utf-8", errors="replace")


async def build(source: Path) -> LatexBuild:
    """Compile *source* with latexmk, waiting for a build of the same file already running.

    Raises `LatexUnavailable` when `latexmk` is missing. TeX errors and timeouts are reported in the result.
    """
    if not is_latexmk_installed():
        raise LatexUnavailable("latexmk is not installed on the server, install a TeX distribution such as TeX Live")
    async with _build_locks.setdefault(source, asyncio.Lock()):
        started = time.monotonic()
        output_directory = source.parent / BUILD_DIRECTORY
        engine = engine_flag(await asyncio.to_thread(_read_head, source))
        code, output = await _run_latexmk(_command(source, output_directory, engine), source.parent)
        if code == 0:
            await asyncio.to_thread(_publish, output_directory / f"{source.stem}.pdf")
        pdf = _published(output_directory / f"{source.stem}.pdf")
        duration_ms = round((time.monotonic() - started) * 1000)
    if code is None:
        errors = [LatexIssue(None, f"The build was stopped after {_BUILD_TIMEOUT_SECONDS} seconds")]
    else:
        log = output_directory / f"{source.stem}.log"
        errors = [] if code == 0 else (
            parse_errors(await asyncio.to_thread(_read_log, log), source) or parse_errors(output, source)
            or [LatexIssue(None, f"latexmk exited with code {code}")]
        )
    return LatexBuild(
        ok=code == 0,
        pdf_path=str(pdf) if pdf.is_file() else None,
        errors=errors,
        log_tail="" if code == 0 else _tail(output),
        duration_ms=duration_ms,
    )


def _published(pdf: Path) -> Path:
    """The copy of *pdf* the viewer reads: TeX rewrites *pdf* in place while it builds."""
    return pdf.with_suffix(".view.pdf")


def _publish(pdf: Path) -> None:
    """Replace the published copy of *pdf* in one step, so a reader never sees a half-written file."""
    if not pdf.is_file():
        return
    staging = pdf.with_suffix(".view.tmp")
    shutil.copyfile(pdf, staging)
    staging.replace(_published(pdf))


def _read_log(log: Path) -> str:
    return log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""


def discard_build(source: Path) -> None:
    """Delete `<directory>/.latex-build/<stem>.*` of *source*, leaving the outputs of a sibling whose stem extends it (`a.b` for `a`)."""
    outputs = source.parent / BUILD_DIRECTORY
    if not outputs.is_dir():
        return
    longer = [f"{sibling.stem}." for sibling in source.parent.glob("*.tex") if sibling.stem.startswith(f"{source.stem}.")]
    for output in outputs.iterdir():
        if output.is_file() and output.name.startswith(f"{source.stem}.") and not output.name.startswith(tuple(longer)):
            output.unlink()


def is_build_output(path: Path) -> bool:
    """Whether *path* is LaTeX build output: inside `.latex-build/`, a `*.view.pdf`, or an aux file of a sibling `.tex`."""
    if BUILD_DIRECTORY in path.parts or path.name.endswith(".view.pdf"):
        return True
    suffix = next((suffix for suffix in _AUX_SUFFIXES if path.name.endswith(suffix)), None)
    return suffix is not None and path.with_name(f"{path.name.removesuffix(suffix)}.tex").is_file()


def _read_head(source: Path) -> str:
    with source.open(encoding="utf-8", errors="replace") as file:
        return "".join(file.readline() for _ in range(MAGIC_COMMENT_LINES))


def ensure_package(root: Path, package: str) -> bool:
    r"""Add `\usepackage{package}` on its own line before the `\begin{document}` line of *root* unless it loads it already. Returns whether it wrote.

    Loads are looked up in *root* alone (`\usepackage`/`\RequirePackage`, comments ignored, options and lists across lines): a package
    that only an `\input` preamble or a `.sty` loads is added again. The file keeps its line endings; an unreadable, non-UTF-8 or
    unwritable root answers False. The read-modify-write is not locked against a concurrent save.
    """
    try:
        text = root.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    lines = text.splitlines(keepends=True)
    code = "".join(COMMENT.sub("", line) for line in lines)
    for loaded in _PACKAGE_LOAD.finditer(code):
        if package in (name.strip() for name in loaded.group("names").split(",")):
            return False
    begun = next((index for index, line in enumerate(lines) if r"\begin{document}" in COMMENT.sub("", line)), None)
    if begun is None:
        return False
    newline = "\r\n" if "\r\n" in text else "\n"
    lines.insert(begun, f"\\usepackage{{{package}}}{newline}")
    try:
        root.write_bytes("".join(lines).encode("utf-8"))
    except OSError:
        return False
    return True


def is_root_document(path: Path) -> bool:
    r"""Whether the `.tex` file *path* has a `\documentclass` before its `\begin{document}`, comments ignored."""
    try:
        with path.open(encoding="utf-8", errors="replace") as file:
            for line in file:
                code = COMMENT.sub("", line)
                declared, begun = code.find(r"\documentclass"), code.find(r"\begin{document}")
                if declared >= 0 and (begun < 0 or declared < begun):
                    return True
                if begun >= 0:
                    return False
    except OSError:
        pass
    return False


def _roots_in(folder: Path) -> list[Path]:
    """The `.tex` files of *folder* that are root documents, by name."""
    return [
        path for path in sorted(folder.glob("*.tex")) if not path.name.startswith(".") and path.is_file() and is_root_document(path)
    ]


def _default_root(folder: Path) -> Path | None:
    """`main.tex` of *folder*, else its only root document."""
    main = folder / "main.tex"
    if main.is_file():
        return main
    roots = _roots_in(folder)
    return roots[0] if len(roots) == 1 else None


def _magic_root(path: Path, ceiling: Path) -> Path | None:
    """The existing `.tex` file *path* names in a `% !TEX root = ...` line of its first lines, when inside *ceiling*."""
    try:
        head = _read_head(path)
    except OSError:
        return None
    for line in head.splitlines():
        if match := ROOT_COMMENT.match(line):
            target = Path(os.path.normpath(path.parent / match["root"]))
            return target if target.is_relative_to(ceiling) and target.suffix == ".tex" and target.is_file() else None
    return None


def project_of(path: Path, *, ceiling: Path) -> TexProject:
    """The tex project of *path*, reading only inside *ceiling*.

    A `.tex` file's root is the one its `% !TEX root` comment names, else itself when it is a root document, else the
    default root of its folder (`main.tex`, else its only root document). Without a root, `candidates` lists the root
    documents of its folder and the two folders above, never above *ceiling*. Other files take the default root of the
    nearest of those folders that has one and have no candidates.
    """
    if not path.parent.is_relative_to(ceiling):
        return TexProject(path.parent, None, [])
    levels = [path.parent, *path.parent.parents][: _ANCESTOR_LEVELS + 1]
    folders = [folder for folder in levels if folder.is_relative_to(ceiling)]
    candidates: list[Path] = []
    if path.suffix == ".tex":
        root = _magic_root(path, ceiling) or (path if is_root_document(path) else None) or _default_root(path.parent)
        if root is None:
            candidates = [candidate for folder in folders for candidate in _roots_in(folder)]
    else:
        root = next((found for folder in folders if (found := _default_root(folder))), None)
    return TexProject(root.parent if root else path.parent, root, candidates)


def set_root(path: Path, root: Path) -> None:
    """Make `% !TEX root = <root relative to the folder>` the first line of the `.tex` file *path*, replacing its current one."""
    lines = path.read_bytes().decode("utf-8").splitlines(keepends=True)
    head = [line for line in lines[:MAGIC_COMMENT_LINES] if not ROOT_COMMENT.match(line)]
    line = f"% !TEX root = {Path(os.path.relpath(root, path.parent)).as_posix()}\n"
    path.write_bytes("".join([line, *head, *lines[MAGIC_COMMENT_LINES:]]).encode("utf-8"))


def new_file_content(path: Path, root: Path | None) -> str:
    """What a new file at *path* starts with: a `.tex` file points at its *root*, or is a new document without one."""
    if path.suffix != ".tex":
        return ""
    return NEW_DOCUMENT if root is None else f"% !TEX root = {Path(os.path.relpath(root, path.parent)).as_posix()}\n\n"
