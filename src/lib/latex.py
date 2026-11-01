"""LaTeX builds with `latexmk`.

- `build` compiles one file into `<directory>/.latex-build`, one build at a time per file, without shell escape
- `discard_build` deletes the outputs of a file that was renamed or replaced, so a build never shows or reuses another file's PDF
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
# A first section, an empty body builds no PDF ("No pages of output")
NEW_DOCUMENT = """\
\\documentclass[11pt]{article}

\\begin{document}

\\section{Introduction}

\\end{document}
"""

_BUILD_TIMEOUT_SECONDS = 120
_LOG_TAIL_LINES = 40
_MAGIC_COMMENT_LINES = 5
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
    """One TeX error, `line` is None when the log does not name a line of the built file."""

    line: int | None
    message: str


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
    for line in source.splitlines()[:_MAGIC_COMMENT_LINES]:
        if match := _ENGINE_COMMENT.match(line):
            return _ENGINE_FLAGS.get(match.group(1).lower(), "-pdf")
    return "-pdf"


def parse_errors(output: str, document_name: str) -> list[LatexIssue]:
    """TeX errors from latexmk or `.log` text: `file:line: message` lines, else `! message` lines with their `l.N` line.

    Errors in other files than *document_name* carry the file and line in the message and no line.
    """
    lines = output.splitlines()
    errors: list[LatexIssue] = []
    for index, text in enumerate(lines):
        if match := _FILE_LINE_ERROR.match(text):
            issue = _issue_in(match["file"], int(match["line"]), match["message"], document_name)
        elif match := _BARE_ERROR.match(text):
            issue = LatexIssue(_context_line(lines[index + 1 : index + 1 + _BARE_ERROR_LOOKAHEAD]), match["message"])
        else:
            continue
        if issue not in errors:
            errors.append(issue)
    return errors


def _issue_in(file: str, line: int, message: str, document_name: str) -> LatexIssue:
    if Path(file).name == document_name:
        return LatexIssue(line, message)
    return LatexIssue(None, f"{Path(file).name}:{line}: {message}")


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
            parse_errors(await asyncio.to_thread(_read_log, log), source.name) or parse_errors(output, source.name)
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


def _read_head(source: Path) -> str:
    with source.open(encoding="utf-8", errors="replace") as file:
        return "".join(file.readline() for _ in range(_MAGIC_COMMENT_LINES))
