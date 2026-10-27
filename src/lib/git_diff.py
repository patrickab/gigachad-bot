"""Working-tree changes of a git directory, as the file list and unified patches the diff viewer shows."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Literal

from lib.git import GitError, git_ok, git_text, run_git

MAX_DIFF_BYTES = 1_000_000

ChangeStatus = Literal["modified", "added", "deleted", "renamed", "untracked", "conflict"]
_CONFLICT_PAIRS = frozenset({"DD", "AU", "UD", "UA", "DU", "AA", "UU"})


@dataclass(frozen=True)
class ChangedFile:
    """*path* is relative to the inspected directory, *original_path* is set for renames."""

    path: str
    status: ChangeStatus
    original_path: str | None = None


def _repository_root(directory: Path) -> Path | None:
    try:
        return Path(git_text(directory, "rev-parse", "--show-toplevel").strip())
    except GitError:
        return None


def _classify(pair: str) -> ChangeStatus:
    if pair == "??":
        return "untracked"
    if pair in _CONFLICT_PAIRS or "U" in pair:
        return "conflict"
    if "D" in pair:
        return "deleted"
    if "R" in pair or "C" in pair:
        return "renamed"
    if "A" in pair:
        return "added"
    return "modified"


def _parse_porcelain(output: str) -> list[tuple[str, str, str | None]]:
    """`git status --porcelain=v1 -z` as ``(repository-relative path, XY pair, original path)``."""
    records = output.split("\0")
    entries: list[tuple[str, str, str | None]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if len(record) < 4 or record[2] != " ":
            continue
        pair, path = record[:2], record[3:]
        original = None
        if pair[0] in "RC" or pair[1] in "RC":
            original = records[index] or None
            index += 1
        entries.append((path, pair, original))
    return entries


def _relative_to(directory: Path, path: Path) -> str:
    return (path.relative_to(directory) if path.is_relative_to(directory) else path).as_posix()


def changed_files(directory: Path, only: str | None = None, repository: Path | None = None) -> list[ChangedFile] | None:
    """Changed files at or below *directory*, None when it is not inside a git repository.

    - *only* limits the scan to one file below *directory*, but then git cannot pair a staged rename
    - *repository* is the already known top level of *directory*'s repository, saving the git call that finds it
    """
    repository = repository or _repository_root(directory)
    if repository is None:
        return None
    pathspec = ["--", str(directory / only)] if only else []
    output = git_text(repository, "--literal-pathspecs", "status", "--porcelain=v1", "-z", "--untracked-files=all", *pathspec)
    files = []
    for path, pair, original in _parse_porcelain(output):
        if not (repository / path).is_relative_to(directory):
            continue
        files.append(ChangedFile(
            path=(repository / path).relative_to(directory).as_posix(),
            status=_classify(pair),
            original_path=_relative_to(directory, repository / original) if original else None,
        ))
    return sorted(files, key=lambda file: file.path.lower())


def _added_file_patch(path: str, content: str) -> str:
    """Patch that adds *content* as a new file."""
    lines = content.split("\n")
    has_trailing_newline = content.endswith("\n")
    if has_trailing_newline:
        lines.pop()
    body = "\n".join(f"+{line}" for line in lines)
    marker = "" if has_trailing_newline or not lines else "\n\\ No newline at end of file"
    return "\n".join([
        f"diff --git a/{path} b/{path}",
        "new file mode 100644",
        "--- /dev/null",
        f"+++ b/{path}",
        f"@@ -0,0 +1,{len(lines)} @@",
        f"{body}{marker}",
    ])


def _find_change(directory: Path, relative_path: str, repository: Path | None = None) -> ChangedFile | None:
    """Change of one file, scanning the whole tree only when it may be the new side of a staged rename."""
    repository = repository or _repository_root(directory)
    if repository is None:
        return None
    change = next(iter(changed_files(directory, relative_path, repository) or []), None)
    if change is None or change.status != "added":
        return change
    return next((file for file in changed_files(directory, repository=repository) or [] if file.path == relative_path), None)


def _read_text_candidate(target: Path) -> bytes | None:
    """Bytes of a file small enough and free of NUL bytes to diff as text, else None."""
    try:
        if target.stat().st_size > MAX_DIFF_BYTES:
            return None
        content = target.read_bytes()
    except OSError:
        return None
    return None if b"\0" in content else content


def file_patch(directory: Path, relative_path: str) -> tuple[ChangeStatus, str] | None:
    """Unified patch of one changed file against HEAD, None when it is unchanged, binary or too large."""
    change = _find_change(directory, relative_path)
    if change is None:
        return None
    target = directory / relative_path
    if change.status == "untracked" or (change.status == "added" and not _has_head(directory)):
        content = _read_text_candidate(target)
        if content is None:
            return None
        return change.status, _added_file_patch(relative_path, content.decode("utf-8", errors="replace"))
    paths = [change.original_path, relative_path] if change.original_path else [relative_path]
    try:
        patch = git_text(directory, "diff", "--no-color", "--no-ext-diff", "--relative", "--unified=3", "HEAD", "--", *paths)
    except GitError:
        return None
    if "\n@@ " not in patch or len(patch) > MAX_DIFF_BYTES:
        return None
    return change.status, patch


def file_stamp(directory: Path, relative_path: str) -> str:
    """Modification fingerprint of a working file, empty when it is gone, lets clients spot content edits."""
    try:
        stat = (directory / relative_path).stat()
    except OSError:
        return ""
    return f"{stat.st_mtime_ns}-{stat.st_size}"


def _parse_numstat(output: str) -> dict[str, tuple[int, int]]:
    """`git diff --numstat -z` as ``{path: (added, removed)}``, binary files are left out."""
    records = output.split("\0")
    counts: dict[str, tuple[int, int]] = {}
    index = 0
    while index < len(records):
        parts = records[index].split("\t", 2)
        index += 1
        if len(parts) != 3:
            continue
        added, removed, path = parts
        if not path:  # rename: the next two records are the old and the new path
            path = records[index + 1] if index + 1 < len(records) else ""
            index += 2
        if added.isdigit() and removed.isdigit():
            counts[path] = (int(added), int(removed))
    return counts


def _added_line_count(target: Path) -> tuple[int, int] | None:
    """Line count of a file that is new, None when `file_patch` would show no text diff."""
    content = _read_text_candidate(target)
    if content is None:
        return None
    return content.count(b"\n") + (0 if not content or content.endswith(b"\n") else 1), 0


def line_counts(directory: Path, files: list[ChangedFile]) -> dict[str, tuple[int, int]]:
    """``{path: (added, removed)}`` of *files* against HEAD, one git call for all tracked files."""
    has_head = _has_head(directory)
    tracked: dict[str, tuple[int, int]] = {}
    if has_head:
        with contextlib.suppress(GitError):
            output = git_text(directory, "diff", "--numstat", "-z", "--no-color", "--no-ext-diff", "--relative", "HEAD")
            tracked = _parse_numstat(output)
    counts: dict[str, tuple[int, int]] = {}
    for file in files:
        is_new = file.status == "untracked" or (file.status == "added" and not has_head)
        count = _added_line_count(directory / file.path) if is_new else tracked.get(file.path)
        if count is not None:
            counts[file.path] = count
    return counts


def _has_head(directory: Path) -> bool:
    return git_ok(directory, "rev-parse", "--verify", "HEAD")


class UnsupportedDiffContent(ValueError):
    """The file cannot be edited in a native diff (outside the mount, deleted, binary, oversized, not UTF-8)."""


_HEAD_UNREADABLE = "Could not read the HEAD version"


def _head_blob(repository: Path, repository_path: str) -> bytes:
    """Bytes of *repository_path* at HEAD; empty when HEAD or the path does not exist there."""
    spec = f"HEAD:{repository_path}"
    env = {"GIT_EXTERNAL_DIFF": "", "GIT_LITERAL_PATHSPECS": "1"}
    # Bytes, not text, so git_text does not apply: a timeout, an odd size or a failed blob read is an unreadable HEAD.
    try:
        probe = run_git(repository, "--no-pager", "cat-file", "-s", spec, check=False, env=env)
        if probe.returncode:
            return b""
        if int(probe.stdout.strip()) <= MAX_DIFF_BYTES:
            return run_git(repository, "--no-pager", "cat-file", "blob", spec, text=False, env=env).stdout
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        raise UnsupportedDiffContent(_HEAD_UNREADABLE) from exc
    raise UnsupportedDiffContent("File is too large for a native diff")


def _decode(data: bytes) -> str:
    if b"\0" in data:
        raise UnsupportedDiffContent("Binary files cannot be diffed natively")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedDiffContent("File is not valid UTF-8") from exc


def _checked_target(directory: Path, relative_path: str) -> Path:
    """*relative_path* below the resolved *directory*, refused when it escapes it or passes through a symlink."""
    if not relative_path or "\0" in relative_path or Path(relative_path).is_absolute():
        raise UnsupportedDiffContent("Invalid path")
    lexical = Path(os.path.normpath(directory / relative_path))
    if not lexical.is_relative_to(directory) or lexical == directory:
        raise UnsupportedDiffContent("Path is outside the mount")
    if lexical.resolve() != lexical:
        raise UnsupportedDiffContent("Symlinked paths are not supported")
    return lexical


def _head_baseline(repository: Path, directory: Path, target: Path) -> str:
    """HEAD text of *target*: empty when untracked, the original path's for a staged rename."""
    try:
        change = _find_change(directory, target.relative_to(directory).as_posix(), repository)
    except (GitError, ValueError) as exc:
        # ValueError covers non-UTF-8 porcelain output, such as a staged rename's undecodable original path.
        raise UnsupportedDiffContent(_HEAD_UNREADABLE) from exc
    if change is not None and change.status == "untracked":
        return ""
    if change is not None and change.original_path:
        target = Path(os.path.normpath(directory / change.original_path))
        if not target.is_relative_to(directory):
            raise UnsupportedDiffContent("Original path is outside the mount")
        if not target.is_relative_to(repository):
            raise UnsupportedDiffContent(_HEAD_UNREADABLE)
    return _decode(_head_blob(repository, target.relative_to(repository).as_posix()))


def file_contents(directory: Path, relative_path: str) -> tuple[str, str]:
    """``(HEAD content, working content)`` of one regular file below *directory*.

    The baseline is empty for untracked files and unborn HEAD, and follows renames to the original path.
    Raises UnsupportedDiffContent for paths escaping *directory* (including through symlinks), deleted,
    binary, non-UTF-8 or oversized files.
    """
    directory = directory.resolve()
    target = _checked_target(directory, relative_path)
    try:
        if not target.is_file():
            raise UnsupportedDiffContent("Not an existing regular file")
        if target.stat().st_size > MAX_DIFF_BYTES:
            raise UnsupportedDiffContent("File is too large for a native diff")
        content = _decode(target.read_bytes())
    except OSError as exc:
        raise UnsupportedDiffContent(str(exc)) from exc
    repository = _repository_root(directory)
    if repository is None:
        return "", content
    repository = repository.resolve()
    if not target.is_relative_to(repository):
        return "", content
    return _head_baseline(repository, directory, target), content
