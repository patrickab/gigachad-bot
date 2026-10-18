"""Working-tree changes of a git directory, as the file list and unified patches the diff viewer shows."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Literal

GIT_TIMEOUT_SECONDS = 10
MAX_DIFF_BYTES = 1_000_000

ChangeStatus = Literal["modified", "added", "deleted", "renamed", "untracked", "conflict"]
_CONFLICT_PAIRS = frozenset({"DD", "AU", "UD", "UA", "DU", "AA", "UU"})


@dataclass(frozen=True)
class ChangedFile:
    """*path* is relative to the inspected directory, *original_path* is set for renames."""

    path: str
    status: ChangeStatus
    original_path: str | None = None


def _git(directory: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), *args],
        capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    return result.stdout


def _repository_root(directory: Path) -> Path | None:
    try:
        return Path(_git(directory, "rev-parse", "--show-toplevel").strip())
    except (subprocess.SubprocessError, OSError):
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


def changed_files(directory: Path) -> list[ChangedFile] | None:
    """Changed files at or below *directory*, None when it is not inside a git repository."""
    repository = _repository_root(directory)
    if repository is None:
        return None
    output = _git(repository, "status", "--porcelain=v1", "-z", "--untracked-files=all")
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


def file_patch(directory: Path, relative_path: str) -> tuple[ChangeStatus, str] | None:
    """Unified patch of one changed file against HEAD, None when it is unchanged, binary or too large."""
    change = next((file for file in changed_files(directory) or [] if file.path == relative_path), None)
    if change is None:
        return None
    target = directory / relative_path
    if change.status == "untracked" or (change.status == "added" and not _has_head(directory)):
        try:
            if target.stat().st_size > MAX_DIFF_BYTES:
                return None
            content = target.read_bytes()
        except OSError:
            return None
        if b"\0" in content:
            return None
        return change.status, _added_file_patch(relative_path, content.decode("utf-8", errors="replace"))
    paths = [change.original_path, relative_path] if change.original_path else [relative_path]
    try:
        patch = _git(directory, "diff", "--no-color", "--no-ext-diff", "--relative", "--unified=3", "HEAD", "--", *paths)
    except (subprocess.SubprocessError, OSError):
        return None
    if "\n@@ " not in patch or len(patch) > MAX_DIFF_BYTES:
        return None
    return change.status, patch


def _has_head(directory: Path) -> bool:
    try:
        _git(directory, "rev-parse", "--verify", "HEAD")
    except (subprocess.SubprocessError, OSError):
        return False
    return True


class UnsupportedDiffContent(ValueError):
    """The file cannot be edited in a native diff (outside the mount, deleted, binary, oversized, not UTF-8)."""


def _head_blob(repository: Path, repository_path: str) -> bytes:
    """Bytes of *repository_path* at HEAD; empty when HEAD or the path does not exist there."""
    spec = f"HEAD:{repository_path}"
    env = {**os.environ, "LC_ALL": "C", "GIT_EXTERNAL_DIFF": "", "GIT_LITERAL_PATHSPECS": "1"}
    base = ["git", "--no-pager", "-C", str(repository)]
    try:
        size = subprocess.run(
            [*base, "cat-file", "-s", spec], capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=True, env=env,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return b""
    if int(size) > MAX_DIFF_BYTES:
        raise UnsupportedDiffContent("File is too large for a native diff")
    return subprocess.run(
        [*base, "cat-file", "blob", spec], capture_output=True, timeout=GIT_TIMEOUT_SECONDS, check=True, env=env,
    ).stdout


def _decode(data: bytes) -> str:
    if b"\0" in data:
        raise UnsupportedDiffContent("Binary files cannot be diffed natively")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedDiffContent("File is not valid UTF-8") from exc


def file_contents(directory: Path, relative_path: str) -> tuple[str, str]:
    """``(HEAD content, working content)`` of one regular file below *directory*.

    The baseline is empty for untracked files and unborn HEAD, and follows renames to the original path.
    Raises UnsupportedDiffContent for paths escaping *directory* (including through symlinks), deleted,
    binary, non-UTF-8 or oversized files.
    """
    if not relative_path or "\0" in relative_path or Path(relative_path).is_absolute():
        raise UnsupportedDiffContent("Invalid path")
    directory = directory.resolve()
    lexical = Path(os.path.normpath(directory / relative_path))
    if not lexical.is_relative_to(directory) or lexical == directory:
        raise UnsupportedDiffContent("Path is outside the mount")
    if lexical.resolve() != lexical:
        raise UnsupportedDiffContent("Symlinked paths are not supported")
    try:
        if not lexical.is_file():
            raise UnsupportedDiffContent("Not an existing regular file")
        if lexical.stat().st_size > MAX_DIFF_BYTES:
            raise UnsupportedDiffContent("File is too large for a native diff")
        content = _decode(lexical.read_bytes())
    except OSError as exc:
        raise UnsupportedDiffContent(str(exc)) from exc
    repository = _repository_root(directory)
    if repository is None:
        return "", content
    repository = repository.resolve()
    head_path = lexical.relative_to(repository).as_posix() if lexical.is_relative_to(repository) else None
    if head_path is None:
        return "", content
    try:
        change = next((file for file in changed_files(directory) or [] if file.path == lexical.relative_to(directory).as_posix()), None)
        if change is not None and change.status == "untracked":
            return "", content
        if change is not None and change.original_path:
            original = Path(os.path.normpath(directory / change.original_path))
            if not original.is_relative_to(directory):
                raise UnsupportedDiffContent("Original path is outside the mount")
            head_path = original.relative_to(repository).as_posix()
        return _decode(_head_blob(repository, head_path)), content
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        if isinstance(exc, UnsupportedDiffContent):
            raise
        raise UnsupportedDiffContent("Could not read the HEAD version") from exc
