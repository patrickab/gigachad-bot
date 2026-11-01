"""Three-way text merge through `git merge-file`, conflicting hunks take the other side's text."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

_MERGE_SECONDS = 10


class TextMergeError(RuntimeError):
    """`git` is missing or `git merge-file` failed."""


def merge_theirs(base: str, ours: str, theirs: str) -> str:
    """Merge *ours* and *theirs* edits of *base*, a conflict resolves to *theirs* (`git merge-file -p --theirs`).

    Non-empty texts get a final newline first, so a last line that only gained or lost one is not an edit.
    """
    with tempfile.TemporaryDirectory(prefix="gigachad-merge-") as directory:
        files = []
        for name, text in (("ours", ours), ("base", base), ("theirs", theirs)):
            file = Path(directory) / name
            file.write_text(_with_final_newline(text), encoding="utf-8", newline="")
            files.append(str(file))
        try:
            result = subprocess.run(
                ["git", "merge-file", "-p", "--theirs", *files],
                capture_output=True,
                timeout=_MERGE_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TextMergeError(f"git merge-file could not run: {exc}") from exc
    if result.returncode != 0:
        raise TextMergeError(f"git merge-file failed: {result.stderr.decode('utf-8', 'replace').strip()}")
    return result.stdout.decode("utf-8")


def _with_final_newline(text: str) -> str:
    return text if not text or text.endswith("\n") else text + "\n"
