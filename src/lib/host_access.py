"""What logins with host access reach outside the file roots: the home directory minus credential stores.

They can already run commands as the host user, so the guard is about not handing out secrets through the file APIs,
not about confinement.
"""

from __future__ import annotations

import os
from pathlib import Path

# Credential stores host access never opens outside the roots.
_SECRET_DIRECTORIES = frozenset({
    ".ssh", ".gnupg", ".secrets", ".aws", ".azure", ".kube", ".docker", ".password-store", ".pki", ".mozilla",
})
# Credential stores below `~/.config` and `~/.local/share`, matched as `(parent, child)` directory pairs.
_SECRET_SUBDIRECTORIES = frozenset({
    *((".config", name) for name in ("gh", "gcloud", "google-chrome", "chromium", "BraveSoftware", "op", "rclone")),
    ("share", "keyrings"),
})
_SECRET_NAMES = frozenset({
    ".netrc", ".git-credentials", ".pgpass", ".npmrc", ".pypirc",
    ".bash_history", ".zsh_history", ".psql_history", ".python_history",
})
_OMP_SECRET_NAMES = ("agent.db", "models.yml")


def is_secret_path(path: Path) -> bool:
    """Whether *path* is a credential store: SSH/GPG/cloud keys, browser profiles, token files, env files, omp's API keys."""
    name = path.name.lower()
    pairs = zip(path.parts, path.parts[1:], strict=False)
    if _SECRET_DIRECTORIES.intersection(path.parts) or _SECRET_SUBDIRECTORIES.intersection(pairs):
        return True
    if name in _SECRET_NAMES or name == ".env" or name.startswith(".env.") or (name == "env" and path.parent.name == "gigachad-bot"):
        return True
    return ".omp" in path.parts and name.startswith(_OMP_SECRET_NAMES)


def host_readable(target: Path) -> bool:
    """Whether host access may open the resolved *target* outside the roots: under the home directory and no credential store."""
    return target.is_relative_to(Path.home().resolve()) and not is_secret_path(target)


def _is_gitdir_pointer(git_file: Path) -> bool:
    """Whether a `.git` file is a linked worktree's pointer (`gitdir: <clone>/.git/worktrees/<name>`)."""
    try:
        with git_file.open(encoding="utf-8") as handle:
            return handle.read(len("gitdir:")) == "gitdir:"
    except (OSError, UnicodeDecodeError):
        return False


def linked_worktree_of(target: Path) -> Path | None:
    """The linked git worktree holding *target*: the nearest directory with a `.git` marker, when that marker points at a clone."""
    for directory in (target, *target.parents):
        marker = directory / ".git"
        if marker.exists():
            return directory if marker.is_file() and _is_gitdir_pointer(marker) else None
    return None


def list_directories(path: str) -> tuple[Path, list[str]]:
    """Resolve *path* (`~` expanded) and list its subdirectory names, hidden ones included.

    - Used to pick a new root, so it browses the whole host, not only registered roots
    - Raises NotADirectoryError for a missing or non-directory path, PermissionError when unreadable
    """
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(path)
    with os.scandir(directory) as entries:
        names = [entry.name for entry in entries if entry.is_dir()]
    return directory, sorted(names, key=str.casefold)
