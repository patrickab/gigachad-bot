"""Disk layout of workbench documents: one omp session directory per kind, one folder per document inside it.

- `<root>/<kind>/.omp/` holds the kind's omp prompt and config, mirrored from `omp/` on every backend start
- `<root>/<kind>/<slug>/<stem>/<stem>.<ext>` is a document of project `slug`
- PDFs are only read, never moved: `pdf` sessions run in `<root>/pdf` and read the PDF where it is
"""

from __future__ import annotations

from contextlib import contextmanager
from itertools import takewhile
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from collections.abc import Iterator

Kind = Literal["latex", "markdown", "pdf"]
KINDS: tuple[Kind, ...] = get_args(Kind)
OMP_DIRECTORY = ".omp"
_SUFFIXES: dict[str, Kind] = {".tex": "latex", ".md": "markdown", ".pdf": "pdf"}
_PROMPTS = Path(__file__).parent / "omp"

log = logging.getLogger(__name__)


def kind_of(path: str | Path) -> Kind | None:
    """The workbench kind of *path* by its suffix, case-insensitive, None for a file the workbench does not edit."""
    return _SUFFIXES.get(Path(path).suffix.lower())


def suffix_of(kind: Kind) -> str:
    """The file suffix of a new document of *kind*."""
    return next(suffix for suffix, known in _SUFFIXES.items() if known == kind)


def kind_directory(root: Path, kind: Kind) -> Path:
    """The cwd of every omp session editing a *kind* document."""
    return root / kind


def _segment(value: str, label: str) -> str:
    invalid = any(ord(character) < 32 for character in value)
    if not value or invalid or Path(value).name != value or value.startswith("."):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def document_folder(root: Path, kind: Kind, slug: str, stem: str) -> Path:
    """The folder of document *stem* of project *slug*, ValueError for an empty, nested or hidden segment or a control character."""
    return kind_directory(root, kind) / _segment(slug, "project slug") / _segment(stem, "document name")


@contextmanager
def new_folder(folder: Path) -> Iterator[Path]:
    """Create *folder* with its missing parents, removing the ones still empty when the block raises."""
    created = list(takewhile(lambda directory: not directory.exists(), (folder, *folder.parents)))
    folder.mkdir(parents=True, exist_ok=True)
    try:
        yield folder
    except BaseException:
        for directory in created:
            try:
                directory.rmdir()
            except OSError:
                break
        raise


def _write_changed(path: Path, content: bytes) -> None:
    if not path.is_file() or path.read_bytes() != content:
        path.write_bytes(content)


def mirror_omp(root: Path) -> None:
    """Make `.omp/SYSTEM.md` and `.omp/config.yml` of every kind directory equal the committed prompt files.

    - A file already equal is left untouched, so its mtime stays and file sync uploads nothing
    - A kind that fails is logged and skipped, the others are still mirrored
    """
    for kind in KINDS:
        try:
            directory = kind_directory(root, kind) / OMP_DIRECTORY
            directory.mkdir(parents=True, exist_ok=True)
            _write_changed(directory / "SYSTEM.md", (_PROMPTS / f"{kind}.md").read_bytes())
            _write_changed(directory / "config.yml", (_PROMPTS / "config.yml").read_bytes())
        except OSError:
            log.exception("Could not mirror the %s omp prompt into %s", kind, root)


def project_directories(root: Path, slug: str) -> list[Path]:
    """The existing document directories of project *slug*, one per kind at most."""
    _segment(slug, "project slug")
    return [directory for kind in KINDS if (directory := kind_directory(root, kind) / slug).is_dir()]


def directory_moves(root: Path, old: str, new: str) -> list[tuple[str, str]]:
    """`(old, new)` absolute directory paths of every kind when project *old* becomes *new*, existing on disk or not."""
    return [(str(kind_directory(root, kind) / old), str(kind_directory(root, kind) / new)) for kind in KINDS]


def _remove_empty_tree(directory: Path) -> bool:
    """Remove *directory* when it holds nothing but empty directories, returning whether it did."""
    if directory.is_symlink() or not directory.is_dir():
        return False
    inside = list(directory.rglob("*"))
    if any(path.is_symlink() or not path.is_dir() for path in inside):
        return False
    for path in sorted(inside, key=lambda path: len(path.parts), reverse=True):
        path.rmdir()
    directory.rmdir()
    return True


def move_project_directories(root: Path, old: str, new: str) -> list[tuple[Path, Path]]:
    """Move every directory of project *old* to *new*, returning the moves done.

    - A destination holding only empty directories (left by a deleted project) is removed first
    - Any other existing destination raises FileExistsError
    - A failed move puts back the ones already done, then raises
    """
    _segment(new, "project slug")
    done: list[tuple[Path, Path]] = []
    try:
        for source in project_directories(root, old):
            target = source.with_name(new)
            if target.exists() and not _remove_empty_tree(target):
                raise FileExistsError(f"Workbench folder already exists: {target}")
            source.rename(target)
            done.append((source, target))
    except OSError:
        restore_directories(done)
        raise
    return done


def restore_directories(moves: list[tuple[Path, Path]]) -> None:
    """Undo *moves* from `move_project_directories`, latest first, logging a move that cannot be undone."""
    for source, target in reversed(moves):
        try:
            target.rename(source)
        except OSError:
            log.exception("Could not move workbench folder %s back to %s", target, source)


def owe_new_paths(before: object, after: object) -> object:
    """The session links *after* a relocation, every file whose path changed owing each of its sessions the new path (`moved`).

    A session told its file's path earlier would edit the old one, so its next message names the file again
    """
    if not isinstance(before, dict) or not isinstance(after, dict):
        return after
    return {path: link if path in before else _owing_every_session(link) for path, link in after.items()}


def _owing_every_session(link: object) -> object:
    """A file's session link with all its sessions in `moved`, a bare session id as stored before sessions were kept too."""
    if isinstance(link, str):
        return {"active": link, "sessions": [link], "moved": [link]}
    if not isinstance(link, dict):
        return link
    return {**link, "moved": [session for session in link.get("sessions", []) if isinstance(session, str)]}
