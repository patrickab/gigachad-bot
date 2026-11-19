"""Disk layout of workbench documents: one chat directory per kind, one folder per document inside it.

- `<root>/<kind>/` is where the Pi chats of a *kind* document run, unless they run in a LaTeX project or mount
- `<root>/<kind>/<slug>/<stem>/<stem>.<ext>` is a document of project `slug`
- PDFs are only read, never moved: `pdf` sessions run in `<root>/pdf` and read the PDF where it is
- Canvases stay stored: `canvas` sessions run in `<root>/canvas` and see a canvas through the screenshots sent to them
"""

from __future__ import annotations

from contextlib import contextmanager
from itertools import takewhile
import logging
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Literal, get_args

from lib.naming import segment

if TYPE_CHECKING:
    from collections.abc import Iterator

Kind = Literal["latex", "markdown", "pdf", "canvas"]
KINDS: tuple[Kind, ...] = get_args(Kind)
# `.tex` first, it is the suffix of a new latex document. The others are the supporting files of a LaTeX project.
_SUFFIXES: dict[str, Kind] = {".tex": "latex", ".bib": "latex", ".sty": "latex", ".cls": "latex", ".md": "markdown", ".pdf": "pdf"}

log = logging.getLogger(__name__)


def kind_of(path: str | Path) -> Kind | None:
    """The workbench kind of *path* by its suffix, case-insensitive, None for a file the workbench does not edit."""
    return _SUFFIXES.get(Path(path).suffix.lower())


def suffix_of(kind: Kind) -> str:
    """The file suffix of a new document of *kind*."""
    return next(suffix for suffix, known in _SUFFIXES.items() if known == kind)


def kind_directory(root: Path, kind: Kind) -> Path:
    """The folder of every *kind* document, where their chats run unless they have a project folder of their own."""
    return root / kind


def document_folder(root: Path, kind: Kind, slug: str, stem: str) -> Path:
    """The folder of document *stem* of project *slug*, ValueError for an empty, nested or hidden segment or a control character."""
    return kind_directory(root, kind) / segment(slug) / segment(stem)


@contextmanager
def new_folder(folder: Path) -> Iterator[Path]:
    """Create *folder* with its missing parents. When the block raises, the topmost folder it created is removed with everything inside, a folder that existed before never is."""
    created = list(takewhile(lambda directory: not directory.exists(), (folder, *folder.parents)))
    folder.mkdir(parents=True, exist_ok=True)
    try:
        yield folder
    except BaseException:
        if created:
            shutil.rmtree(created[-1], ignore_errors=True)
        raise


def project_directories(root: Path, slug: str) -> list[Path]:
    """The existing document directories of project *slug*, one per kind at most."""
    segment(slug)
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
    segment(new)
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
