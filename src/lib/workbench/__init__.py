"""Disk layout of workbench documents: one omp session directory per kind, one folder per document inside it.

- `<root>/<kind>/.omp/` holds the kind's omp prompt and config, mirrored from `omp/` on every backend start
- `<root>/latex/.templates/<set>/` holds every LaTeX template set without its `example.pdf` and `previews/`, mirrored on every backend start too
- `<root>/<kind>/<slug>/<stem>/<stem>.<ext>` is a document of project `slug`
- PDFs are only read, never moved: `pdf` sessions run in `<root>/pdf` and read the PDF where it is
"""

from __future__ import annotations

from contextlib import contextmanager
from itertools import takewhile
import logging
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Literal, get_args

import config
from lib import latex_templates
from lib.naming import segment

if TYPE_CHECKING:
    from collections.abc import Iterator

Kind = Literal["latex", "markdown", "pdf"]
KINDS: tuple[Kind, ...] = get_args(Kind)
OMP_DIRECTORY = ".omp"
TEMPLATE_DIRECTORY = ".templates"
# `.tex` first, it is the suffix of a new latex document. The others are the supporting files of a LaTeX project.
_SUFFIXES: dict[str, Kind] = {".tex": "latex", ".bib": "latex", ".sty": "latex", ".cls": "latex", ".md": "markdown", ".pdf": "pdf"}
_PROMPTS = config.TEMPLATES / "omp"

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


def _write_changed(path: Path, content: bytes) -> None:
    if not path.is_file() or path.read_bytes() != content:
        path.write_bytes(content)


def mirror_omp(root: Path) -> None:
    """Make `.omp/SYSTEM.md` and `.omp/config.yml` of every kind directory equal the committed prompt files, `latex/.templates/` the template sets.

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
    try:
        _mirror_template_sets(kind_directory(root, "latex") / TEMPLATE_DIRECTORY)
    except OSError:
        log.exception("Could not mirror the LaTeX template sets into %s", root)


def _mirror_template_sets(directory: Path) -> None:
    """Make *directory* hold the agent files of every template set (`TemplateSet.agent_files`), removing the files the sets no longer have."""
    wanted = {
        Path(template.name) / relative: path
        for template in latex_templates.template_sets()
        for relative, path in template.agent_files()
    }
    for relative, path in wanted.items():
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        _write_changed(directory / relative, path.read_bytes())
    for path in sorted(directory.rglob("*"), reverse=True):
        if path.is_dir() and not path.is_symlink():
            if not any(path.iterdir()):
                path.rmdir()
        elif path.relative_to(directory) not in wanted:
            path.unlink()


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
