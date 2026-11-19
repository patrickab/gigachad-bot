"""Disk layout of workbench documents: one omp session directory per kind, one folder per document inside it.

- `<root>/<kind>/.omp/` holds the kind's omp prompt (`prompts.render`, as Pi chats get it) and config, mirrored on every start
- `<root>/latex/.omp/templates/<set>/` holds every LaTeX template set without its `example.pdf` and `previews/`, mirrored on every backend start too
- A mounted LaTeX project folder gets the same `.omp/` (`mounted_latex_session`), with a config that keeps its repository's `AGENTS.md`
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

import config
from lib import latex, latex_templates
from lib.git import GitError, exclude_locally, repository_top
from lib.naming import segment
from lib.workbench.prompts import render

if TYPE_CHECKING:
    from collections.abc import Iterator

Kind = Literal["latex", "markdown", "pdf", "canvas"]
KINDS: tuple[Kind, ...] = get_args(Kind)
OMP_DIRECTORY = ".omp"
TEMPLATE_DIRECTORY = "templates"
# `.tex` first, it is the suffix of a new latex document. The others are the supporting files of a LaTeX project.
_SUFFIXES: dict[str, Kind] = {".tex": "latex", ".bib": "latex", ".sty": "latex", ".cls": "latex", ".md": "markdown", ".pdf": "pdf"}
_OMP_CONFIGS = config.TEMPLATES / "omp"

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
    """Make `.omp/SYSTEM.md` and `.omp/config.yml` of every kind directory equal the committed prompt files, `latex/.omp/templates/` the template sets.

    - A file already equal is left untouched, so its mtime stays and file sync uploads nothing
    - A kind that fails is logged and skipped, the others are still mirrored
    """
    for kind in KINDS:
        try:
            _mirror_prompt(kind_directory(root, kind) / OMP_DIRECTORY, kind, "config.yml")
        except OSError:
            log.exception("Could not mirror the %s omp prompt into %s", kind, root)
    try:
        _mirror_template_sets(kind_directory(root, "latex") / OMP_DIRECTORY / TEMPLATE_DIRECTORY)
    except OSError:
        log.exception("Could not mirror the LaTeX template sets into %s", root)


def _mirror_prompt(directory: Path, kind: Kind, config_name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_changed(directory / "SYSTEM.md", render(kind).encode())
    _write_changed(directory / "config.yml", (_OMP_CONFIGS / config_name).read_bytes())


def mounted_latex_session(path: Path, mount: Path) -> Path | None:
    """The folder an omp session on the mounted LaTeX file *path* starts in: its project folder, given the LaTeX `.omp/`.

    - omp then takes the LaTeX prompt instead of its coding prompt, `@` lists the project's files and the repository's `AGENTS.md` still loads
    - The `.omp/` is ignored through the repository's `info/exclude`, so there is nothing to commit
    - None when the folder is *mount* or a repository top, whose `.omp/` is the user's, or when writing fails (logged)
    """
    folder = latex.project_of(path, ceiling=mount).folder
    top = repository_top(folder)
    if folder in (mount, top):
        return None
    try:
        directory = folder / OMP_DIRECTORY
        _mirror_prompt(directory, "latex", "mounted.yml")
        _mirror_template_sets(directory / TEMPLATE_DIRECTORY)
        if top is not None:
            exclude_locally(top, directory)
    except (OSError, GitError):
        log.exception("Could not give the LaTeX project %s its omp prompt", folder)
        return None
    return folder


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
