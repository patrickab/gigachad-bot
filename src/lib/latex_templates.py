r"""LaTeX template sets in `templates/latex`: Beamer presentations with slide primitives and plain document templates such as letters.

- A set is a folder holding `template.toml` (its `label`, the `files` a new document gets beside `main.tex`, optional `image_macro`
  and `[[groups]]` with `key` and `label` in order) and `skeleton.tex`. A set with `primitives/` is a presentation set
- `presentation_theme` tells the presentation set a root uses, `TemplateSet.primitives` lists its slides
- `templates/latex/render.py` builds each presentation set's example and previews with this module
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
import re
import shutil
import tomllib
from typing import TYPE_CHECKING

import config
from lib.tex_source import plain_lines

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

TEMPLATE_SETS = config.TEMPLATES / "latex"
IMAGE_DIRECTORY = "imgs"
_MANIFEST = "template.toml"
_THEME = re.compile(r"\\usetheme\s*(?:\[[^\]]*\])?\s*\{(?P<name>[^}]*)\}")
_PRIMITIVE_FILE = re.compile(r"^(?P<number>\d+)-(?P<name>[^.]+)\.tex$")
_PRIMITIVE_HEADER = re.compile(r"^%\s*(?P<label>[^:]*?\S)\s*:\s*(?P<description>\S.*?)\s*$")
_MACRO = re.compile(r"[A-Za-z]+")
_COMMAND_WITH_ARGUMENT = re.compile(r"\\[a-zA-Z]+\{([^}]*)\}")
_COMMAND = re.compile(r"\\[a-zA-Z]+")

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatexPrimitive:
    """One slide of a template set: id `<group>/<name>`, its group's label, the label and description of its header line, the description's plain first clause, its frame and the version (mtime in ms) of its preview PNG, None without one."""

    id: str
    group: str
    group_label: str
    label: str
    description: str
    summary: str
    text: str
    preview: int | None = None


@dataclass(frozen=True)
class PrimitiveGroup:
    """A group of primitives, the folder `primitives/<key>`, shown as *label*."""

    key: str
    label: str


class TemplateSet:
    """A template set folder: its `template.toml`, `skeleton.tex`, the `files` a new document gets and, for a presentation set, the slides in `primitives/`."""

    def __init__(self, path: Path, label: str, files: tuple[str, ...], image_macro: str | None, groups: tuple[PrimitiveGroup, ...]) -> None:
        self.name = path.name
        self.label = label
        self.path = path
        self.files = files
        self.groups = groups
        self._image_macro = image_macro
        # The mtimes of the primitive files and previews the slides were read at, the slides and each one's file by id
        self._read: tuple[tuple[tuple[Path, int | None, int | None], ...], tuple[list[LatexPrimitive], dict[str, Path]]] | None = None

    @property
    def is_presentation(self) -> bool:
        """Whether the set is a Beamer theme with slide primitives."""
        return (self.path / "primitives").is_dir()

    def primitives(self) -> list[LatexPrimitive]:
        r"""The slides `primitives/<group>/<NN>-<name>.tex` by group then number, read again when a file or preview changes. A malformed file is logged and skipped.

        A file is its header line `% <Label>: <description>` followed by one `\begin{frame}...\end{frame}`. Groups follow
        `groups`, unknown ones come last by name.
        """
        return self._slides()[0]

    def source(self, primitive_id: str) -> Path | None:
        """The file of primitive *primitive_id*, None for an unknown one."""
        return self._slides()[1].get(primitive_id)

    def preview(self, primitive_id: str) -> Path | None:
        """The preview PNG of primitive *primitive_id*, None for an unknown primitive or without one."""
        source = self.source(primitive_id)
        if source is None:
            return None
        preview = preview_path(source)
        return preview if preview.is_file() else None

    def image_reference(self, image: str) -> str | None:
        """The LaTeX that shows the image file *image* with this set's image macro, None for a set without one."""
        return None if self._image_macro is None else rf"\{self._image_macro}{{{image}}}"

    def populate(self, folder: Path) -> None:
        """Write `main.tex` from the skeleton, the set's `files` and an empty `imgs/` into the new *folder*, FileExistsError when `main.tex` exists."""
        skeleton = (self.path / "skeleton.tex").read_bytes()
        with (folder / "main.tex").open("xb") as file:
            file.write(skeleton)
        for name in self.files:
            shutil.copyfile(self.path / name, folder / name)
        (folder / IMAGE_DIRECTORY).mkdir()

    def agent_files(self) -> Iterator[tuple[Path, Path]]:
        """The `(relative, path)` of every file of the set an agent reads, all but `example.pdf` and `previews/`."""
        for path in sorted(self.path.rglob("*")):
            relative = path.relative_to(self.path)
            if path.is_file() and relative != Path("example.pdf") and relative.parts[0] != "previews":
                yield relative, path

    def _slides(self) -> tuple[list[LatexPrimitive], dict[str, Path]]:
        """The slides and each one's file by id, read again when the mtimes of the primitive files or their previews changed."""
        versions = tuple(
            (path, _mtime_ns(path), _mtime_ns(preview_path(path))) for path in sorted(self.path.glob("primitives/*/*.tex"))
        )
        read = self._read
        if read is None or read[0] != versions:
            read = (versions, self._read_slides(path for path, _, _ in versions))
            self._read = read
        return read[1]

    def _read_slides(self, paths: Iterable[Path]) -> tuple[list[LatexPrimitive], dict[str, Path]]:
        labels = {group.key: group.label for group in self.groups}
        order = list(labels)
        ranked: list[tuple[tuple[int, str], int, str, LatexPrimitive, Path]] = []
        for path in paths:
            read = _read_primitive(path, labels)
            if read is None:
                log.warning("Skipping malformed LaTeX primitive %s", path)
                continue
            number, primitive = read
            rank = (order.index(primitive.group), "") if primitive.group in labels else (len(order), primitive.group)
            ranked.append((rank, number, primitive.id, primitive, path))
        ranked.sort(key=lambda entry: entry[:3])
        return [entry[3] for entry in ranked], {entry[2]: entry[4] for entry in ranked}


def template_sets() -> list[TemplateSet]:
    """The template sets by name, the folders of `TEMPLATE_SETS` holding `template.toml` and `skeleton.tex`. A malformed `template.toml` is logged and skipped."""
    if not TEMPLATE_SETS.is_dir():
        return []
    found = []
    for folder in sorted(TEMPLATE_SETS.iterdir()):
        manifest = folder / _MANIFEST
        if (folder / "skeleton.tex").is_file() and (version := _mtime_ns(manifest)) is not None:
            template = _load_set(folder, version)
            if template is not None:
                found.append(template)
    return found


def template_set(name: str) -> TemplateSet | None:
    """The template set *name*, None for any other name."""
    return next((template for template in template_sets() if template.name == name), None)


def presentation_theme(root: Path) -> TemplateSet | None:
    r"""The presentation set the root document *root* uses: the first `\usetheme` naming one, comments ignored. None without one."""
    try:
        text = root.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    code = "".join(plain_lines(text))
    sets = {template.name: template for template in template_sets() if template.is_presentation}
    return next((sets[name] for match in _THEME.finditer(code) if (name := match.group("name").strip()) in sets), None)


def preview_path(path: Path) -> Path:
    """Where `render.py` writes the preview of the primitive file `<set>/primitives/<group>/<NN>-<name>.tex`."""
    return path.parents[2] / "previews" / path.parent.name / f"{path.stem}.png"


def _summary(description: str) -> str:
    r"""The first clause of a primitive's *description* as plain text, capitalized: `\cmd{x}` reads as `x`, other commands drop."""
    clause = _COMMAND.sub("", _COMMAND_WITH_ARGUMENT.sub(r"\1", description.split(", ")[0]))
    clause = " ".join(clause.split())
    return clause[:1].upper() + clause[1:]


@lru_cache(maxsize=32)
def _load_set(folder: Path, _version: int) -> TemplateSet | None:
    """The template set in *folder* from its `template.toml`, cached per mtime *_version* of that file. None when it is malformed."""
    try:
        manifest = tomllib.loads((folder / _MANIFEST).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        manifest = None
    if manifest is None or not _is_valid_manifest(folder, manifest):
        log.warning("Skipping LaTeX template set %s with a malformed %s", folder, _MANIFEST)
        return None
    groups = tuple(PrimitiveGroup(group["key"], group["label"]) for group in manifest.get("groups", []))
    return TemplateSet(folder, manifest["label"], tuple(manifest.get("files", [])), manifest.get("image_macro"), groups)


def _is_valid_manifest(folder: Path, manifest: dict) -> bool:
    """Check the manifest has a `label`, `files` naming files of *folder*, an `image_macro` name if any and well-formed `groups`."""
    files, macro, groups = manifest.get("files", []), manifest.get("image_macro"), manifest.get("groups", [])
    names_own_files = isinstance(files, list) and all(
        isinstance(name, str) and Path(name).name == name and (folder / name).is_file() for name in files
    )
    valid_macro = macro is None or (isinstance(macro, str) and _MACRO.fullmatch(macro) is not None)
    valid_groups = isinstance(groups, list) and all(
        isinstance(group, dict) and isinstance(group.get("key"), str) and isinstance(group.get("label"), str) for group in groups
    )
    return isinstance(manifest.get("label"), str) and names_own_files and valid_macro and valid_groups


def _mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _read_primitive(path: Path, group_labels: dict[str, str]) -> tuple[int, LatexPrimitive] | None:
    """The number and slide of the primitive file *path*, None when its name, header or frame is malformed."""
    named = _PRIMITIVE_FILE.match(path.name)
    try:
        header, _, body = path.read_text(encoding="utf-8").partition("\n")
    except (OSError, UnicodeDecodeError):
        return None
    described = _PRIMITIVE_HEADER.match(header)
    text = body.strip()
    one_frame = text.startswith(r"\begin{frame}") and text.endswith(r"\end{frame}") and text.count(r"\begin{frame}") == 1
    if named is None or described is None or not one_frame:
        return None
    preview = _mtime_ns(preview_path(path))
    group = path.parent.name
    description = described.group("description")
    return int(named.group("number")), LatexPrimitive(
        id=f"{group}/{named.group('name')}",
        group=group,
        group_label=group_labels.get(group, group[:1].upper() + group[1:]),
        label=described.group("label"),
        description=description,
        summary=_summary(description),
        text=text,
        preview=None if preview is None else preview // 1_000_000,
    )
