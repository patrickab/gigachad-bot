"""Repoints the references a LaTeX project holds to a file that was renamed."""

import io
import logging
import os
import re
from pathlib import Path

from lib import latex

logger = logging.getLogger(__name__)

_SOURCE_SUFFIXES = {".tex", ".bib"}
_SKIPPED_FOLDERS = {"node_modules"}
_GRAPHICS = (".pdf", ".png", ".jpg", ".jpeg", ".eps")
_REFERENCED_SUFFIXES = {*_SOURCE_SUFFIXES, *_GRAPHICS}
_IMPLIED_SUFFIXES = {
    "includegraphics": _GRAPHICS,
    "input": (".tex",),
    "include": (".tex",),
    "includeonly": (".tex",),
    "bibliography": (".bib",),
    "addbibresource": (),
}
_LISTS = {"includeonly", "bibliography"}
_COMMAND = re.compile(
    r"(?P<head>\\(?P<name>includegraphics|includeonly|include|input|bibliography|addbibresource)\*?(?![A-Za-z])"
    r"\s*(?:\[[^\]]*\]\s*)?\{)(?P<args>[^{}]*)(?=\})"
)
_PADDED = re.compile(r"(\s*)(.*?)(\s*)", re.DOTALL)


def rewrite_references(folder: Path, renamed_from: Path, renamed_to: Path) -> list[Path]:
    r"""Make every reference in the `.tex` and `.bib` files under *folder* that resolves to *renamed_from* resolve to *renamed_to*.

    - Commands: `\includegraphics`, `\input`, `\include`, `\includeonly`, `\bibliography`, `\addbibresource`, resolved against *folder*
    - The `% !TEX root` line of a file resolves against the file's own folder
    - A file no command can point at (not `.tex`, `.bib` or a graphic) rewrites nothing and walks nothing
    - A reference without extension stays without one while the extension is unchanged and no other format shadows it,
      else it names the new file in full. A `./` prefix stays, commented-out text is left alone
    - Every other byte of a file stays, a file that cannot be read, decoded or written is logged and skipped
    - Dot folders and `node_modules` are not searched
    Returns the files it rewrote.
    """
    source = Path(os.path.normpath(renamed_from))
    target = Path(os.path.normpath(renamed_to))
    if source.suffix.lower() not in _REFERENCED_SUFFIXES:
        return []
    rewritten = []
    for file in _source_files(folder):
        try:
            text = file.read_bytes().decode("utf-8")
            updated = _rewritten(text, file, folder, source, target) if source.stem in text else text
            if updated != text:
                file.write_bytes(updated.encode("utf-8"))
                rewritten.append(file)
        except (OSError, UnicodeDecodeError) as error:
            logger.warning("Skipped the references in %s: %s", file, error)
    return rewritten


def _source_files(folder: Path) -> list[Path]:
    """The `.tex` and `.bib` files under *folder* that are not build output, sorted."""
    found = []
    for directory, subfolders, names in os.walk(folder):
        subfolders[:] = sorted(name for name in subfolders if not name.startswith(".") and name not in _SKIPPED_FOLDERS)
        found += [
            path for name in sorted(names)
            if (path := Path(directory, name)).suffix.lower() in _SOURCE_SUFFIXES and not latex.is_build_output(path)
        ]
    return found


def _rewritten(text: str, file: Path, folder: Path, source: Path, target: Path) -> str:
    """*text* of *file* with its references to *source* pointing at *target*."""
    lines = []
    for number, line in enumerate(io.StringIO(text, newline="")):
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        if number < latex.MAGIC_COMMENT_LINES and (root := latex.ROOT_COMMENT.match(body)):
            body = root["head"] + _followed(root["root"], file.parent, (), source, target) + root["tail"]
        else:
            comment = latex.COMMENT.search(body)
            cut = comment.start() if comment else len(body)
            body = _COMMAND.sub(lambda command: _command_followed(command, folder, source, target), body[:cut]) + body[cut:]
        lines.append(body + ending)
    return "".join(lines)


def _command_followed(command: re.Match[str], folder: Path, source: Path, target: Path) -> str:
    """The command *command* with its file references repointed, its options and braces untouched."""
    name = command["name"]
    references = command["args"].split(",") if name in _LISTS else [command["args"]]
    implied = _IMPLIED_SUFFIXES[name]
    return command["head"] + ",".join(_followed(reference, folder, implied, source, target) for reference in references)


def _followed(reference: str, base: Path, implied: tuple[str, ...], source: Path, target: Path) -> str:
    """*reference* (resolved against *base*, possibly padded) pointing at *target* when it points at *source*, else unchanged."""
    leading, path, trailing = _PADDED.fullmatch(reference).groups()
    resolved = os.path.normpath(base / path)
    followed = Path(os.path.relpath(target, base)).as_posix()
    if resolved != str(source):
        if not _implies(resolved, implied, source):
            return reference
        if source.suffix.lower() != target.suffix.lower():
            return reference
        bare = followed.removesuffix(target.suffix)
        if _implies(os.path.normpath(base / bare), implied, target):
            followed = bare
    if path.startswith("./") and not followed.startswith("../"):
        followed = f"./{followed}"
    return leading + followed + trailing


def _implies(resolved: str, implied: tuple[str, ...], source: Path) -> bool:
    """Whether the extension-less *resolved* names *source* by one of the *implied* suffixes and no earlier suffix shadows it."""
    for suffix in implied:
        if resolved + suffix == str(source):
            return True
        if Path(resolved + suffix).is_file():
            return False
    return False
