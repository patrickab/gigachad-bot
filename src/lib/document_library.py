"""Helpers for document MIME types, storage keys, and FileRoots path checks.

PDFs and MinerU output live in Postgres (``AssetStore``) and are mirrored to the
Nextcloud ``Documents`` tree as a copy the app never reads back. The only path
resolved off disk is a ``FileRoots`` reference: an external file with no database copy.

Pure path and MIME logic. Project metadata belongs to ``ProjectStore`` and MinerU
spawning to ``lib.mineru``. Staying free of route imports lets the ``files`` and
``documents`` routes share it without cycles.
"""

import logging
import mimetypes
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from config import DOCUMENTS
from lib.data_store import InvalidStorageKey, validate_key

if TYPE_CHECKING:
    from lib.file_roots import FileRoots

log = logging.getLogger(__name__)


def mime_for(path: str | Path) -> str:
    """Best-effort MIME for a document path: PDF and registered types by suffix, everything else is text."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    guessed, _ = mimetypes.guess_type(p.name)
    if guessed:
        return guessed
    return "text/markdown" if suffix == ".md" else "text/plain"


def document_meta(path: str | Path) -> dict[str, str]:
    """Describe a document for the API: path (logical key, or absolute for a FileRoots reference), name, MIME."""
    p = Path(path)
    return {"path": str(p), "name": p.name, "mime": mime_for(p)}


class PathNotAllowed(Exception):
    """A user-supplied path is not one the app is willing to serve.

    ``outside_roots`` separates the two rejections callers report differently:
    True when the path is under none of the allowed roots, False when the root
    is fine but no file actually lives there.
    """

    def __init__(self, *, outside_roots: bool) -> None:
        super().__init__("path outside the allowed roots" if outside_roots else "no file at path")
        self.outside_roots = outside_roots


def storage_key(path: str | Path, *, root: str | Path | None = None) -> str | None:
    """The logical store key for an app-owned path, or None when it is not one.

    Accepts an absolute path under *root* (``DOCUMENTS`` by default) or an
    already-logical key. Paths outside *root*, including FileRoots files, return
    None because they are live references with no stored copy.
    """
    text = str(path)
    if not text:
        return None
    if PurePosixPath(text).is_absolute():
        try:
            text = Path(text).resolve().relative_to(Path(root or DOCUMENTS).resolve()).as_posix()
        except ValueError:
            return None
    try:
        return validate_key(text)
    except InvalidStorageKey:
        return None


def resolve_known_path(path: str | Path, *, roots: "FileRoots | None") -> Path:
    """Resolve a live FileRoots reference to a real file.

    App-owned artifacts are read from storage, never through this function.
    Raises ``PathNotAllowed`` so each caller maps the two reasons to its own status.
    """
    resolved = Path(path).expanduser().resolve()
    if roots is None or not roots.contains(resolved):
        raise PathNotAllowed(outside_roots=True)
    if not resolved.is_file():
        raise PathNotAllowed(outside_roots=False)
    return resolved
