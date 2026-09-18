"""The document library — a tidy, central home for database-owned artifacts.

PDFs and MinerU output are Postgres rows (``AssetStore``) mirrored to the
Nextcloud ``Documents`` tree as a convenience copy for other tools; the app
itself never reads that mirror back. The one path this module still resolves
off disk is a ``FileVault`` reference — an external file the user pointed the
app at directly, which was never stored in Postgres and has no database copy
to read instead.

This module is pure I/O over paths and mime types — it never touches
``project.json`` (that is ``ProjectStore``'s seam) and never spawns MinerU
(that lives in ``lib.mineru``). Keeping it dependency-free of the route layer
lets both ``files`` and ``documents`` routes reuse it without import cycles.
"""

import logging
import mimetypes
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from config import DOCUMENTS
from lib.data_store import InvalidStorageKey, validate_key

if TYPE_CHECKING:
    from lib.file_vault import FileVault

log = logging.getLogger(__name__)

_TEXT_EXTS = {".canvas", ".csv", ".json", ".yaml", ".yml", ".xml", ".toml", ".ini", ".cfg", ".conf", ".log", ".md", ".rst", ".tex", ".txt", ".svg"}


def mime_for(path: str | Path) -> str:
    """Best-effort MIME for a document path (suffix-driven, PDF-aware)."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    guessed, _ = mimetypes.guess_type(p.name)
    if guessed:
        return guessed
    if suffix in _TEXT_EXTS:
        return "text/markdown" if suffix == ".md" else "text/plain"
    return "application/octet-stream"


def document_meta(path: str | Path) -> dict[str, str]:
    """Describe a document for the API: its path (a database-native logical
    key, or an absolute path for a live FileVault reference), display name, and MIME."""
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

    Callers hold one of two spellings for the same file: an absolute path (what the
    filesystem branch of a route records) or an already-logical key (what the storage
    branch records). Both map onto the key the DataStore uses. Vault files and anything
    outside *root* (``DOCUMENTS`` by default) return None — those are live references,
    never stored copies.
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


def resolve_known_path(path: str | Path, *, vault: "FileVault | None") -> Path:
    """Resolve a FileVault-referenced path, rejecting anything outside it.

    The only paths this ever resolves off disk are live FileVault references:
    external files the user pointed the app at directly, never stored in
    Postgres. Every app-owned artifact (PDFs, MinerU output, documents,
    canvases) is read from storage before a caller ever reaches this function.

    Raises ``PathNotAllowed`` so each caller maps the two rejection reasons
    onto its own status codes and wording.
    """
    resolved = Path(path).expanduser().resolve()
    if vault is None or not vault.contains(resolved):
        raise PathNotAllowed(outside_roots=True)
    if not resolved.is_file():
        raise PathNotAllowed(outside_roots=False)
    return resolved
