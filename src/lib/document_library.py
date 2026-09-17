"""The document library — a tidy, central home for vault-tree documents.

Every document a project references lives (by convention) under
``DIRECTORY_OUTPUT_PDF``, the same directory MinerU already organizes parsed
PDFs into. Project ``files`` lists in ``project.json`` hold *arbitrary absolute
paths* so documents may come from arbitrary sources, but the add/upload flow
copies them here so the library stays self-contained.

This module is pure I/O over the filesystem — it never touches ``project.json``
(that is ``ProjectStore``'s seam) and never spawns MinerU (that lives in
``lib.mineru``). Keeping it dependency-free of the route layer lets both
``files`` and ``documents`` routes reuse it without import cycles.
"""

import logging
import mimetypes
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from config import DIRECTORY_OUTPUT_PDF, DOCUMENTS
from lib.data_store import InvalidStorageKey, validate_key

if TYPE_CHECKING:
    from lib.file_vault import FileVault
    from lib.project_store import ProjectStore

log = logging.getLogger(__name__)

LIBRARY_DIR = DIRECTORY_OUTPUT_PDF

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
    """Describe a document for the API: absolute path, display name, MIME."""
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


def resolve_known_path(
    path: str | Path,
    *,
    store: "ProjectStore",
    vault: "FileVault | None" = None,
) -> Path:
    """Resolve a user-supplied path, rejecting anything outside the allowed roots.

    Allowed are the document library, every path *store* already references, and,
    when given, anything *vault* contains. Architecture graphs never reach here:
    they are Postgres-only and resolved by the caller's storage-backed branch first.

    Raises ``PathNotAllowed`` so each caller maps the two rejection reasons onto
    its own status codes and wording.
    """
    resolved = Path(path).expanduser().resolve()
    allowed = (
        resolved.is_relative_to(LIBRARY_DIR.resolve())
        or (vault is not None and vault.contains(resolved))
        or str(resolved) in {str(Path(known).expanduser().resolve()) for known in store.list_all_files()}
    )
    if not allowed:
        raise PathNotAllowed(outside_roots=True)
    if not resolved.is_file():
        raise PathNotAllowed(outside_roots=False)
    return resolved


def organize_file(src: Path) -> Path:
    """Copy *src* into the library under its own name, overwriting any existing
    file with that name. Filename is the document's identity, so re-promoting the
    same PDF refreshes the library copy instead of spawning ``<name> (n).pdf``.
    """
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    dest = LIBRARY_DIR / src.name
    if src.resolve() != dest.resolve():
        dest.write_bytes(src.read_bytes())
    return dest

