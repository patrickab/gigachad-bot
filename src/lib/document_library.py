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

from collections.abc import Iterable
import logging
import mimetypes
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

from config import (
    DIRECTORY_CHAT_HISTORIES,
    DIRECTORY_OUTPUT_ARCHITECTURE_GRAPHS,
    DIRECTORY_OUTPUT_MINERU,
    DIRECTORY_OUTPUT_PDF,
)

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


def resolve_known_path(
    path: str | Path,
    *,
    store: "ProjectStore",
    vault: "FileVault | None" = None,
    extra_roots: Iterable[str | Path] = (),
) -> Path:
    """Resolve a user-supplied path, rejecting anything outside the allowed roots.

    Allowed are the document library, every path *store* already references and
    the canonical architecture-graph files, plus whatever the caller opts into:
    *extra_roots* directory trees and, when given, anything *vault* contains.

    Raises ``PathNotAllowed`` so each caller maps the two rejection reasons onto
    its own status codes and wording.
    """
    resolved = Path(path).expanduser().resolve()
    roots = [LIBRARY_DIR.resolve(), *(Path(root).resolve() for root in extra_roots)]
    graphs = DIRECTORY_OUTPUT_ARCHITECTURE_GRAPHS.resolve()
    allowed = (
        any(resolved.is_relative_to(root) for root in roots)
        # Graph drafts are deliberately not generic documents: they stay
        # proposal state until an explicit accept publishes them as canonical.
        or (resolved.parent == graphs and resolved.name.endswith(".architecture.yaml"))
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


def backfill_pdf_library() -> int:
    """Recover chat-attached PDFs that never reached the library.

    Historically only the ``/api/mineru`` routes copied raw PDFs into the
    library; PDFs attached to chats (parsed via ``/api/files``) stayed only in
    their per-chat ``_uploads`` dir. This one-shot, idempotent pass copies each
    unique PDF filename from any ``_uploads`` directory into the library — but
    only when its parsed ``<stem>.md`` already exists in the MinerU cache, so we
    never resurrect a PDF we never actually parsed.
    """
    if not DIRECTORY_CHAT_HISTORIES.exists():
        return 0
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    seen: set[str] = set()
    for pdf in DIRECTORY_CHAT_HISTORIES.rglob("*.pdf"):
        if "_uploads" not in pdf.parts or not pdf.is_file():
            continue
        if pdf.name in seen:
            continue
        seen.add(pdf.name)
        dest = LIBRARY_DIR / pdf.name
        if dest.exists():
            continue
        if not (DIRECTORY_OUTPUT_MINERU / f"{pdf.stem}.md").is_file():
            continue
        shutil.copy2(pdf, dest)
        copied += 1
    if copied:
        log.info("Document library backfill: recovered %d PDF(s) into %s", copied, LIBRARY_DIR)
    return copied
