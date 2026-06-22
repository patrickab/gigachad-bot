"""The document library — a tidy, central home for vault-tree documents.

Every document a project references lives (by convention) under
``DIRECTORY_OUTPUT_PDF``, the same directory MinerU already organizes parsed
PDFs into. Project ``files`` lists in ``project.json`` hold *arbitrary absolute
paths* so documents may come from arbitrary sources, but the add/upload flow
copies them here so the library stays self-contained.

This module is pure I/O over the filesystem — it never touches ``project.json``
(that is ``ProjectStore``'s seam) and never spawns MinerU (that lives in the
``mineru`` route). Keeping it dependency-free of the route layer lets both
``files`` and ``documents`` routes reuse it without import cycles.
"""

import logging
import mimetypes
from pathlib import Path
import shutil

from config import DIRECTORY_CHAT_HISTORIES, DIRECTORY_OUTPUT_MINERU, DIRECTORY_OUTPUT_PDF

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
