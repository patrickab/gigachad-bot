"""MIME types and image suffixes of stored and referenced document paths."""

import mimetypes
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"})


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
