"""Shared attach/parse core for the documents/files/file_vaults routes:
cached-MinerU-markdown lookup, background-extraction enqueue, text reads.
Route-specific behavior (vault promotion, library organizing, sync parsing)
stays in the routes."""

from dataclasses import dataclass
from pathlib import Path

from config import DIRECTORY_OUTPUT_MINERU
from lib import extract_queue
from lib.document_library import mime_for


def mineru_cache_path(stem_or_path: str | Path) -> Path:
    """Path to the cached MinerU markdown for a PDF, given its stem or the PDF path itself."""
    # Strip only a .pdf suffix — Path.stem would also truncate stems containing dots ("paper v1.2").
    name = Path(stem_or_path).name
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    return DIRECTORY_OUTPUT_MINERU / f"{name}.md"


@dataclass
class Materialized:
    mime: str
    parsed_md: str | None = None
    content: str | None = None


def materialize(path: Path, *, enqueue_on_miss: bool = True) -> Materialized:
    """Resolve what an attach/parse endpoint needs to respond with for *path*.

    PDFs: read the cached MinerU markdown if present. On a cache miss, queue
    background extraction unless *enqueue_on_miss* is False — pass False when
    the caller handles the miss itself (e.g. parsing synchronously instead).
    Any other text/* mime: read file content, returning None on decode error.
    """
    mime = mime_for(path)
    result = Materialized(mime=mime)

    if path.suffix.lower() == ".pdf":
        cached_md = mineru_cache_path(path)
        if cached_md.is_file():
            result.parsed_md = cached_md.read_text(encoding="utf-8")
        elif enqueue_on_miss:
            extract_queue.enqueue(path)
    elif mime.startswith("text/"):
        try:
            result.content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            result.content = None

    return result
