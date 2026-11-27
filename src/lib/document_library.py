"""Helpers for document metadata and storage keys.

PDFs and MinerU output live in Postgres (``AssetStore``) and are mirrored to the
Nextcloud ``Documents`` tree as a copy the app never reads back. The only path
resolved off disk is a ``FileRoots`` reference: an external file with no database copy.

Pure path logic, MIME types come from ``lib.mime``. Project metadata belongs to ``ProjectStore``
and MinerU spawning to ``lib.mineru``. Staying free of route imports lets the ``files`` and
``documents`` routes share it without cycles.
"""

from pathlib import Path, PurePosixPath

from config import DOCUMENTS
from lib.data_store import InvalidStorageKey, validate_key
from lib.mime import mime_for


def document_meta(path: str | Path) -> dict[str, str]:
    """Describe a document for the API: path (logical key, or absolute for a FileRoots reference), name, MIME."""
    p = Path(path)
    return {"path": str(p), "name": p.name, "mime": mime_for(p)}


def storage_key(path: str | Path) -> str | None:
    """The logical store key for an app-owned path, or None when it is not one.

    Accepts an absolute path under ``DOCUMENTS`` or an already-logical key.
    Paths outside ``DOCUMENTS``, including FileRoots files, return
    None because they are live references with no stored copy.
    """
    text = str(path)
    if not text:
        return None
    if PurePosixPath(text).is_absolute():
        try:
            text = Path(text).resolve().relative_to(DOCUMENTS.resolve()).as_posix()
        except ValueError:
            return None
    try:
        return validate_key(text)
    except InvalidStorageKey:
        return None

