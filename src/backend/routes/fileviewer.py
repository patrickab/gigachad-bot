"""Generic file-viewer routes backing the reusable ``FileViewer`` primitive.

The frontend ``FileViewer`` is handed a flat list of paths and previews each
by kind: images and PDFs stream their bytes from ``/raw``; markdown and
unknown (treated as text) files read their content from ``/text``. PDF
annotations live in a canvas keyed by the PDF's content hash, see ``/annotations``.

App-owned files (documents, canvases, PDFs, MinerU output) are read from the
request's storage. Postgres is authoritative, so serving from disk would
hand every caller the last filesystem write instead of the current one. Only
a live FileRoots reference, which was never stored in Postgres, is read off
disk, and only after storage has been checked and found not to own the path.
"""

import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse

from backend.routes.deps import get_asset_store, get_document_store, get_file_roots, http_errors
from backend.routes.schemas import FileContent
from lib import document_library as lib_docs
from lib.architecture_workspace import ArchitectureError, ArchitectureStore, split_source_path
from lib.asset_store import AssetStore
from lib.data_store import DataStore, StorageNotFoundError, read_text
from lib.file_roots import FileRoots
from lib.storage_namespace import pdf_annotations

router = APIRouter(prefix="/api/fileviewer", tags=["fileviewer"])


def _resolve_allowed(path: str, roots: FileRoots) -> Path:
    try:
        return lib_docs.resolve_known_path(path, roots=roots)
    except lib_docs.PathNotAllowed as exc:
        if exc.outside_roots:
            raise HTTPException(status_code=403, detail="Unknown file path") from exc
        raise HTTPException(status_code=404, detail="File not found") from exc


def _stored(path: str, docs: DataStore) -> str | None:
    """The stored logical key for *path*, or None when storage does not hold it."""
    key = lib_docs.storage_key(path)
    return key if key is not None and docs.exists(key) else None


def _stored_asset(path: str, assets: AssetStore) -> bytes | None:
    """Binary asset bytes for *path* (PDFs, MinerU output), or None when the asset store does not hold it.

    These live in the `assets` table, not `documents`, under the same key `document_meta` returns.
    """
    key = lib_docs.storage_key(path)
    if key is None:
        return None
    try:
        return assets.read(key).content
    except StorageNotFoundError:
        return None


@router.get("/text", response_model=FileContent)
async def read_text_content(
    path: str = Query(...),
    roots: FileRoots = Depends(get_file_roots),
    docs: DataStore = Depends(get_document_store),
) -> FileContent:
    if architecture := split_source_path(path):
        slug, relative = architecture
        with http_errors(((StorageNotFoundError, ArchitectureError), 404, "File not found")):
            return FileContent(path=path, content=ArchitectureStore.for_project(docs, slug, roots.roots()).read_source(relative))
    key = _stored(path, docs)
    if key is not None:
        with http_errors((StorageNotFoundError, 404, "File not found")):
            content, _ = read_text(docs, key)
        return FileContent(path=path, content=content)

    resolved = _resolve_allowed(path, roots)
    # Strict decode: a lossy one would let the editor save replacement characters over a binary file.
    with http_errors((UnicodeDecodeError, 415, "Not a UTF-8 text file"), (OSError, 500, None)):
        content = resolved.read_text(encoding="utf-8")
    return FileContent(path=path, content=content)


def _raw_source(path: str, roots: FileRoots, docs: DataStore, assets: AssetStore) -> tuple[bytes, str] | Path:
    """Stored bytes with their MIME type, or the allowed host path when storage does not own *path*."""
    key = _stored(path, docs)
    if key is not None:
        with http_errors((StorageNotFoundError, 404, "File not found")):
            content, _ = docs.read_bytes(key)
        return content, lib_docs.mime_for(key)
    asset_content = _stored_asset(path, assets)
    if asset_content is not None:
        return asset_content, lib_docs.mime_for(path)
    return _resolve_allowed(path, roots)


@router.get("/raw")
async def read_raw(
    path: str = Query(...),
    roots: FileRoots = Depends(get_file_roots),
    docs: DataStore = Depends(get_document_store),
    assets: AssetStore = Depends(get_asset_store),
) -> Response:
    source = _raw_source(path, roots, docs, assets)
    if isinstance(source, Path):
        return FileResponse(source, media_type=lib_docs.mime_for(source), content_disposition_type="inline")
    content, media_type = source
    return Response(content=content, media_type=media_type, headers={"Content-Disposition": "inline"})


@router.post("/annotations")
async def open_pdf_annotations(
    path: str = Query(...),
    roots: FileRoots = Depends(get_file_roots),
    docs: DataStore = Depends(get_document_store),
    assets: AssetStore = Depends(get_asset_store),
) -> dict[str, str]:
    """Return the annotation canvas key of the PDF at *path*, creating the empty canvas on first use.

    - Keyed by the SHA-256 of the PDF bytes: every copy shares annotations, renames keep them
    - Changed PDF content gets a new, empty canvas
    """
    source = _raw_source(path, roots, docs, assets)
    if isinstance(source, Path):
        with http_errors((OSError, 404, "File not found")), source.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
    else:
        digest = hashlib.sha256(source[0]).hexdigest()
    key = pdf_annotations(digest)
    if not docs.exists(key):
        docs.write_bytes(key, b"")
    return {"key": key}
