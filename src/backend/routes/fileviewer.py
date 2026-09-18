"""Generic file-viewer routes backing the reusable ``FileViewer`` primitive.

The frontend ``FileViewer`` is handed a flat list of absolute filepaths and
previews each by kind: images and PDFs stream their bytes from ``/raw``;
markdown and unknown (treated as text) files read their content from ``/text``.

App-owned files are read from the request's storage: once Postgres is
authoritative the on-disk ``Documents`` tree is only a replica, so serving from
disk would hand every caller the last filesystem write instead of the current
document. Vault files and other live references have no stored copy and keep
being read in place.

Every path is validated against the union of places the app legitimately knows
about — the file vaults, the document library, and any path referenced by a
project — so this generic reader can never be coaxed into serving an arbitrary
file off disk.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse

from backend.routes.deps import get_asset_store, get_document_store, get_file_vault, get_project_store
from backend.routes.schemas import FileContent
from lib import document_library as lib_docs
from lib.asset_store import AssetStore
from lib.data_store import DataStore, StorageNotFoundError, read_text
from lib.file_vault import FileVault
from lib.project_store import ProjectStore

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fileviewer", tags=["fileviewer"])


def _resolve_allowed(path: str, vault: FileVault, store: ProjectStore) -> Path:
    try:
        return lib_docs.resolve_known_path(path, store=store, vault=vault)
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

    PDFs and MinerU assets live in the ``assets`` table, not ``documents``: their logical
    path (e.g. ``PDFs/foo.pdf``) is the same key ``document_meta`` hands back to the
    frontend, so it must be tried here too, not just the on-disk mirror.
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
    vault: FileVault = Depends(get_file_vault),
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
) -> FileContent:
    key = _stored(path, docs)
    if key is not None:
        try:
            content, _ = read_text(docs, key)
        except StorageNotFoundError as exc:
            raise HTTPException(status_code=404, detail="File not found") from exc
        return FileContent(path=path, content=content)

    resolved = _resolve_allowed(path, vault, store)
    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return FileContent(path=path, content=content)


@router.get("/raw")
async def read_raw(
    path: str = Query(...),
    vault: FileVault = Depends(get_file_vault),
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
    assets: AssetStore = Depends(get_asset_store),
) -> Response:
    key = _stored(path, docs)
    if key is not None:
        try:
            content, _ = docs.read_bytes(key)
        except StorageNotFoundError as exc:
            raise HTTPException(status_code=404, detail="File not found") from exc
        return Response(
            content=content,
            media_type=lib_docs.mime_for(key),
            headers={"Content-Disposition": "inline"},
        )

    asset_content = _stored_asset(path, assets)
    if asset_content is not None:
        return Response(
            content=asset_content,
            media_type=lib_docs.mime_for(path),
            headers={"Content-Disposition": "inline"},
        )

    resolved = _resolve_allowed(path, vault, store)
    return FileResponse(resolved, media_type=lib_docs.mime_for(resolved), content_disposition_type="inline")
