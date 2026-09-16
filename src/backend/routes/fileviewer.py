"""Generic file-viewer routes backing the reusable ``FileViewer`` primitive.

The frontend ``FileViewer`` is handed a flat list of absolute filepaths and
previews each by kind: images and PDFs stream their bytes from ``/raw``;
markdown and unknown (treated as text) files read their content from ``/text``.

Every path is validated against the union of places the app legitimately knows
about — the file vaults, the document library, and any path referenced by a
project — so this generic reader can never be coaxed into serving an arbitrary
file off disk.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from backend.routes.deps import get_file_vault, get_project_store
from backend.routes.schemas import FileContent
from config import DIRECTORY_NOTES
from lib import document_library as lib_docs
from lib.file_vault import FileVault
from lib.project_store import ProjectStore

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fileviewer", tags=["fileviewer"])


def _resolve_allowed(path: str, vault: FileVault, store: ProjectStore) -> Path:
    try:
        return lib_docs.resolve_known_path(path, store=store, vault=vault, extra_roots=(DIRECTORY_NOTES,))
    except lib_docs.PathNotAllowed as exc:
        if exc.outside_roots:
            raise HTTPException(status_code=403, detail="Unknown file path") from exc
        raise HTTPException(status_code=404, detail="File not found") from exc


@router.get("/text", response_model=FileContent)
async def read_text(
    path: str = Query(...),
    vault: FileVault = Depends(get_file_vault),
    store: ProjectStore = Depends(get_project_store),
) -> FileContent:
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
) -> FileResponse:
    resolved = _resolve_allowed(path, vault, store)
    return FileResponse(resolved, media_type=lib_docs.mime_for(resolved), content_disposition_type="inline")
