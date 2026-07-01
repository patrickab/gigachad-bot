"""Generic file-viewer routes backing the reusable ``FileViewer`` primitive.

The frontend ``FileViewer`` is handed a flat list of absolute filepaths and
previews each by kind: images and PDFs stream their bytes from ``/raw``;
markdown and unknown (treated as text) files read their content from ``/text``.

Every path is validated against the union of places the app legitimately knows
about — the Obsidian vault, the document library, and any path referenced by a
project — so this generic reader can never be coaxed into serving an arbitrary
file off disk.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.routes.deps import get_obsidian_vault, get_project_store
from lib import document_library as lib_docs
from lib.obsidian_vault import ObsidianVault
from lib.project_store import ProjectStore

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fileviewer", tags=["fileviewer"])


class FileTextContent(BaseModel):
    path: str
    content: str


def _resolve_allowed(path: str, vault: ObsidianVault, store: ProjectStore) -> Path:
    resolved = Path(path).expanduser().resolve()
    library = lib_docs.LIBRARY_DIR.resolve()
    allowed = (
        resolved.is_relative_to(library)
        or vault.contains(resolved)
        or str(resolved) in {str(Path(p).expanduser().resolve()) for p in store.list_all_files()}
    )
    if not allowed:
        raise HTTPException(status_code=403, detail="Unknown file path")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return resolved


@router.get("/text", response_model=FileTextContent)
async def read_text(
    path: str = Query(...),
    vault: ObsidianVault = Depends(get_obsidian_vault),
    store: ProjectStore = Depends(get_project_store),
) -> FileTextContent:
    resolved = _resolve_allowed(path, vault, store)
    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return FileTextContent(path=path, content=content)


@router.get("/raw")
async def read_raw(
    path: str = Query(...),
    vault: ObsidianVault = Depends(get_obsidian_vault),
    store: ProjectStore = Depends(get_project_store),
) -> FileResponse:
    resolved = _resolve_allowed(path, vault, store)
    return FileResponse(resolved, media_type=lib_docs.mime_for(resolved), content_disposition_type="inline")
