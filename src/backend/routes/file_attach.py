"""Attaching file-root files to chats and projects as live references.

Attach copies nothing: the returned `path` is stored on the Attachment and content is
read from (and written back to) the actual file. Handlers are plain `def` because they
read files synchronously, so FastAPI runs them in its threadpool.
"""

from pathlib import Path

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from backend.http_plumbing import http_errors
from backend.routes.deps import get_asset_store, get_file_roots, get_project_store
from backend.routes.schemas import FileListResponse, FileMeta, file_meta
from lib.asset_store import AssetStore
from lib.attachments import AttachResult, attach_host_file, attach_library_pdf, kind_of, read_text_or_none, sync_pdf
from lib.data_store import StorageNotFoundError
from lib.file_roots import FileRoots
from lib.project_store import ProjectStore

router = APIRouter(prefix="/api/fileroots", tags=["fileroots"])


class AddToProjectBody(BaseModel):
    slug: str
    paths: list[str]


@router.post("/attach", response_model=AttachResult)
def attach_file(
    path: str = Query(...),
    roots: FileRoots = Depends(get_file_roots),
    assets: AssetStore = Depends(get_asset_store),
) -> AttachResult:
    """Attach a file as a live reference: validate the path, return current content.

    - Idempotent and copy-free, the frontend re-calls it at send time to refresh content
    - A file that moved inside its mount is followed, the returned `path` is where it is now
    - PDFs are synced into the shared library (`PDFs/<name>`), which drops the cached extraction when the file changed
    - A library key (`PDFs/<name>`, what older chats stored) is still accepted and reads the library copy
    """
    if kind_of(path) == "pdf" and not Path(path).is_absolute():
        with http_errors((StorageNotFoundError, 404, None)):
            return attach_library_pdf(assets, path)

    with http_errors(((FileNotFoundError, ValueError), 404, None)):
        resolved = roots.resolve_reference(path)
    return attach_host_file(assets, resolved)


@router.post("/add-to-project", response_model=FileListResponse)
def add_to_project(
    body: AddToProjectBody,
    roots: FileRoots = Depends(get_file_roots),
    store: ProjectStore = Depends(get_project_store),
    assets: AssetStore = Depends(get_asset_store),
) -> FileListResponse:
    """Reference file-root files (folders expand recursively) from a project.

    - Every file is stored as its absolute path, so the project always sees the mounted file
      and removing it from the project never touches the filesystem
    - PDFs are also synced into the shared library and queued for extraction, as on attach
    - Files that are neither PDF nor valid UTF-8 text are skipped
    """
    with http_errors((FileNotFoundError, 404, None), (ValueError, 400, None)):
        store.require(body.slug)
        files = roots.files_under(body.paths)

    added: list[FileMeta] = []
    for file in files:
        if kind_of(file.name) == "pdf":
            sync_pdf(assets, file.name, file.read_bytes())
        elif read_text_or_none(file) is None:
            continue
        key = str(file)
        store.add_file(body.slug, key)
        added.append(file_meta(key))
    return FileListResponse(documents=added)
