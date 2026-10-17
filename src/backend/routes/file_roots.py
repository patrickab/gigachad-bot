"""File-root routes. The API owns every filesystem access.

The UI never touches a file root directly: it lists files (`/files`), previews one
(`/rendered`), writes one (`/file`), and *attaches* a chosen file (`/attach`).
Attach is a **live reference**: nothing is copied, the returned `path` is
stored on the Attachment and content is read from (and written back to) the
actual file.

Handlers are plain `def`: they walk directories and read files synchronously,
so FastAPI runs them in its threadpool instead of blocking the event loop.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.routes.deps import get_asset_store, get_document_store, get_file_roots, get_project_store
from backend.routes.schemas import AttachResult, FileContent, FileListResponse, FileMeta
from lib import document_library as lib_docs
from lib.asset_store import AssetStore
from lib.attachment_materialize import materialize, store_library_pdf
from lib.data_store import DataStore, StorageNotFoundError, write_text
from lib.file_roots import FileRoots
from lib.project_store import ProjectStore
from lib.storage_namespace import PDFS, project_document

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fileroots", tags=["fileroots"])


class RootFile(BaseModel):
    path: str
    name: str


class RootFileListResponse(BaseModel):
    enabled: bool
    files: list[RootFile]


class FileTreeNode(BaseModel):
    name: str
    path: str
    type: str
    project: str | None = None
    children: list["FileTreeNode"] | None = None


class FileTreeResponse(BaseModel):
    enabled: bool
    tree: list[FileTreeNode]


class RootBody(BaseModel):
    path: str
    project: str | None = None


@router.get("/files", response_model=RootFileListResponse)
def list_files(roots: FileRoots = Depends(get_file_roots)) -> RootFileListResponse:
    return RootFileListResponse(enabled=roots.enabled, files=roots.list_files())


@router.get("/project-documents", response_model=FileListResponse)
def project_documents(
    slug: str = Query(...),
    roots: FileRoots = Depends(get_file_roots),
    assets: AssetStore = Depends(get_asset_store),
) -> FileListResponse:
    """Files from file roots mounted to *slug*, shaped like project documents.

    A PDF whose library copy exists is reported at the library key, so the sidebar never lists it twice.
    """
    docs: list[FileMeta] = []
    for f in roots.list_files_for_project(slug):
        p = Path(f["path"])
        if p.suffix.lower() == ".pdf":
            library_key = f"{PDFS}/{p.name}"
            if assets.list(library_key):
                docs.append(FileMeta(**lib_docs.document_meta(library_key)))
                continue
        docs.append(FileMeta(path=str(p), name=p.name, mime=lib_docs.mime_for(p)))
    return FileListResponse(documents=docs)


@router.get("/tree", response_model=FileTreeResponse)
def list_tree(roots: FileRoots = Depends(get_file_roots)) -> FileTreeResponse:
    return FileTreeResponse(enabled=roots.enabled, tree=roots.tree())


@router.post("/roots", response_model=FileTreeResponse)
def add_root(body: RootBody, roots: FileRoots = Depends(get_file_roots)) -> FileTreeResponse:
    try:
        roots.add_root(body.path, project=body.project)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileTreeResponse(enabled=roots.enabled, tree=roots.tree())


@router.delete("/roots", response_model=FileTreeResponse)
def remove_root(path: str = Query(...), roots: FileRoots = Depends(get_file_roots)) -> FileTreeResponse:
    roots.remove_root(path)
    return FileTreeResponse(enabled=roots.enabled, tree=roots.tree())


class MountpointBody(BaseModel):
    path: str


@router.post("/mountpoints", response_model=FileTreeResponse)
def add_mountpoint(
    body: MountpointBody,
    root_path: str = Query(..., alias="root"),
    roots: FileRoots = Depends(get_file_roots),
) -> FileTreeResponse:
    try:
        roots.add_mountpoint(root_path, body.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileTreeResponse(enabled=roots.enabled, tree=roots.tree())


@router.delete("/mountpoints", response_model=FileTreeResponse)
def remove_mountpoint(
    root_path: str = Query(..., alias="root"),
    path: str = Query(...),
    roots: FileRoots = Depends(get_file_roots),
) -> FileTreeResponse:
    roots.remove_mountpoint(root_path, path)
    return FileTreeResponse(enabled=roots.enabled, tree=roots.tree())


@router.post("/file")
def write_file(body: FileContent, roots: FileRoots = Depends(get_file_roots)) -> dict[str, bool]:
    try:
        roots.write(body.path, body.content)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@router.get("/rendered", response_model=FileContent)
def read_rendered(path: str = Query(...), roots: FileRoots = Depends(get_file_roots)) -> FileContent:
    try:
        raw = roots.read(path)
        return FileContent(path=path, content=roots.resolve_wiki_content(raw, path))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/attach", response_model=AttachResult)
def attach_file(
    path: str = Query(...),
    roots: FileRoots = Depends(get_file_roots),
    assets: AssetStore = Depends(get_asset_store),
) -> AttachResult:
    """Attach a file as a live reference: validate the path, return current content.

    - Idempotent and copy-free, the frontend re-calls it at send time to refresh content
    - PDFs are promoted into the shared library (`PDFs/<name>`) on first attach, the library copy wins afterwards
    - The library key is also accepted as `path`, since that is what this endpoint returns for promoted PDFs
    """
    mime = lib_docs.mime_for(path)

    if Path(path).suffix.lower() == ".pdf":
        if Path(path).is_absolute():
            try:
                root_resolved = roots.resolve(path)
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            name = root_resolved.name
            # Library copy wins so a parsed PDF is never clobbered, else promote by name (no "<name> (n).pdf" dupes)
            if assets.list(f"{PDFS}/{name}"):
                content = assets.read(f"{PDFS}/{name}").content or b""
            else:
                content = root_resolved.read_bytes()
                store_library_pdf(assets, name, content)
        else:
            # Send-time refresh: `path` is the logical library key this
            # endpoint returned on first attach.
            name = Path(path).name
            try:
                content = assets.read(f"{PDFS}/{name}").content or b""
            except StorageNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        parsed = materialize(name, content, assets, enqueue_on_miss=True)
        return AttachResult(name=name, mime=mime, path=f"{PDFS}/{name}", parsedMd=parsed)

    try:
        root_resolved = roots.resolve(path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    result = AttachResult(name=root_resolved.name, mime=mime, path=str(root_resolved))
    try:
        result.content = roots.read(path)
    except UnicodeDecodeError:
        result.content = None
    return result


class AddToProjectBody(BaseModel):
    slug: str
    paths: list[str]


@router.post("/add-to-project", response_model=FileListResponse)
def add_to_project(
    body: AddToProjectBody,
    roots: FileRoots = Depends(get_file_roots),
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
    assets: AssetStore = Depends(get_asset_store),
) -> FileListResponse:
    """Add file-root files (folders expand recursively, valid extensions only) to a project's documents.

    - PDFs are promoted into the shared library, as on attach
    - Text files are copied into the project's documents, overwriting a same-named one
    - Files that are not valid UTF-8 text are skipped
    """
    try:
        store.list_files(body.slug)  # validates project exists
        files = roots.files_under(body.paths)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    added: list[FileMeta] = []
    for file in files:
        if file.suffix.lower() == ".pdf":
            content = file.read_bytes()
            key = store_library_pdf(assets, file.name, content).logical_path
            materialize(file.name, content, assets, enqueue_on_miss=True)
        else:
            try:
                text = file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            key = project_document(body.slug, file.name)
            write_text(docs, key, text)
        store.add_file(body.slug, key)
        added.append(FileMeta(**lib_docs.document_meta(key)))
    return FileListResponse(documents=added)
