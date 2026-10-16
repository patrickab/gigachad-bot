"""File-root routes — the API owns every filesystem access.

The UI never touches a file root directly: it lists files (`/files`), previews one
(`/rendered`), writes one (`/file`), and *attaches* a chosen file (`/attach`).
Attach is a **live reference** — nothing is copied; the returned `path` is
stored on the Attachment and content is read from (and written back to) the
actual file.
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
async def list_files(roots: FileRoots = Depends(get_file_roots)) -> RootFileListResponse:
    return RootFileListResponse(enabled=roots.enabled, files=roots.list_files())


@router.get("/project-documents", response_model=FileListResponse)
async def project_documents(
    slug: str = Query(...),
    roots: FileRoots = Depends(get_file_roots),
    assets: AssetStore = Depends(get_asset_store),
) -> FileListResponse:
    """Files from file roots mounted to *slug*, shaped like project documents.

    Surfaced so the chat sidebar can list mounted-root files alongside library
    documents. They attach as live references (no copy). A file root PDF whose
    cloud-library copy already exists is reported at the *library* key — the
    parsed cloud copy takes precedence over the file root original, so the sidebar
    never lists the same PDF twice (file root row + library row).
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
async def list_tree(roots: FileRoots = Depends(get_file_roots)) -> FileTreeResponse:
    return FileTreeResponse(enabled=roots.enabled, tree=roots.tree())


@router.post("/roots", response_model=FileTreeResponse)
async def add_root(body: RootBody, roots: FileRoots = Depends(get_file_roots)) -> FileTreeResponse:
    try:
        roots.add_root(body.path, project=body.project)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileTreeResponse(enabled=roots.enabled, tree=roots.tree())


@router.delete("/roots", response_model=FileTreeResponse)
async def remove_root(path: str = Query(...), roots: FileRoots = Depends(get_file_roots)) -> FileTreeResponse:
    roots.remove_root(path)
    return FileTreeResponse(enabled=roots.enabled, tree=roots.tree())


class MountpointBody(BaseModel):
    path: str


@router.post("/mountpoints", response_model=FileTreeResponse)
async def add_mountpoint(
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
async def remove_mountpoint(
    root_path: str = Query(..., alias="root"),
    path: str = Query(...),
    roots: FileRoots = Depends(get_file_roots),
) -> FileTreeResponse:
    roots.remove_mountpoint(root_path, path)
    return FileTreeResponse(enabled=roots.enabled, tree=roots.tree())


@router.post("/file")
async def write_file(body: FileContent, roots: FileRoots = Depends(get_file_roots)) -> dict[str, bool]:
    try:
        roots.write(body.path, body.content)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@router.get("/rendered", response_model=FileContent)
async def read_rendered(path: str = Query(...), roots: FileRoots = Depends(get_file_roots)) -> FileContent:
    try:
        raw = roots.read(path)
        return FileContent(path=path, content=roots.resolve_wiki_content(raw, path))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/attach", response_model=AttachResult)
async def attach_file(
    path: str = Query(...),
    roots: FileRoots = Depends(get_file_roots),
    assets: AssetStore = Depends(get_asset_store),
) -> AttachResult:
    """Attach as a live reference: validate the path, return current content.

    Idempotent and copy-free — the frontend calls this again at send time to
    refresh content (and pick up a finished PDF extraction).

    PDFs are the one exception to the copy-free rule: a file root PDF is *promoted*
    into the shared Postgres library (``PDFs/<name>``) on first attach and the
    library copy takes precedence thereafter. This dedupes against any
    same-named PDF already parsed by MinerU, so a file root never shadows the
    canonical library copy. The library's logical key is also accepted here
    (it is what this endpoint itself returns), so the send-time refresh — which
    re-calls this endpoint with whatever path it was last given — keeps working
    for promoted PDFs. Non-PDF file-root files stay pure live references.
    """
    mime = lib_docs.mime_for(path)

    if Path(path).suffix.lower() == ".pdf":
        if Path(path).is_absolute():
            try:
                root_resolved = roots.resolve(path)
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            name = root_resolved.name
            # Cloud copy takes precedence if it already exists (don't clobber a
            # possibly-parsed library PDF with a same-named file-root file). Otherwise
            # promote the file root PDF into the library — filename is identity, so
            # this is an overwrite-by-name, never a "<name> (n).pdf" dupe.
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
async def add_to_project(
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
