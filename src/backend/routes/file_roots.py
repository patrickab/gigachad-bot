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

from backend.identity import RequestIdentity, get_request_identity
from backend.routes.deps import get_asset_store, get_file_roots, get_project_store
from backend.routes.schemas import AttachResult, FileContent, FileListResponse, FileMeta
from lib import document_library as lib_docs
from lib.asset_store import AssetStore
from lib.attachment_materialize import materialize, store_library_pdf
from lib.data_store import StorageNotFoundError
from lib import git_diff
from lib.file_roots import FileRoots, list_directories
from lib.project_store import ProjectStore
from lib.storage_namespace import PDFS

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


class DirectoryListing(BaseModel):
    path: str
    parent: str | None
    directories: list[str]


@router.get("/files", response_model=RootFileListResponse)
def list_files(roots: FileRoots = Depends(get_file_roots)) -> RootFileListResponse:
    return RootFileListResponse(enabled=roots.enabled, files=roots.list_files())


@router.get("/tree", response_model=FileTreeResponse)
def list_tree(roots: FileRoots = Depends(get_file_roots)) -> FileTreeResponse:
    return FileTreeResponse(enabled=roots.enabled, tree=roots.tree())


@router.post("/roots", response_model=FileTreeResponse)
def add_root(body: RootBody, roots: FileRoots = Depends(get_file_roots)) -> FileTreeResponse:
    try:
        roots.add_root(body.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileTreeResponse(enabled=roots.enabled, tree=roots.tree())


@router.delete("/roots", response_model=FileTreeResponse)
def remove_root(path: str = Query(...), roots: FileRoots = Depends(get_file_roots)) -> FileTreeResponse:
    roots.remove_root(path)
    return FileTreeResponse(enabled=roots.enabled, tree=roots.tree())


@router.get("/browse", response_model=DirectoryListing)
def browse_directories(
    path: str = Query(default="~"),
    _identity: RequestIdentity = Depends(get_request_identity),
) -> DirectoryListing:
    """List the subdirectories of *path* on the backend host, to pick a new root."""
    try:
        directory, names = list_directories(path)
    except NotADirectoryError as exc:
        raise HTTPException(status_code=404, detail=f"Not a directory: {path}") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Cannot read {path}") from exc
    parent = str(directory.parent) if directory.parent != directory else None
    return DirectoryListing(path=str(directory), parent=parent, directories=names)


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
    assets: AssetStore = Depends(get_asset_store),
) -> FileListResponse:
    """Reference file-root files (folders expand recursively) from a project.

    - Text files are stored as their absolute path, so the project always sees the mounted file
      and removing it from the project never touches the filesystem
    - PDFs are promoted into the shared library, as on attach
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
                file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            key = str(file)
        store.add_file(body.slug, key)
        added.append(FileMeta(**lib_docs.document_meta(key)))
    return FileListResponse(documents=added)


class DiffFile(BaseModel):
    path: str
    status: git_diff.ChangeStatus


class DiffStatusResponse(BaseModel):
    is_git_repository: bool
    files: list[DiffFile]


class FileDiffResponse(BaseModel):
    supported: bool
    status: git_diff.ChangeStatus | None = None
    patch: str | None = None


def _git_directory(root: str, roots: FileRoots) -> Path:
    try:
        roots.root_of(root)
        directory = Path(root).expanduser().resolve()
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail=f"Not a directory: {root}")
    return directory


@router.get("/diff/files", response_model=DiffStatusResponse)
def list_changed_files(root: str = Query(...), roots: FileRoots = Depends(get_file_roots)) -> DiffStatusResponse:
    """Files of the git working tree at *root* that differ from HEAD."""
    files = git_diff.changed_files(_git_directory(root, roots))
    if files is None:
        return DiffStatusResponse(is_git_repository=False, files=[])
    return DiffStatusResponse(is_git_repository=True, files=[DiffFile(path=file.path, status=file.status) for file in files])


@router.get("/diff", response_model=FileDiffResponse)
def read_file_diff(root: str = Query(...), path: str = Query(...), roots: FileRoots = Depends(get_file_roots)) -> FileDiffResponse:
    """Unified patch of *path* (relative to *root*) against HEAD, `supported=false` for binary or oversized files."""
    patch = git_diff.file_patch(_git_directory(root, roots), path)
    if patch is None:
        return FileDiffResponse(supported=False)
    return FileDiffResponse(supported=True, status=patch[0], patch=patch[1])


class DiffContentResponse(BaseModel):
    base_content: str
    content: str


@router.get("/diff/content", response_model=DiffContentResponse)
def read_diff_content(root: str = Query(...), path: str = Query(...), roots: FileRoots = Depends(get_file_roots)) -> DiffContentResponse:
    """HEAD and working-tree text of *path* (relative to *root*) for the native Neovim diff."""
    try:
        base, content = git_diff.file_contents(_git_directory(root, roots), path)
    except git_diff.UnsupportedDiffContent as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return DiffContentResponse(base_content=base, content=content)
