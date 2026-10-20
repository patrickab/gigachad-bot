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
from lib.attachment_materialize import materialize, sync_library_pdf
from lib.data_store import StorageNotFoundError
from lib import git_diff, git_worktree
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
    mount: str | None = None
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
def list_tree(worktree: list[str] = Query(default=[]), roots: FileRoots = Depends(get_file_roots)) -> FileTreeResponse:
    """Every root's tree, a root shows one of its *worktree* paths instead of its own files when listed."""
    return FileTreeResponse(enabled=roots.enabled, tree=roots.tree(worktree))


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
    - A file that moved inside its mount is followed, the returned `path` is where it is now
    - PDFs are synced into the shared library (`PDFs/<name>`), which drops the cached extraction when the file changed
    - A library key (`PDFs/<name>`, what older chats stored) is still accepted and reads the library copy
    """
    if Path(path).suffix.lower() == ".pdf" and not Path(path).is_absolute():
        name = Path(path).name
        try:
            content = assets.read(f"{PDFS}/{name}").content or b""
        except StorageNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return AttachResult(name=name, mime=lib_docs.mime_for(path), path=f"{PDFS}/{name}", parsedMd=materialize(name, content, assets, enqueue_on_miss=True))

    try:
        resolved = roots.resolve_reference(path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    result = AttachResult(name=resolved.name, mime=lib_docs.mime_for(resolved), path=str(resolved))
    if resolved.suffix.lower() == ".pdf":
        content = resolved.read_bytes()
        sync_library_pdf(assets, resolved.name, content)
        result.parsedMd = materialize(resolved.name, content, assets, enqueue_on_miss=True)
        return result
    try:
        result.content = resolved.read_text(encoding="utf-8")
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

    - Every file is stored as its absolute path, so the project always sees the mounted file
      and removing it from the project never touches the filesystem
    - PDFs are also synced into the shared library and queued for extraction, as on attach
    - Files that are neither PDF nor valid UTF-8 text are skipped
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
            sync_library_pdf(assets, file.name, content)
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


class WorktreeEntry(BaseModel):
    path: str
    branch: str | None
    is_main: bool
    changed_files: int
    ahead: int
    behind: int


class WorktreeListResponse(BaseModel):
    is_clone: bool
    default_base: str | None = None
    branches: list[str] = []
    worktrees: list[WorktreeEntry] = []


class CreateWorktreeBody(BaseModel):
    root: str
    name: str
    base: str


def _clone_root(root: str, roots: FileRoots) -> Path:
    """The registered mount *root*, 403 for anything else."""
    directory = Path(root).expanduser().resolve()
    if str(directory) not in roots.roots():
        raise HTTPException(status_code=403, detail=f"Not a file mount: {root}")
    return directory


def _git_refusal(exc: git_worktree.WorktreeError) -> HTTPException:
    conflict = isinstance(exc, (git_worktree.WorktreeDirty, git_worktree.BranchNotMerged))
    return HTTPException(status_code=409 if conflict else 400, detail=str(exc))


@router.get("/worktrees", response_model=WorktreeListResponse)
def list_worktrees(root: str = Query(...), roots: FileRoots = Depends(get_file_roots)) -> WorktreeListResponse:
    """Worktrees of the clone mounted at *root*, each with its uncommitted files and commits ahead/behind the base."""
    repository = _clone_root(root, roots)
    if not git_worktree.is_clone(repository):
        return WorktreeListResponse(is_clone=False)
    try:
        base = git_worktree.default_base_branch(repository)
        entries = [
            WorktreeEntry(path=worktree.path, branch=worktree.branch, is_main=worktree.is_main, **vars(git_worktree.status(Path(worktree.path), base)))
            for worktree in git_worktree.list_worktrees(repository)
        ]
        return WorktreeListResponse(is_clone=True, default_base=base, branches=git_worktree.branches(repository), worktrees=entries)
    except git_worktree.WorktreeError as exc:
        raise _git_refusal(exc) from exc


@router.post("/worktrees", response_model=RootFile)
def create_worktree(body: CreateWorktreeBody, roots: FileRoots = Depends(get_file_roots)) -> RootFile:
    """Pull *base* and branch a new worktree off it, `name` is the branch."""
    repository = _clone_root(body.root, roots)
    try:
        worktree = git_worktree.create_worktree(repository, body.name, body.base)
    except git_worktree.WorktreeError as exc:
        raise _git_refusal(exc) from exc
    return RootFile(path=worktree.path, name=worktree.branch or Path(worktree.path).name)


@router.delete("/worktrees")
def remove_worktree(
    root: str = Query(...), path: str = Query(...), force: bool = Query(default=False), roots: FileRoots = Depends(get_file_roots),
) -> dict[str, bool]:
    """Remove a worktree, keeping its branch. 409 when it holds uncommitted changes and *force* is off."""
    try:
        git_worktree.remove_worktree(_clone_root(root, roots), path, force=force)
    except git_worktree.WorktreeError as exc:
        raise _git_refusal(exc) from exc
    return {"ok": True}


@router.delete("/worktrees/branch")
def delete_worktree_branch(
    root: str = Query(...), branch: str = Query(...), force: bool = Query(default=False), roots: FileRoots = Depends(get_file_roots),
) -> dict[str, bool]:
    """Delete a local branch. 409 when it is not fully merged and *force* is off."""
    try:
        git_worktree.delete_branch(_clone_root(root, roots), branch, force=force)
    except git_worktree.WorktreeError as exc:
        raise _git_refusal(exc) from exc
    return {"ok": True}
