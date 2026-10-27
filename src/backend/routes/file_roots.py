"""File-root routes: mounts, their trees, host browsing, and reading and writing mounted files. The API owns every filesystem access.

The UI never touches a file root directly: it lists files (`/files`), previews one
(`/rendered`) and writes one (`/file`). Attaching lives in `file_attach`, git diffs
and worktrees in `file_git`, all under the same `/api/fileroots` prefix.

Handlers are plain `def`: they walk directories and read files synchronously,
so FastAPI runs them in its threadpool instead of blocking the event loop.
"""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from backend.http_plumbing import http_errors
from backend.identity import host_access_only
from backend.routes.deps import get_file_roots
from backend.routes.schemas import FileContent, RootFile
from lib import git_worktree
from lib.file_roots import FileRoots
from lib.git import GitError
from lib.host_access import list_directories

router = APIRouter(prefix="/api/fileroots", tags=["fileroots"])


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


class CloneBody(BaseModel):
    url: str


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


# Mounting, cloning and browsing act on the machine owner's host: a mounted root is readable in full.
_MOUNTING = host_access_only("mounting host folders")


@router.post("/roots", response_model=FileTreeResponse, dependencies=[_MOUNTING])
def add_root(body: RootBody, roots: FileRoots = Depends(get_file_roots)) -> FileTreeResponse:
    with http_errors((ValueError, 400, None)):
        roots.add_root(body.path)
    return FileTreeResponse(enabled=roots.enabled, tree=roots.tree())


@router.post("/clone", response_model=FileTreeResponse, dependencies=[_MOUNTING])
def clone_root(body: CloneBody, roots: FileRoots = Depends(get_file_roots)) -> FileTreeResponse:
    """Clone an https or ssh repository link into `~/git/<repository>` and mount it."""
    with http_errors((GitError, 400, None)):
        roots.add_root(str(git_worktree.clone_repository(body.url)))
    return FileTreeResponse(enabled=roots.enabled, tree=roots.tree())


@router.delete("/roots", response_model=FileTreeResponse)
def remove_root(path: str = Query(...), roots: FileRoots = Depends(get_file_roots)) -> FileTreeResponse:
    roots.remove_root(path)
    return FileTreeResponse(enabled=roots.enabled, tree=roots.tree())


@router.get("/browse", response_model=DirectoryListing, dependencies=[_MOUNTING])
def browse_directories(path: str = Query(default="~")) -> DirectoryListing:
    """List the subdirectories of *path* on the backend host, to pick a new root."""
    with http_errors((NotADirectoryError, 404, f"Not a directory: {path}"), (PermissionError, 403, f"Cannot read {path}")):
        directory, names = list_directories(path)
    parent = str(directory.parent) if directory.parent != directory else None
    return DirectoryListing(path=str(directory), parent=parent, directories=names)


@router.post("/file")
def write_file(body: FileContent, roots: FileRoots = Depends(get_file_roots)) -> dict[str, bool]:
    with http_errors(((OSError, ValueError), 400, None)):
        roots.write(body.path, body.content)
    return {"ok": True}


@router.get("/rendered", response_model=FileContent)
def read_rendered(path: str = Query(...), roots: FileRoots = Depends(get_file_roots)) -> FileContent:
    with http_errors(((FileNotFoundError, ValueError), 404, None)):
        raw = roots.read(path)
        return FileContent(path=path, content=roots.resolve_wiki_content(raw, path))
