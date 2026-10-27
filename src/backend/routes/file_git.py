"""Git views of file roots: working-tree diffs against HEAD, and the worktrees of a mounted clone.

Handlers are plain `def` because git runs synchronously, so FastAPI runs them in its threadpool.
"""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.http_plumbing import http_errors
from backend.routes.deps import get_file_roots
from backend.routes.schemas import RootFile
from lib import git_diff, git_worktree
from lib.file_roots import FileRoots
from lib.git import GitError

router = APIRouter(prefix="/api/fileroots", tags=["fileroots"])

# A dirty worktree or an unmerged branch needs the user's decision (409), any other git refusal is a bad request.
_GIT_REFUSALS = (((git_worktree.WorktreeDirty, git_worktree.BranchNotMerged), 409, None), (GitError, 400, None))


class DiffFile(BaseModel):
    path: str
    status: git_diff.ChangeStatus
    stamp: str
    added: int | None = None
    removed: int | None = None


class DiffStatusResponse(BaseModel):
    is_git_repository: bool
    files: list[DiffFile]


class FileDiffResponse(BaseModel):
    supported: bool
    status: git_diff.ChangeStatus | None = None
    patch: str | None = None


class DiffContentResponse(BaseModel):
    base_content: str
    content: str


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
    default_branch: str | None = None
    branches: list[str] = []
    worktrees: list[WorktreeEntry] = []


class DefaultWorktreeBody(BaseModel):
    root: str


class CreateWorktreeBody(BaseModel):
    root: str
    name: str
    base: str


def _git_directory(root: str, roots: FileRoots) -> Path:
    """The directory *root* inside a file root or one of its worktrees, 403 outside them, 404 when it is no directory."""
    with http_errors((ValueError, 403, None)):
        roots.root_of(root)
        directory = Path(root).expanduser().resolve()
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail=f"Not a directory: {root}")
    return directory


def _clone_root(root: str, roots: FileRoots) -> Path:
    """The registered mount *root*, 403 for anything else."""
    directory = Path(root).expanduser().resolve()
    if str(directory) not in roots.roots():
        raise HTTPException(status_code=403, detail=f"Not a file mount: {root}")
    return directory


def _checkout(worktree: git_worktree.Worktree) -> RootFile:
    """A worktree as a root file, named by its branch (by its directory when detached)."""
    return RootFile(path=worktree.path, name=worktree.branch or Path(worktree.path).name)


@router.get("/diff/files", response_model=DiffStatusResponse)
def list_changed_files(root: str = Query(...), roots: FileRoots = Depends(get_file_roots)) -> DiffStatusResponse:
    """Files of the git working tree at *root* that differ from HEAD."""
    directory = _git_directory(root, roots)
    files = git_diff.changed_files(directory)
    if files is None:
        return DiffStatusResponse(is_git_repository=False, files=[])
    counts = git_diff.line_counts(directory, files)
    changed: list[DiffFile] = []
    for file in files:
        added, removed = counts.get(file.path, (None, None))
        stamp = git_diff.file_stamp(directory, file.path)
        changed.append(DiffFile(path=file.path, status=file.status, stamp=stamp, added=added, removed=removed))
    return DiffStatusResponse(is_git_repository=True, files=changed)


@router.get("/diff", response_model=FileDiffResponse)
def read_file_diff(root: str = Query(...), path: str = Query(...), roots: FileRoots = Depends(get_file_roots)) -> FileDiffResponse:
    """Unified patch of *path* (relative to *root*) against HEAD, `supported=false` for binary or oversized files."""
    patch = git_diff.file_patch(_git_directory(root, roots), path)
    if patch is None:
        return FileDiffResponse(supported=False)
    return FileDiffResponse(supported=True, status=patch[0], patch=patch[1])


@router.get("/diff/content", response_model=DiffContentResponse)
def read_diff_content(
    root: str = Query(...), path: str = Query(...), roots: FileRoots = Depends(get_file_roots),
) -> DiffContentResponse:
    """HEAD and working-tree text of *path* (relative to *root*) for the native Neovim diff."""
    with http_errors((git_diff.UnsupportedDiffContent, 422, None)):
        base, content = git_diff.file_contents(_git_directory(root, roots), path)
    return DiffContentResponse(base_content=base, content=content)


@router.get("/worktrees", response_model=WorktreeListResponse)
def list_worktrees(root: str = Query(...), roots: FileRoots = Depends(get_file_roots)) -> WorktreeListResponse:
    """Worktrees of the clone mounted at *root*, each with its uncommitted files and commits ahead/behind the base."""
    repository = _clone_root(root, roots)
    if not git_worktree.is_clone(repository):
        return WorktreeListResponse(is_clone=False)
    with http_errors(*_GIT_REFUSALS):
        # One lookup each for the branches, the default branch and the base ref, shared by every worktree.
        known = git_worktree.branches(repository)
        default = git_worktree.default_branch(repository)
        base = git_worktree.default_base_branch(known, default)
        compared = git_worktree.local_branch(repository, base)
        entries = [
            WorktreeEntry(
                path=worktree.path, branch=worktree.branch, is_main=worktree.is_main,
                **vars(git_worktree.status(Path(worktree.path), compared)),
            )
            for worktree in git_worktree.list_worktrees(repository)
        ]
        return WorktreeListResponse(is_clone=True, default_base=base, default_branch=default, branches=known, worktrees=entries)


@router.post("/worktrees", response_model=RootFile)
def create_worktree(body: CreateWorktreeBody, roots: FileRoots = Depends(get_file_roots)) -> RootFile:
    """Pull *base* and branch a new worktree off it, `name` is the branch."""
    repository = _clone_root(body.root, roots)
    with http_errors(*_GIT_REFUSALS):
        return _checkout(git_worktree.create_worktree(repository, body.name, body.base))


@router.post("/worktrees/default", response_model=RootFile)
def default_worktree(body: DefaultWorktreeBody, roots: FileRoots = Depends(get_file_roots)) -> RootFile:
    """Open the inferred default branch, creating its checkout only when needed."""
    repository = _clone_root(body.root, roots)
    with http_errors(*_GIT_REFUSALS):
        return _checkout(git_worktree.default_worktree(repository))


@router.delete("/worktrees")
def remove_worktree(
    root: str = Query(...), path: str = Query(...), force: bool = Query(default=False), roots: FileRoots = Depends(get_file_roots),
) -> dict[str, bool]:
    """Remove a worktree, keeping its branch. 409 when it holds uncommitted changes and *force* is off."""
    with http_errors(*_GIT_REFUSALS):
        git_worktree.remove_worktree(_clone_root(root, roots), path, force=force)
    return {"ok": True}


@router.delete("/worktrees/branch")
def delete_worktree_branch(
    root: str = Query(...), branch: str = Query(...), force: bool = Query(default=False), roots: FileRoots = Depends(get_file_roots),
) -> dict[str, bool]:
    """Delete a local branch. 409 when it is not fully merged and *force* is off."""
    with http_errors(*_GIT_REFUSALS):
        git_worktree.delete_branch(_clone_root(root, roots), branch, force=force)
    return {"ok": True}
