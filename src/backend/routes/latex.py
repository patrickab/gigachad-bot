"""LaTeX builds and tex projects (a folder holding a root `.tex` file), for logins with host access."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.http_plumbing import http_errors
from backend.identity import host_access_only
from backend.routes.deps import get_disk_files, get_file_roots
from backend.routes.fileviewer import resolve_allowed
from lib import latex, naming
from lib.disk_files import DiskFiles
from lib.file_roots import FileRoots
from lib.workbench import kind_of

router = APIRouter(prefix="/api/latex", tags=["latex"], dependencies=[host_access_only("LaTeX")])


class TexPath(BaseModel):
    """Absolute path of a file."""

    path: str


class TexRootChoice(BaseModel):
    """The `.tex` file *path* and the root document *root* it belongs to, both absolute."""

    path: str
    root: str


class TexFileRequest(BaseModel):
    """The new file *name* (one segment) in the absolute *folder*."""

    folder: str
    name: str


class TexProjectResponse(BaseModel):
    """Absolute paths of a tex project's folder, root (null when unknown) and root candidates."""

    folder: str
    root: str | None
    candidates: list[str]


def _resolve_tex_file(path: str, roots: FileRoots) -> Path:
    resolved = resolve_allowed(path, roots)
    if resolved.suffix != ".tex":
        raise HTTPException(status_code=400, detail="Not a .tex file")
    return resolved


def _project_response(path: Path, roots: FileRoots) -> TexProjectResponse:
    project = latex.project_of(path, ceiling=roots.ceiling_of(path))
    return TexProjectResponse(
        folder=str(project.folder),
        root=None if project.root is None else str(project.root),
        candidates=[str(candidate) for candidate in project.candidates],
    )


@router.post("/build")
async def post_build(body: TexPath, roots: FileRoots = Depends(get_file_roots)) -> latex.LatexBuild:
    source = await asyncio.to_thread(_resolve_tex_file, body.path, roots)
    with http_errors((latex.LatexUnavailable, 503, None)):
        return await latex.build(source)


@router.get("/project")
def get_project(path: str, roots: FileRoots = Depends(get_file_roots)) -> TexProjectResponse:
    """The tex project the file at *path* belongs to."""
    return _project_response(resolve_allowed(path, roots), roots)


@router.post("/root")
def post_root(
    body: TexRootChoice, roots: FileRoots = Depends(get_file_roots), files: DiskFiles = Depends(get_disk_files),
) -> TexProjectResponse:
    """Make *root* the root document of the `.tex` file *path* by writing its `% !TEX root` line."""
    path = _resolve_tex_file(body.path, roots)
    root = _resolve_tex_file(body.root, roots)
    if not latex.is_root_document(root) or not root.is_relative_to(roots.ceiling_of(path)):
        raise HTTPException(status_code=400, detail="Not a root document of the same folder tree")
    with http_errors(((OSError, UnicodeError), 400, None)):
        latex.set_root(path, root)
    files.record(path)
    return _project_response(path, roots)


@router.post("/file")
def post_file(
    body: TexFileRequest, roots: FileRoots = Depends(get_file_roots), files: DiskFiles = Depends(get_disk_files),
) -> TexPath:
    """Create the empty file *name* in *folder*, a `.tex` file pointing at the folder's root, and record the change."""
    with http_errors((ValueError, 403, "Unknown file path"), (NotADirectoryError, 400, "Not a folder")):
        folder = roots.resolve_folder(body.folder)
    try:
        name = naming.segment(body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if kind_of(name) != "latex":
        raise HTTPException(status_code=400, detail="Name one .tex, .bib, .sty or .cls file")
    path = folder / name
    root = latex.project_of(path, ceiling=roots.ceiling_of(folder)).root
    with http_errors((FileExistsError, 409, f"{name} already exists"), (OSError, 400, None)):
        created = files.create(path, latex.new_file_content(path, root))
    return TexPath(path=str(created))
