"""LaTeX builds and omp workspace setup for one `.tex` file, for logins with host access."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.http_plumbing import http_errors
from backend.identity import host_access_only
from backend.routes.deps import get_document_store, get_file_roots, get_project_store, get_relocate
from backend.routes.fileviewer import resolve_allowed
import config
from lib import document_service, latex
from lib.data_store import DataStore
from lib.document_service import DocumentNameConflict, DocumentNotFound, Relocate
from lib.file_roots import FileRoots
from lib.project_store import ProjectStore

router = APIRouter(prefix="/api/latex", tags=["latex"], dependencies=[host_access_only("LaTeX")])


class TexPath(BaseModel):
    """Absolute path of the `.tex` file."""

    path: str


class PreparedWorkspace(BaseModel):
    cwd: str


def _resolve_tex_file(path: str, roots: FileRoots) -> Path:
    resolved = resolve_allowed(path, roots)
    if resolved.suffix != ".tex":
        raise HTTPException(status_code=400, detail="Not a .tex file")
    return resolved


class ProjectTex(BaseModel):
    """`name` of a new document, or `path` (storage key) of a stored `.tex` project file to move to disk."""

    slug: str
    name: str = ""
    path: str = ""


class TexFile(BaseModel):
    path: str


def _linked_root(store: ProjectStore, roots: FileRoots, slug: str) -> Path | None:
    """The project's linked repository: its `linked_root`, else the mounted root named like the slug."""
    linked = store.get_project(slug).get("linked_root")
    mounted = [Path(root) for root in roots.roots()]
    return next((root for root in mounted if (str(root) == linked if linked else root.name == slug) and root.is_dir()), None)


def _folder(store: ProjectStore, roots: FileRoots, slug: str, name: str) -> tuple[Path, str]:
    stem = Path(name.strip()).name.removesuffix(".tex")
    with http_errors((ValueError, 400, None)):
        return latex.project_folder(config.LATEX, slug, stem, _linked_root(store, roots, slug)), stem


@router.post("/create")
def post_create(
    body: ProjectTex, store: ProjectStore = Depends(get_project_store), roots: FileRoots = Depends(get_file_roots),
) -> TexFile:
    """Create a LaTeX document of project `slug` on disk and list it in the project's files."""
    conflict = (FileExistsError, 409, "A document with that name already exists")
    with http_errors(conflict, (FileNotFoundError, 404, "Project not found")):
        folder, stem = _folder(store, roots, body.slug, body.name)
        source = latex.create_document(folder, stem)
    store.add_file(body.slug, str(source))
    return TexFile(path=str(source))


@router.post("/adopt")
def post_adopt(
    body: ProjectTex,
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
    roots: FileRoots = Depends(get_file_roots),
    relocate: Relocate = Depends(get_relocate),
) -> TexFile:
    """Move the stored `.tex` project file `path` into its LaTeX folder, references follow, so it can build."""
    with http_errors((FileNotFoundError, 404, "Project not found")):
        name = dict(store.list_named_files(body.slug)).get(body.path)
    if not name or not name.endswith(".tex"):
        raise HTTPException(status_code=404, detail="No such .tex file in this project")
    with http_errors((DocumentNotFound, 404, None), (DocumentNameConflict, 409, "A document with that name already exists")):
        folder, _ = _folder(store, roots, body.slug, name)
        folder.mkdir(parents=True, exist_ok=True)
        return TexFile(path=str(document_service.move_to_folder(store, docs, relocate, body.path, folder)))


@router.post("/prepare")
def post_prepare(body: TexPath, roots: FileRoots = Depends(get_file_roots)) -> PreparedWorkspace:
    """Give omp sessions in the file's directory the LaTeX prompt and no AGENTS.md, keeping any existing `.omp` files."""
    directory = _resolve_tex_file(body.path, roots).parent
    latex.prepare_omp_workspace(directory)
    return PreparedWorkspace(cwd=str(directory))


@router.post("/build")
async def post_build(body: TexPath, roots: FileRoots = Depends(get_file_roots)) -> latex.LatexBuild:
    source = await asyncio.to_thread(_resolve_tex_file, body.path, roots)
    with http_errors((latex.LatexUnavailable, 503, None)):
        return await latex.build(source)
