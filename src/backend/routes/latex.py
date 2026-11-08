"""LaTeX builds and tex projects (a folder holding a root `.tex` file), for logins with host access."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.http_plumbing import http_errors
from backend.identity import host_access_only
from backend.routes.deps import get_disk_files, get_file_roots
from backend.routes.fileviewer import resolve_allowed
from lib import latex, latex_templates, naming, synctex
from lib.disk_files import DiskFiles
from lib.file_roots import FileRoots
from lib.workbench import kind_of

router = APIRouter(prefix="/api/latex", tags=["latex"], dependencies=[host_access_only("LaTeX")])
# Preview URLs carry the preview's version, so a cached one never goes stale
_PREVIEW_CACHE = "private, max-age=31536000, immutable"


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


class LatexTemplates(BaseModel):
    """The template set a root document is a presentation of (null for any other root) and that set's slides."""

    theme: str | None
    primitives: list[latex_templates.LatexPrimitive]


class TemplateSetEntry(BaseModel):
    """A template set a presentation starts from: its folder *name* and the *label* it is shown as."""

    name: str
    label: str


class SynctexPoint(BaseModel):
    """A point of the PDF of the root document *path*: 1-based *page*, PDF points from its top-left corner, the clicked *word*, the clicked character's 0-based *offset* in it and the word's 0-based *occurrence* among equal words of its rendered line."""

    path: str
    page: int = Field(ge=1)
    x: float
    y: float
    word: str | None = None
    offset: int = Field(default=0, ge=0)
    occurrence: int = Field(default=0, ge=0)


class SynctexSource(BaseModel):
    """The absolute source file and 1-based line that typeset a PDF point."""

    path: str
    line: int


class SynctexClickedSource(SynctexSource):
    """The source spot of a double-clicked PDF point, with the 0-based UTF-8 byte *column* of the clicked character (0 when unknown)."""

    column: int


class SynctexLine(BaseModel):
    """A 1-based *line* of the source *file* typeset by the root document *path*, both absolute."""

    path: str
    file: str
    line: int = Field(ge=1)


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


@router.get("/template-sets")
def get_template_sets() -> list[TemplateSetEntry]:
    """The template sets a new presentation can start from, by name."""
    return [TemplateSetEntry(name=template.name, label=template.label) for template in latex_templates.template_sets()]


@router.get("/templates")
def get_templates(root: str, roots: FileRoots = Depends(get_file_roots)) -> LatexTemplates:
    """The template set the root document *root* is a presentation of and its slides, null and none for any other root."""
    template = latex_templates.presentation_theme(_resolve_tex_file(root, roots))
    if template is None:
        return LatexTemplates(theme=None, primitives=[])
    return LatexTemplates(theme=template.name, primitives=template.primitives())


@router.get("/templates/{theme}/previews/{group}/{name}.png")
def get_template_preview(theme: str, group: str, name: str) -> FileResponse:
    """The preview PNG of primitive `<group>/<name>` of template set *theme*, 404 for an unknown set or primitive or without one."""
    template = latex_templates.template_set(theme)
    preview = None if template is None else template.preview(f"{group}/{name}")
    if preview is None:
        raise HTTPException(status_code=404, detail="No preview for this primitive")
    return FileResponse(preview, media_type="image/png", headers={"Cache-Control": _PREVIEW_CACHE})


@router.post("/synctex")
def post_synctex(body: SynctexPoint, roots: FileRoots = Depends(get_file_roots)) -> SynctexClickedSource:
    """The source line and column that typeset a point of the last build of the root document *path*, 404 without a match."""
    root = _resolve_tex_file(body.path, roots)
    with http_errors((synctex.SynctexUnavailable, 503, None)):
        match = synctex.synctex_source(root, body.page, body.x, body.y, body.word, body.offset, body.occurrence)
    if match is None:
        raise HTTPException(status_code=404, detail="No source for this point")
    source, line, column = match
    return SynctexClickedSource(path=str(resolve_allowed(str(source), roots)), line=line, column=column)


@router.post("/synctex/view")
def post_synctex_view(body: SynctexLine, roots: FileRoots = Depends(get_file_roots)) -> synctex.SynctexSpot:
    """Where a line of *file* lands in the last build of the root document *path*, 404 without a match."""
    root = _resolve_tex_file(body.path, roots)
    file = resolve_allowed(body.file, roots)
    with http_errors((synctex.SynctexUnavailable, 503, None)):
        spot = synctex.synctex_view(root, file, body.line)
    if spot is None:
        raise HTTPException(status_code=404, detail="No PDF spot for this line")
    return spot


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


class SynctexPage(BaseModel):
    """A 1-based *page* of the PDF of the root document *path*."""

    path: str
    page: int = Field(ge=1)


@router.post("/synctex/slide")
def post_synctex_slide(body: SynctexPage, roots: FileRoots = Depends(get_file_roots)) -> SynctexSource:
    """The source line after the slide on a page of the last build of the root document *path*, where a new frame goes, 404 without one."""
    root = _resolve_tex_file(body.path, roots)
    end = synctex.slide_end(root, body.page)
    if end is None:
        raise HTTPException(status_code=404, detail="No slide on this page")
    source, line = end
    return SynctexSource(path=str(resolve_allowed(str(source), roots)), line=line)
