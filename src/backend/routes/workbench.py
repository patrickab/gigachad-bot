"""Workbench files on disk (`lib.workbench`) and the omp session of each, for logins with host access."""

from datetime import datetime
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from backend.http_plumbing import http_errors
from backend.identity import host_access_only
from backend.routes.deps import (
    get_asset_store,
    get_disk_files,
    get_document_store,
    get_file_roots,
    get_project_store,
    get_relocate,
)
from backend.routes.fileviewer import resolve_allowed
import config
from lib import document_service, latex, latex_templates
from lib.asset_store import AssetStore
from lib.data_store import DataStore, InvalidStorageKey, StorageConflictError, StorageNotFoundError, validate_key
from lib.disk_files import DiskFiles
from lib.document_service import DocumentNameConflict, DocumentNotFound, Relocate, under
from lib.file_roots import FileRoots
from lib.project_store import ProjectStore
from lib.storage_namespace import NOTE, PDFS, library_pdf
from lib.workbench import Kind, document_folder, kind_directory, kind_of, new_folder, suffix_of
from lib.workbench.links import Link, SessionLinks

router = APIRouter(prefix="/api/workbench", tags=["workbench"], dependencies=[host_access_only("Workbench")])

_CONTENT: dict[Kind, str] = {"latex": latex.NEW_DOCUMENT, "markdown": ""}
_IMAGE_DIRECTORY = latex_templates.IMAGE_DIRECTORY
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class FilePath(BaseModel):
    """A project document key (`project/<slug>/document/<name>`) or an absolute disk path."""

    path: str


class PreparedFile(BaseModel):
    """The file on disk, the omp session cwd of its kind and the kind."""

    path: str
    cwd: str
    kind: Kind


class NewDocument(BaseModel):
    """The document `name` of project `slug`, a LaTeX presentation of the template set `template` when given."""

    slug: str
    name: str
    template: str | None = None


class Promotion(BaseModel):
    """The LaTeX file `path` of project `slug` becomes a project folder."""

    slug: str
    path: str


class PromotedFolder(BaseModel):
    folder: str


class StoredImage(BaseModel):
    """Where a pasted image went, its path relative to the folder it is stored under (the project folder for LaTeX) and the text that references it from the file it was pasted into."""

    path: str
    placeholder_path: str
    reference: str
    changed: list[str] = []


class SessionLink(BaseModel):
    """Make `session_id` the active omp session of the absolute file `path`, None starts a new one."""

    path: str
    session_id: str | None


class LinkedSession(BaseModel):
    """The active session (None: a new one), every session the file had (first linked first) and those owed its new path."""

    session_id: str | None
    session_ids: list[str]
    moved: list[str] = []


class SessionNotice(BaseModel):
    """The session `session_id` of the absolute file `path` was told where the file is now."""

    path: str
    session_id: str


def _linked_session(link: Link) -> LinkedSession:
    return LinkedSession(session_id=link.active, session_ids=list(link.sessions), moved=list(link.moved))


def _absolute(path: str) -> str:
    """*path*, 400 unless it is absolute."""
    if not PurePosixPath(path).is_absolute():
        raise HTTPException(status_code=400, detail="Path must be absolute")
    return path


def _kind(name: str) -> Kind:
    kind = kind_of(name)
    if kind is None:
        raise HTTPException(status_code=400, detail="The workbench edits only .tex, .bib, .sty, .cls and .md files")
    return kind


def _prepared(path: Path | str, kind: Kind) -> PreparedFile:
    return PreparedFile(path=str(path), cwd=str(kind_directory(config.WORKBENCH, kind)), kind=kind)


def _pdf_on_disk(key: str, assets: AssetStore) -> Path:
    """The disk copy of the stored PDF `key`: its library mirror, written again when missing."""
    name = PurePosixPath(key).name
    mirrored = config.DOCUMENTS / PDFS / name
    if mirrored.is_file():
        return mirrored
    with http_errors((StorageNotFoundError, 404, "This PDF has no library copy to read")):
        return assets.mirror(assets.read(library_pdf(name)), config.DOCUMENTS)


@router.post("/prepare")
def post_prepare(
    body: FilePath,
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
    roots: FileRoots = Depends(get_file_roots),
    relocate: Relocate = Depends(get_relocate),
    assets: AssetStore = Depends(get_asset_store),
) -> PreparedFile:
    """The disk file to open for `path`: an absolute file resolved in place, a stored project document moved into its workbench folder.

    - A stored PDF is never moved, its library copy on disk is read where it is
    - Notes stay stored and open in the document editor, so their keys answer 400
    - A workbench folder the file roots cannot reach answers 403 and nothing moves
    """
    if PurePosixPath(body.path).is_absolute():
        resolved = resolve_allowed(body.path, roots)
        return _prepared(resolved, _kind(resolved.name))
    with http_errors((InvalidStorageKey, 400, None)):
        key = validate_key(body.path)
    if under(key, NOTE):
        raise HTTPException(status_code=400, detail="Notes open in the document editor")
    with http_errors((DocumentNotFound, 404, "Document not found")):
        name = PurePosixPath(key).name if under(key, PDFS) else document_service.file_name(store, docs, key)
    kind = _kind(name)
    if kind == "pdf":
        return _prepared(_pdf_on_disk(key, assets), kind)
    with http_errors((ValueError, 400, None)):
        folder = document_folder(config.WORKBENCH, kind, key.split("/")[1], Path(name).stem)
    if not roots.contains(folder / name):
        raise HTTPException(status_code=403, detail="The workbench folder is outside the allowed file roots")
    conflict = "A document with that name already exists"
    with new_folder(folder), http_errors(
        (DocumentNotFound, 404, "Document not found"), ((DocumentNameConflict, StorageConflictError), 409, conflict)
    ):
        return _prepared(document_service.move_to_folder(store, docs, relocate, key, folder), kind)


@router.post("/create")
def post_create(body: NewDocument, store: ProjectStore = Depends(get_project_store)) -> FilePath:
    """Create the document `name` of project `slug` in its workbench folder and list it in the project's files.

    - With `template` the document is a LaTeX project folder, listed as such: `main.tex` from the set's skeleton, its theme `.sty` and `imgs/`
    - A template for a Markdown document or one that is no template set answers 400
    - A folder that exists already answers 409 even without a file of that name, a leftover is never adopted
    - A failure to list it removes the new folder with everything written into it
    """
    name = body.name.strip()
    kind = _kind(name)
    if kind == "pdf" or Path(name).suffix.lower() != suffix_of(kind):
        raise HTTPException(status_code=400, detail="New documents are .md or .tex")
    template = None if body.template is None else latex_templates.template_set(body.template)
    if body.template is not None and (kind != "latex" or template is None):
        raise HTTPException(status_code=400, detail="Unknown presentation template")
    with http_errors((FileNotFoundError, 404, "Project not found"), (ValueError, 400, None)):
        store.require(body.slug)
        folder = document_folder(config.WORKBENCH, kind, body.slug, name.removesuffix(Path(name).suffix))
    conflict = "A document with that name already exists"
    if folder.exists():
        raise HTTPException(status_code=409, detail=conflict)
    with new_folder(folder):
        if template is None:
            written = listed = folder / f"{folder.name}{suffix_of(kind)}"
            with http_errors((FileExistsError, 409, conflict)), written.open("x", encoding="utf-8") as file:
                file.write(_CONTENT[kind])
        else:
            written, listed = folder / "main.tex", folder
            with http_errors((FileExistsError, 409, conflict)):
                template.populate(folder)
        store.add_file(body.slug, str(listed))
    return FilePath(path=str(written))


@router.post("/promote")
def post_promote(body: Promotion, store: ProjectStore = Depends(get_project_store)) -> PromotedFolder:
    """Replace the listed `.tex` directly inside `<slug>/<folder>/` of a project by that folder, where it was in the list. There is no way back.

    - A file the project does not list answers 404, any other path 400, a folder that is listed already 409
    - A display name `X.tex` becomes `X`, other display names stay
    """
    with http_errors((FileNotFoundError, 404, "Project not found")):
        named = dict(store.list_named_files(body.slug))
    if body.path not in named:
        raise HTTPException(status_code=404, detail="Document not found")
    source = Path(body.path)
    with http_errors((ValueError, 400, None)):
        folder = document_folder(config.WORKBENCH, "latex", body.slug, source.parent.name)
    if source.suffix.lower() != ".tex" or source != folder / source.name:
        raise HTTPException(status_code=400, detail="Only a LaTeX document in a workbench folder becomes a project")
    if str(folder) in named:
        raise HTTPException(status_code=409, detail="This folder is already listed")
    shown = named[body.path]
    store.replace_file(body.slug, body.path, str(folder), name=Path(shown).stem if shown.lower().endswith(".tex") else shown)
    return PromotedFolder(folder=str(folder))


@router.post("/image")
def post_image(
    path: str = Query(...),
    file: UploadFile = File(...),
    roots: FileRoots = Depends(get_file_roots),
    files: DiskFiles = Depends(get_disk_files),
) -> StoredImage:
    """Store the PNG `file` pasted into the workbench file `path` as `imgs/img-<YYYYMMDD-HHMMSS>.png` and say how to reference it.

    - A LaTeX file stores into its project folder and is referenced by `\\includegraphics`, a Markdown file next to itself with `![]()`
    - A LaTeX presentation of a template set is referenced by the set's image macro
    - A LaTeX root that does not load `graphicx` gets it, so the reference builds. Beamer, so every presentation, loads it
    - Anything but a PNG answers 400, more than 20 MB 413, a name already taken 409
    """
    source = resolve_allowed(path, roots)
    kind = _kind(source.name)
    if kind == "pdf":
        raise HTTPException(status_code=400, detail="Images are pasted into .tex and .md files")
    content = file.file.read(_MAX_IMAGE_BYTES + 1)
    if len(content) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Images are at most 20 MB")
    if not content.startswith(_PNG_SIGNATURE):
        raise HTTPException(status_code=400, detail="Only PNG images can be pasted")
    project = None if kind == "markdown" else latex.project_of(source, ceiling=roots.ceiling_of(source))
    base = source.parent if project is None else project.folder
    theme = None if project is None or project.root is None else latex_templates.presentation_theme(project.root)
    name = f"img-{datetime.now():%Y%m%d-%H%M%S}.png"
    target = base / _IMAGE_DIRECTORY / name
    with http_errors((FileExistsError, 409, "An image with that name exists already"), (OSError, 400, None)):
        target.parent.mkdir(parents=True, exist_ok=True)
        target = files.create(target, content)
    changed = []
    if project is not None and project.root is not None and theme is None and latex.ensure_package(project.root, "graphicx"):
        files.record(project.root)
        changed.append(str(project.root))
    image = f"{_IMAGE_DIRECTORY}/{name}"
    return StoredImage(path=str(target), placeholder_path=image, reference=_image_reference(kind, theme, image), changed=changed)


def _image_reference(kind: Kind, theme: latex_templates.TemplateSet | None, image: str) -> str:
    """The text that shows *image* in a *kind* document: Markdown `![]()`, the presentation *theme*'s image macro, else `\\includegraphics`."""
    if kind == "markdown":
        return f"![]({image})"
    themed = None if theme is None else theme.image_reference(image)
    return themed or rf"\includegraphics[width=\linewidth]{{{image}}}"


@router.get("/session")
def get_session(
    path: str = Query(...), docs: DataStore = Depends(get_document_store), roots: FileRoots = Depends(get_file_roots),
) -> LinkedSession:
    """The omp sessions linked to the absolute file or LaTeX project folder `path` (`SessionLinks.of`)."""
    return _linked_session(SessionLinks(docs).of(path, roots))


@router.put("/session")
def put_session(body: SessionLink, docs: DataStore = Depends(get_document_store)) -> LinkedSession:
    """Make `session_id` the active session of the absolute file `path`, keeping every earlier one. None starts a new session."""
    return _linked_session(SessionLinks(docs).activate(_absolute(body.path), body.session_id))


@router.post("/session/informed")
def post_session_informed(body: SessionNotice, docs: DataStore = Depends(get_document_store)) -> LinkedSession:
    """The session `session_id` knows where the file `path` is now, so its messages stop naming it."""
    return _linked_session(SessionLinks(docs).inform(_absolute(body.path), body.session_id))
