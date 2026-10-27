"""Document routes for the file-tree document feature.

A *document* is a logical key or a mounted file path listed in a project's
``files``. Library PDFs (``PDFs/<name>``) come from mounted PDFs and are sent to
MinerU for a parsed markdown preview. The UI only passes back opaque keys the API
handed it, and every read/write is scoped to the requesting user. PDFs and MinerU
output are also mirrored into Nextcloud as a convenience copy the app never reads back.
"""

import logging
from config import DOCUMENTS
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from backend.routes.deps import (
    get_architecture_store,
    get_asset_store,
    get_document_store,
    get_file_roots,
    get_project_store,
    http_errors,
)
from backend.routes.schemas import FileListResponse, FileMeta, file_meta
from lib.architecture_workspace import ArchitectureStore
from lib.asset_store import AssetStore
from lib.data_store import DataStore, DataStorePath, StorageConflictError, StorageNotFoundError, validate_key, write_text
from lib.file_roots import FileRoots
from lib.project_store import ProjectStore
from lib.storage_namespace import (
    DRAWING,
    GRAPH,
    NOTE,
    PDFS,
    architecture_prefix,
    project_attachment_prefix,
    project_document,
    project_document_prefix,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


class WriteDocumentRequest(BaseModel):
    slug: str
    name: str
    content: str = ""


class AddDocumentRequest(BaseModel):
    path: str


class RenameDocumentRequest(BaseModel):
    slug: str
    path: str
    name: str


def _document_not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Document not found")


def _name_conflict() -> HTTPException:
    return HTTPException(status_code=409, detail="A document with that name already exists there")


def _valid_key(path: str) -> str:
    """*path* as a storage key, 404 when it is not a valid one."""
    with http_errors((ValueError, 404, "Document not found")):
        return validate_key(path)


def _safe_filename(name: str) -> str:
    """The final path component of a client-supplied *name*, 400 when it has none."""
    safe_name = Path(name).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return safe_name


def _upload_name(file: UploadFile) -> str:
    """The safe filename of an upload, 400 when it is missing or unusable."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    return _safe_filename(file.filename)


def _meta_list(paths: list[str]) -> list[FileMeta]:
    # ponytail: canvas-pasted images stay registered (so /fileviewer can serve them)
    # but never surface as documents (they'd flood the sidebar and the canvas add menu).
    return [file_meta(p) for p in paths if not Path(p).name.startswith("pasted-")]


def _under(key: str, prefix: str) -> bool:
    return key == prefix or key.startswith(f"{prefix}/")


def _known_key(store: ProjectStore, path: str) -> str:
    """The logical key for *path*, 404 when it is not one the app knows about.

    Allow-list includes the document library, architecture workspace sources,
    and any document a project references, expressed in logical keys.
    Logical keys cannot escape their application namespace, and anything unknown
    answers 404 so another user's documents stay undiscoverable.
    """
    key = _valid_key(path)
    if not (_under(key, PDFS) or _under(key, GRAPH) or key in set(store.list_all_files())):
        raise _document_not_found()
    return key


def _exists_key(docs: DataStore, assets: AssetStore, key: str) -> bool:
    return DataStorePath(docs, key).is_file() or bool(assets.list(key))


def _follow_moved_reference(store: ProjectStore, roots: FileRoots, slug: str, path: str) -> str:
    """*path*, or where the mount file it names moved to (rewriting the project's reference), unchanged when it exists or cannot be found."""
    if not Path(path).is_absolute() or Path(path).exists():
        return path
    moved = roots.relocate(path)
    if moved is None:
        return path
    store.replace_file(slug, path, str(moved))
    return str(moved)


@router.get("", response_model=FileListResponse)
async def list_documents(
    slug: str = Query(...),
    store: ProjectStore = Depends(get_project_store),
    architecture: ArchitectureStore = Depends(get_architecture_store),
    roots: FileRoots = Depends(get_file_roots),
):
    """The project's referenced documents plus every architecture source, found
    from the workspace root downward and named by their path inside it.

    - A referenced mount file that moved is followed and the project's reference rewritten
    """
    with http_errors((FileNotFoundError, 404, None)):
        files = [_follow_moved_reference(store, roots, slug, f) for f in store.list_files(slug)]
    sources = [file_meta(f"{architecture_prefix(slug)}/{relative}", name=relative) for relative in architecture.source_paths()]
    # The workspace is the source of truth for its files, so a stray reference never duplicates one.
    return FileListResponse(documents=_meta_list([f for f in files if not _under(f, GRAPH)]) + sources)


@router.get("/all", response_model=FileListResponse)
async def list_all_documents(store: ProjectStore = Depends(get_project_store)):
    """Every document referenced by any project."""
    return FileListResponse(documents=_meta_list(store.list_all_files()))


@router.get("/notes", response_model=FileListResponse)
async def list_notes(docs: DataStore = Depends(get_document_store)) -> FileListResponse:
    """List non-project canvases/notes."""
    paths = [entry.key for entry in docs.list(NOTE) if not entry.is_dir]
    return FileListResponse(documents=_meta_list(paths))


@router.post("/write", response_model=FileMeta)
async def write_document(
    req: WriteDocumentRequest,
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
    assets: AssetStore = Depends(get_asset_store),
):
    """Create or overwrite a document. Empty slug = non-project note,
    otherwise the project's documents/ directory."""
    safe_name = _safe_filename(req.name)

    if req.slug == "":
        key = f"{NOTE}/{safe_name}"
        write_text(docs, key, req.content)
        return file_meta(key)

    with http_errors((FileNotFoundError, 404, "Project not found")):
        store.require(req.slug)

    key = project_document(req.slug, safe_name)
    write_text(docs, key, req.content)
    store.add_file(req.slug, key)

    # Keep the active chat attachment copy in sync with its source document.
    content = req.content.encode("utf-8")
    for asset in assets.list(project_attachment_prefix(req.slug), kind="upload"):
        if PurePosixPath(asset.logical_path).name == safe_name:
            assets.write("upload", asset.logical_path, content)

    return file_meta(key)


@router.post("/write-binary", response_model=FileMeta)
async def write_binary_document(
    file: UploadFile = File(...),
    slug: str = Query(...),
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
):
    """Write a binary file (e.g. PDF) to the project's documents/ directory."""
    with http_errors((FileNotFoundError, 404, "Project not found")):
        store.require(slug)
    key = project_document(slug, _upload_name(file))
    docs.write_bytes(key, await file.read())
    store.add_file(slug, key)
    return file_meta(key)


@router.post("/store-drawing")
async def store_drawing(
    file: UploadFile = File(...),
    assets: AssetStore = Depends(get_asset_store),
):
    """Store a rendered canvas image without creating a Nextcloud copy."""
    assets.write("drawing", f"{DRAWING}/{_upload_name(file)}", await file.read())
    return {"status": "ok"}


@router.post("/add", response_model=FileMeta)
async def add_document(
    req: AddDocumentRequest,
    slug: str = Query(...),
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
    assets: AssetStore = Depends(get_asset_store),
):
    """Assign an existing document (from the global pool/library) to a project."""
    key = _known_key(store, req.path)
    if not _exists_key(docs, assets, key):
        raise _document_not_found()
    with http_errors((FileNotFoundError, 404, None)):
        store.add_file(slug, key)
    return file_meta(key)


def _delete_under(docs: DataStore, path: str, prefix: str) -> None:
    """Delete a document only when it really is a file under *prefix*."""
    try:
        key = validate_key(path)
    except ValueError:
        return
    if _under(key, prefix) and DataStorePath(docs, key).is_file():
        docs.delete(key)


@router.delete("")
async def remove_document(
    slug: str = Query(...),
    path: str = Query(...),
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
):
    """Unassign a document from a project. If it lives in the project's
    documents/ directory, also delete the file itself. Empty slug = non-project
    note, deleted directly since there's no project to unlink from."""
    if slug == "":
        _delete_under(docs, path, NOTE)
        return {"status": "ok"}

    with http_errors((FileNotFoundError, 404, None)):
        store.remove_file(slug, path)
    _delete_under(docs, path, project_document_prefix(slug))
    return {"status": "ok"}


class MoveDocumentRequest(BaseModel):
    path: str
    from_slug: str = ""
    to_slug: str = ""


@router.post("/move", response_model=FileMeta)
async def move_document(
    req: MoveDocumentRequest,
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
):
    """Move a document (e.g. a canvas) between projects, or to/from the
    unassigned root notes collection (empty slug = notes)."""
    if req.from_slug == req.to_slug:
        raise HTTPException(status_code=400, detail="Source and destination are the same")

    key = _valid_key(req.path)

    if req.from_slug == "":
        known = _under(key, NOTE)
    else:
        with http_errors((FileNotFoundError, 404, None)):
            known = key in set(store.list_files(req.from_slug))
    if not known or not DataStorePath(docs, key).is_file():
        raise _document_not_found()

    if req.to_slug:
        with http_errors((FileNotFoundError, 404, "Project not found")):
            store.require(req.to_slug)
        dest_prefix = project_document_prefix(req.to_slug)
    else:
        dest_prefix = NOTE

    dest = f"{dest_prefix}/{PurePosixPath(key).name}"
    if dest != key:
        if docs.exists(dest):
            raise _name_conflict()
        docs.move(key, dest)

    if req.from_slug:
        store.remove_file(req.from_slug, key)
    if req.to_slug:
        store.add_file(req.to_slug, dest)

    return file_meta(dest)


def _rename_file(store: ProjectStore, docs: DataStore, slug: str, key: str, prefix: str, name: str) -> FileMeta:
    """Rename a stored document inside *prefix*, repointing the project's reference."""
    if not _under(key, prefix):
        raise _document_not_found()
    destination = f"{prefix}/{name}"
    if destination == key:
        return file_meta(key)
    if docs.exists(destination):
        raise _name_conflict()
    docs.move(key, destination)
    if slug:
        store.remove_file(slug, key)
        store.add_file(slug, destination)
    return file_meta(destination)


def _rename_pdf(store: ProjectStore, assets: AssetStore, key: str, name: str) -> FileMeta:
    """Rename a library PDF, repointing every project that references it."""
    if not _under(key, PDFS):
        raise _document_not_found()
    with http_errors((StorageNotFoundError, 404, "Document not found")):
        asset = assets.read(key)
    if asset.kind != "pdf":
        raise _document_not_found()

    destination = f"{PDFS}/{name}"
    if destination == key:
        return file_meta(key)
    with http_errors((StorageConflictError, 409, "A document with that name already exists there")):
        renamed = assets.move(key, destination)

    # Keep Nextcloud's convenience mirror aligned with the database authority.
    Path(DOCUMENTS, *key.split("/")).unlink(missing_ok=True)
    try:
        assets.mirror(renamed, DOCUMENTS)
    except OSError:
        log.exception("Failed to mirror renamed PDF %s into Nextcloud", destination)
    for project in store.list_projects():
        slug = project["slug"]
        if key in store.list_files(slug):
            store.remove_file(slug, key)
            store.add_file(slug, destination)
    return file_meta(destination)


@router.post("/rename", response_model=FileMeta)
async def rename_document(
    req: RenameDocumentRequest,
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
    assets: AssetStore = Depends(get_asset_store),
):
    """Rename a database-owned document, retaining its content and project links."""
    key = _valid_key(req.path)

    if req.slug:
        with http_errors((FileNotFoundError, 404, None)):
            known = key in set(store.list_files(req.slug))
        prefix = project_document_prefix(req.slug)
    else:
        known = _under(key, NOTE)
        prefix = NOTE
    if not known:
        raise _document_not_found()

    name = Path(req.name).name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Document name is required")
    suffix = "".join(Path(PurePosixPath(key).name).suffixes)
    if suffix and not name.endswith(suffix):
        name += suffix

    if DataStorePath(docs, key).is_file():
        return _rename_file(store, docs, req.slug, key, prefix, name)
    return _rename_pdf(store, assets, key, name)
