"""Document routes for the file-tree document feature.

A *document* is a logical key or a mounted file path listed in a project's
``files``. Library PDFs (``PDFs/<name>``) come from mounted PDFs and are sent to
MinerU for a parsed markdown preview. The UI only passes back opaque keys the API
handed it, and every read/write is scoped to the requesting user. PDFs and MinerU
output are also mirrored into Nextcloud as a convenience copy the app never reads back.
Where a document lives and how a move or rename repoints its references is `lib.document_service`.
"""

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from backend.http_plumbing import http_errors
from backend.routes.deps import (
    get_architecture_store,
    get_asset_store,
    get_document_store,
    get_file_roots,
    get_project_store,
)
from backend.routes.schemas import FileListResponse, FileMeta, file_meta
from lib import document_service
from lib.architecture_workspace import ArchitectureStore
from lib.asset_store import AssetStore
from lib.data_store import DataStore, validate_key
from lib.document_service import DocumentNameConflict, DocumentNotFound, DocumentScope, InvalidDocumentName, ProjectNotFound, under
from lib.file_roots import FileRoots
from lib.project_store import ProjectStore
from lib.storage_namespace import DRAWING, GRAPH, NOTE, architecture_prefix, project_document

router = APIRouter(prefix="/api/documents", tags=["documents"])

_DOCUMENT_NOT_FOUND = "Document not found"
_DOCUMENT_ERRORS = (
    (DocumentNotFound, 404, _DOCUMENT_NOT_FOUND),
    (DocumentNameConflict, 409, "A document with that name already exists there"),
    (ProjectNotFound, 404, None),
    (InvalidDocumentName, 400, None),
)


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


class MoveDocumentRequest(BaseModel):
    path: str
    from_slug: str = ""
    to_slug: str = ""


def _valid_key(path: str) -> str:
    """*path* as a storage key, 404 when it is not a valid one."""
    with http_errors((ValueError, 404, _DOCUMENT_NOT_FOUND)):
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


def _follow_moved_reference(store: ProjectStore, roots: FileRoots, slug: str, path: str) -> str:
    """*path*, or where the mount file it names moved to (rewriting the project's reference).

    Unchanged when it exists or cannot be found.
    """
    if not Path(path).is_absolute() or Path(path).exists():
        return path
    moved = roots.relocate(path)
    if moved is None:
        return path
    store.replace_file(slug, path, str(moved))
    return str(moved)


# Every handler is sync: FastAPI runs it in its threadpool, so blocking store and file IO never stalls the event loop.
# Uploads are read through `UploadFile.file`, the spooled body FastAPI already received.
@router.get("", response_model=FileListResponse)
def list_documents(
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
    return FileListResponse(documents=_meta_list([f for f in files if not under(f, GRAPH)]) + sources)


@router.get("/all", response_model=FileListResponse)
def list_all_documents(store: ProjectStore = Depends(get_project_store)):
    """Every document referenced by any project."""
    return FileListResponse(documents=_meta_list(store.list_all_files()))


@router.get("/notes", response_model=FileListResponse)
def list_notes(docs: DataStore = Depends(get_document_store)) -> FileListResponse:
    """List non-project canvases/notes."""
    paths = [entry.key for entry in docs.list(NOTE) if not entry.is_dir]
    return FileListResponse(documents=_meta_list(paths))


@router.post("/write", response_model=FileMeta)
def write_document(
    req: WriteDocumentRequest,
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
    assets: AssetStore = Depends(get_asset_store),
):
    """Create or overwrite a document. Empty slug = non-project note,
    otherwise the project's documents/ directory."""
    name = _safe_filename(req.name)
    with http_errors(*_DOCUMENT_ERRORS):
        return file_meta(document_service.write(store, docs, assets, DocumentScope(req.slug), name, req.content))


@router.post("/write-binary", response_model=FileMeta)
def write_binary_document(
    file: UploadFile = File(...),
    slug: str = Query(...),
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
):
    """Write a binary file (e.g. PDF) to the project's documents/ directory."""
    with http_errors((FileNotFoundError, 404, "Project not found")):
        store.require(slug)
    key = project_document(slug, _upload_name(file))
    docs.write_bytes(key, file.file.read())
    store.add_file(slug, key)
    return file_meta(key)


@router.post("/store-drawing")
def store_drawing(
    file: UploadFile = File(...),
    assets: AssetStore = Depends(get_asset_store),
):
    """Store a rendered canvas image without creating a Nextcloud copy."""
    assets.write("drawing", f"{DRAWING}/{_upload_name(file)}", file.file.read())
    return {"status": "ok"}


@router.post("/add", response_model=FileMeta)
def add_document(
    req: AddDocumentRequest,
    slug: str = Query(...),
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
    assets: AssetStore = Depends(get_asset_store),
):
    """Assign an existing document (from the global pool/library) to a project."""
    key = _valid_key(req.path)
    if not (document_service.is_known(store, key) and document_service.is_stored(docs, assets, key)):
        raise HTTPException(status_code=404, detail=_DOCUMENT_NOT_FOUND)
    with http_errors((FileNotFoundError, 404, None)):
        store.add_file(slug, key)
    return file_meta(key)


@router.delete("")
def remove_document(
    slug: str = Query(...),
    path: str = Query(...),
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
):
    """Unassign a document from a project. If it lives in the project's
    documents/ directory, also delete the file itself. Empty slug = non-project
    note, deleted directly since there's no project to unlink from."""
    scope = DocumentScope(slug)
    with http_errors((FileNotFoundError, 404, None)):
        scope.unlink(store, path)
    scope.delete_stored(docs, path)
    return {"status": "ok"}


@router.post("/move", response_model=FileMeta)
def move_document(
    req: MoveDocumentRequest,
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
):
    """Move a document (e.g. a canvas) between projects, or to/from the
    unassigned root notes collection (empty slug = notes)."""
    if req.from_slug == req.to_slug:
        raise HTTPException(status_code=400, detail="Source and destination are the same")
    key = _valid_key(req.path)
    with http_errors(*_DOCUMENT_ERRORS):
        return file_meta(document_service.move(store, docs, key, DocumentScope(req.from_slug), DocumentScope(req.to_slug)))


@router.post("/rename", response_model=FileMeta)
def rename_document(
    req: RenameDocumentRequest,
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
    assets: AssetStore = Depends(get_asset_store),
):
    """Rename a database-owned document, retaining its content and project links."""
    key = _valid_key(req.path)
    with http_errors(*_DOCUMENT_ERRORS):
        return file_meta(document_service.rename(store, docs, assets, DocumentScope(req.slug), key, req.name))
