"""Document routes — the vault-tree document feature.

A *document* is a file a project references, stored as a database-native
logical key in ``project.json``'s ``files`` list. The add/upload flow tidies
PDFs into the shared library (``PDFs/<name>``); they are sent to MinerU so a
parsed-markdown preview is available. Selecting a document *attaches* it — a
copy is written into the chat's upload namespace so it behaves exactly like
any other attachment.

The UI only ever passes back opaque path strings the API previously handed it.

Those opaque strings are database-native artifact keys: documents are rows in
their typed namespaces and bytes are asset rows. Every read/write is scoped to
the request's user. PDFs and MinerU output are additionally mirrored into
Nextcloud as a convenience copy for other tools — the app itself never reads
that mirror back. All other application state remains database-only.
"""

import logging
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from backend.routes.deps import get_asset_store, get_document_store, get_project_store
from backend.routes.schemas import AttachResult, FileListResponse, FileMeta
from lib import document_library as lib_docs
from lib.asset_store import Asset, AssetStore
from lib.attachment_materialize import materialize, store_library_pdf
from lib.data_store import DataStore, DataStorePath, StorageNotFoundError, validate_key, write_text
from lib.project_store import ProjectStore
from lib.storage_namespace import (
    ATTACHMENT,
    DRAWING,
    GRAPH,
    NOTE,
    PDFS,
    chat_upload,
    project_attachment_prefix,
    project_document,
    project_document_prefix,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


KEY_DRAWINGS = DRAWING
KEY_GRAPHS = GRAPH
KEY_NOTES = NOTE
KEY_PDFS = PDFS


class WriteDocumentRequest(BaseModel):
    slug: str
    name: str
    content: str = ""


class AddDocumentRequest(BaseModel):
    path: str


def _meta_list(paths: list[str]) -> list[FileMeta]:
    # ponytail: canvas-pasted images stay registered (so /fileviewer can serve them)
    # but never surface as documents — they'd flood the sidebar and the canvas add menu.
    return [FileMeta(**lib_docs.document_meta(p)) for p in paths if not Path(p).name.startswith("pasted-")]


def _under(key: str, prefix: str) -> bool:
    return key == prefix or key.startswith(f"{prefix}/")


def _known_key(store: ProjectStore, path: str) -> str:
    """The logical key for *path*, 404 when it is not one the app knows about.

    Allow-list includes the document library, any document a project
    references, and canonical architecture graphs, expressed in logical keys.
    Logical keys cannot escape their application namespace, and anything unknown
    answers 404 so another user's documents stay undiscoverable.
    """
    try:
        key = validate_key(path)
    except ValueError:
        raise HTTPException(status_code=404, detail="Document not found") from None
    known = (
        _under(key, KEY_PDFS)
        # Graph drafts are deliberately not generic documents (see resolve_known_path).
        or (PurePosixPath(key).parent.as_posix() == KEY_GRAPHS and key.endswith(".architecture.yaml"))
        or key in set(store.list_all_files())
    )
    if not known:
        raise HTTPException(status_code=404, detail="Document not found")
    return key


def _exists_key(docs: DataStore, assets: AssetStore, key: str) -> bool:
    return DataStorePath(docs, key).is_file() or bool(assets.list(key))


def _read_key(docs: DataStore, assets: AssetStore, key: str) -> bytes:
    """Read a logical key from whichever store owns it, 404 when neither does."""
    try:
        return docs.read_bytes(key)[0]
    except StorageNotFoundError:
        pass
    try:
        return assets.read(key).content or b""
    except StorageNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found") from None


def _project_docs_key(slug: str) -> str:
    return project_document_prefix(slug)


def _uploads_key(slug: str | None) -> str:
    if slug:
        return project_attachment_prefix(slug)
    return ATTACHMENT


def _promote_pdf_document(assets: AssetStore, name: str, content: bytes) -> Asset:
    """Store *content* in the shared PDF library and enqueue MinerU extraction
    unless it is already cached."""
    asset = store_library_pdf(assets, name, content)
    materialize(name, content, assets, enqueue_on_miss=True)
    return asset


@router.get("", response_model=FileListResponse)
async def list_documents(slug: str = Query(...), store: ProjectStore = Depends(get_project_store)):
    try:
        return FileListResponse(documents=_meta_list(store.list_files(slug)))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/all", response_model=FileListResponse)
async def list_all_documents(
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
):
    graph_paths = [e.key for e in docs.list(KEY_GRAPHS) if not e.is_dir and e.key.endswith(".architecture.yaml")]
    return FileListResponse(documents=_meta_list(list(dict.fromkeys([*store.list_all_files(), *graph_paths]))))


@router.get("/notes", response_model=FileListResponse)
async def list_notes(docs: DataStore = Depends(get_document_store)) -> FileListResponse:
    """List non-project canvases/notes."""
    paths = [entry.key for entry in docs.list(KEY_NOTES) if not entry.is_dir]
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
    safe_name = Path(req.name).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if req.slug == "":
        key = f"{KEY_NOTES}/{safe_name}"
        write_text(docs, key, req.content)
        return FileMeta(**lib_docs.document_meta(key))

    meta = store._read_meta()
    if not store._find_entry(meta, req.slug):
        raise HTTPException(status_code=404, detail="Project not found")

    key = project_document(req.slug, safe_name)
    write_text(docs, key, req.content)
    store.add_file(req.slug, key)

    # Keep the active chat attachment copy in sync with its source document.
    content = req.content.encode("utf-8")
    for asset in assets.list(_uploads_key(req.slug), kind="upload"):
        if PurePosixPath(asset.logical_path).name == safe_name:
            assets.write("upload", asset.logical_path, content)

    return FileMeta(**lib_docs.document_meta(key))


@router.post("/write-binary", response_model=FileMeta)
async def write_binary_document(
    file: UploadFile = File(...),
    slug: str = Query(...),
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
):
    """Write a binary file (e.g. PDF) to the project's documents/ directory."""
    meta = store._read_meta()
    if not store._find_entry(meta, slug):
        raise HTTPException(status_code=404, detail="Project not found")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    key = project_document(slug, safe_name)
    docs.write_bytes(key, await file.read())
    store.add_file(slug, key)
    return FileMeta(**lib_docs.document_meta(key))


@router.post("/store-drawing")
async def store_drawing(
    file: UploadFile = File(...),
    assets: AssetStore = Depends(get_asset_store),
):
    """Store a rendered canvas image without creating a Nextcloud copy."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    assets.write("drawing", f"{KEY_DRAWINGS}/{safe_name}", await file.read())
    return {"status": "ok"}


@router.post("/upload", response_model=FileMeta)
async def upload_document(
    file: UploadFile = File(...),
    slug: str = Query(...),
    store: ProjectStore = Depends(get_project_store),
    assets: AssetStore = Depends(get_asset_store),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    try:
        store.list_files(slug)  # validates project exists
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    content_b = await file.read()
    name = Path(file.filename).name
    if Path(name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF documents can enter the PDF library")
    asset = _promote_pdf_document(assets, name, content_b)
    store.add_file(slug, asset.logical_path)
    return FileMeta(**lib_docs.document_meta(asset.logical_path))


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
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        store.add_file(slug, key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileMeta(**lib_docs.document_meta(key))


class RegisterUploadRequest(BaseModel):
    chat_id: str
    filename: str


@router.post("/register-upload", response_model=FileMeta)
async def register_upload(
    req: RegisterUploadRequest,
    slug: str | None = Query(default=None),
    store: ProjectStore = Depends(get_project_store),
    assets: AssetStore = Depends(get_asset_store),
):
    """Organize a chat-uploaded file into the library and optionally register it in a project."""
    source = chat_upload(req.chat_id, Path(req.filename).name, slug)
    try:
        upload = assets.read(source)
    except (StorageNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="Uploaded file not found") from None
    name = PurePosixPath(upload.logical_path).name
    if Path(name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF documents can enter the PDF library")
    asset = _promote_pdf_document(assets, name, upload.content or b"")
    if slug:
        store.add_file(slug, asset.logical_path)
    return FileMeta(**lib_docs.document_meta(asset.logical_path))


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
        _delete_under(docs, path, KEY_NOTES)
        return {"status": "ok"}

    try:
        store.remove_file(slug, path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _delete_under(docs, path, _project_docs_key(slug))
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

    try:
        key = validate_key(req.path)
    except ValueError:
        raise HTTPException(status_code=404, detail="Document not found") from None

    if req.from_slug == "":
        known = _under(key, KEY_NOTES)
    else:
        try:
            known = key in set(store.list_files(req.from_slug))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not known or not DataStorePath(docs, key).is_file():
        raise HTTPException(status_code=404, detail="Document not found")

    if req.to_slug:
        meta = store._read_meta()
        if not store._find_entry(meta, req.to_slug):
            raise HTTPException(status_code=404, detail="Project not found")
        dest_prefix = _project_docs_key(req.to_slug)
    else:
        dest_prefix = KEY_NOTES

    dest = f"{dest_prefix}/{PurePosixPath(key).name}"
    if dest != key:
        if docs.exists(dest):
            raise HTTPException(status_code=409, detail="A document with that name already exists there")
        docs.move(key, dest)

    if req.from_slug:
        store.remove_file(req.from_slug, key)
    if req.to_slug:
        store.add_file(req.to_slug, dest)

    return FileMeta(**lib_docs.document_meta(dest))


@router.post("/attach", response_model=AttachResult)
async def attach_document(
    path: str = Query(...),
    chat_id: str = Query(...),
    slug: str | None = Query(default=None),
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore = Depends(get_document_store),
    assets: AssetStore = Depends(get_asset_store),
):
    key = _known_key(store, path)
    content = _read_key(docs, assets, key)
    name = PurePosixPath(key).name
    assets.write("upload", chat_upload(chat_id, name, slug), content)

    result = AttachResult(name=name, mime=lib_docs.mime_for(key))
    if PurePosixPath(key).suffix.lower() == ".pdf":
        store_library_pdf(assets, name, content)
        result.parsedMd = materialize(name, content, assets, enqueue_on_miss=True)
    elif result.mime.startswith("text/"):
        try:
            result.content = content.decode("utf-8")
        except UnicodeDecodeError:
            result.content = None
    return result
