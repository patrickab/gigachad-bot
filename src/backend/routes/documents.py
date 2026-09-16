"""Document routes — the vault-tree document feature.

A *document* is an arbitrary file a project references (stored as an absolute
path in ``project.json``'s ``files`` list). The add/upload flow tidies files
into the central library (``DIRECTORY_OUTPUT_PDF``); PDFs are sent to MinerU so a
parsed-markdown preview is available. Selecting a document *attaches* it — a
copy is materialised into the chat's upload directory so it behaves exactly like
any other attachment.

The API owns every filesystem access; the UI only ever passes back opaque path
strings the API previously handed it.
"""

import logging
from pathlib import Path
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from backend.routes.deps import get_project_store
from backend.routes.schemas import AttachResult, FileListResponse, FileMeta
from config import (
    DIRECTORY_CHAT_HISTORIES,
    DIRECTORY_NOTES,
    DIRECTORY_OUTPUT_ARCHITECTURE_GRAPHS,
    DIRECTORY_OUTPUT_DRAWINGS,
    DIRECTORY_OUTPUT_LATEX,
    DIRECTORY_OUTPUT_MARKDOWN,
    chat_upload_dir,
)
from lib import document_library as lib_docs
from lib import extract_queue
from lib.attachment_materialize import materialize
from lib.project_store import ProjectStore

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


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


def _validate_doc_path(store: ProjectStore, path: str) -> Path:
    """Only allow paths the app already knows about (library or any project)."""
    try:
        return lib_docs.resolve_known_path(path, store=store)
    except lib_docs.PathNotAllowed as exc:
        if exc.outside_roots:
            raise HTTPException(status_code=403, detail="Unknown document path") from exc
        raise HTTPException(status_code=404, detail="Document not found") from exc


@router.get("", response_model=FileListResponse)
async def list_documents(slug: str = Query(...), store: ProjectStore = Depends(get_project_store)):
    try:
        return FileListResponse(documents=_meta_list(store.list_files(slug)))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/all", response_model=FileListResponse)
async def list_all_documents(store: ProjectStore = Depends(get_project_store)):
    graph_paths = [str(path) for path in DIRECTORY_OUTPUT_ARCHITECTURE_GRAPHS.glob("*.architecture.yaml") if path.is_file()]
    return FileListResponse(documents=_meta_list(list(dict.fromkeys([*store.list_all_files(), *graph_paths]))))


@router.get("/notes", response_model=FileListResponse)
async def list_notes() -> FileListResponse:
    """List non-project canvases/notes (DIRECTORY_NOTES)."""
    paths = [str(p) for p in DIRECTORY_NOTES.iterdir() if p.is_file()]
    return FileListResponse(documents=_meta_list(paths))


@router.post("/write", response_model=FileMeta)
async def write_document(
    req: WriteDocumentRequest,
    store: ProjectStore = Depends(get_project_store),
):
    """Create or overwrite a document. Empty slug = non-project note (DIRECTORY_NOTES),
    otherwise the project's documents/ directory."""
    safe_name = Path(req.name).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if req.slug == "":
        dest = DIRECTORY_NOTES / safe_name
        dest.write_text(req.content, encoding="utf-8")
        return FileMeta(**lib_docs.document_meta(dest))

    meta = store._read_meta()
    if not store._find_entry(meta, req.slug):
        raise HTTPException(status_code=404, detail="Project not found")
    # ProjectStore paths are storage keys. Routes write local project documents here.
    docs_dir = DIRECTORY_CHAT_HISTORIES / req.slug / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    dest = docs_dir / safe_name
    dest.write_text(req.content, encoding="utf-8")

    abs_path = str(dest.resolve())
    store.add_file(req.slug, abs_path)

    # keep _uploads copies in sync so attached context stays current
    project_dir = DIRECTORY_CHAT_HISTORIES / req.slug
    for uploads in project_dir.glob("*/_uploads"):
        copy = uploads / safe_name
        if copy.is_file():
            copy.write_text(req.content, encoding="utf-8")

    # mirror into the browsable cloud collection (overwrite by name).
    # canvas → .jpg is rendered client-side; only md/tex mirror here.
    mirror_dir = {".md": DIRECTORY_OUTPUT_MARKDOWN, ".tex": DIRECTORY_OUTPUT_LATEX}.get(dest.suffix.lower())
    if mirror_dir:
        mirror_dir.mkdir(parents=True, exist_ok=True)
        (mirror_dir / safe_name).write_text(req.content, encoding="utf-8")

    return FileMeta(**lib_docs.document_meta(dest))


@router.post("/write-binary", response_model=FileMeta)
async def write_binary_document(
    file: UploadFile = File(...),
    slug: str = Query(...),
    store: ProjectStore = Depends(get_project_store),
):
    """Write a binary file (e.g. PDF) to the project's documents/ directory."""
    meta = store._read_meta()
    if not store._find_entry(meta, slug):
        raise HTTPException(status_code=404, detail="Project not found")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    docs_dir = DIRECTORY_CHAT_HISTORIES / slug / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    dest = docs_dir / safe_name
    dest.write_bytes(await file.read())
    store.add_file(slug, str(dest.resolve()))
    return FileMeta(**lib_docs.document_meta(dest))


@router.post("/mirror-drawing")
async def mirror_drawing(file: UploadFile = File(...)):
    """Mirror a rendered canvas (.jpg) into the browsable cloud Drawings dir,
    overwriting any same-named drawing. The raw .canvas never lands here."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    DIRECTORY_OUTPUT_DRAWINGS.mkdir(parents=True, exist_ok=True)
    (DIRECTORY_OUTPUT_DRAWINGS / safe_name).write_bytes(await file.read())
    return {"status": "ok"}


@router.post("/upload", response_model=FileMeta)
async def upload_document(
    file: UploadFile = File(...),
    slug: str = Query(...),
    store: ProjectStore = Depends(get_project_store),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    try:
        store.list_files(slug)  # validates project exists
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    content_b = await file.read()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / file.filename
        tmp_path.write_bytes(content_b)
        dest = lib_docs.organize_file(tmp_path)

    if dest.suffix.lower() == ".pdf":
        extract_queue.enqueue(dest)

    store.add_file(slug, str(dest))
    return FileMeta(**lib_docs.document_meta(dest))


@router.post("/add", response_model=FileMeta)
async def add_document(
    req: AddDocumentRequest,
    slug: str = Query(...),
    store: ProjectStore = Depends(get_project_store),
):
    """Assign an existing document (from the global pool/library) to a project."""
    try:
        resolved = _validate_doc_path(store, req.path)
        store.add_file(slug, str(resolved))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileMeta(**lib_docs.document_meta(resolved))


class RegisterUploadRequest(BaseModel):
    chat_id: str
    filename: str


@router.post("/register-upload", response_model=FileMeta)
async def register_upload(
    req: RegisterUploadRequest,
    slug: str | None = Query(default=None),
    store: ProjectStore = Depends(get_project_store),
):
    """Organize a chat-uploaded file into the library and optionally register it in a project."""
    src = chat_upload_dir(req.chat_id, slug) / req.filename
    if not src.is_file():
        raise HTTPException(status_code=404, detail="Uploaded file not found")
    dest = lib_docs.organize_file(src)
    if slug:
        store.add_file(slug, str(dest))
    return FileMeta(**lib_docs.document_meta(dest))


@router.delete("")
async def remove_document(
    slug: str = Query(...),
    path: str = Query(...),
    store: ProjectStore = Depends(get_project_store),
):
    """Unassign a document from a project. If it lives in the project's
    documents/ directory, also delete the file itself. Empty slug = non-project
    note (DIRECTORY_NOTES), deleted directly since there's no project to unlink from."""
    if slug == "":
        resolved = Path(path).expanduser().resolve()
        if resolved.is_relative_to(DIRECTORY_NOTES.resolve()) and resolved.is_file():
            resolved.unlink()
        return {"status": "ok"}

    try:
        store.remove_file(slug, path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    resolved = Path(path).expanduser().resolve()
    docs_dir = (DIRECTORY_CHAT_HISTORIES / slug / "documents").resolve()
    if resolved.is_relative_to(docs_dir) and resolved.is_file():
        resolved.unlink()
    return {"status": "ok"}


class MoveDocumentRequest(BaseModel):
    path: str
    from_slug: str = ""
    to_slug: str = ""


@router.post("/move", response_model=FileMeta)
async def move_document(
    req: MoveDocumentRequest,
    store: ProjectStore = Depends(get_project_store),
):
    """Move a document (e.g. a canvas) between projects, or to/from the
    unassigned root notes collection (empty slug = DIRECTORY_NOTES).
    Physically relocates the file and updates the source/destination
    project file registries — DIRECTORY_NOTES itself isn't registry-backed."""
    if req.from_slug == req.to_slug:
        raise HTTPException(status_code=400, detail="Source and destination are the same")

    resolved = Path(req.path).expanduser().resolve()
    if req.from_slug == "":
        known = resolved.is_relative_to(DIRECTORY_NOTES.resolve())
    else:
        try:
            known = str(resolved) in {str(Path(p).resolve()) for p in store.list_files(req.from_slug)}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not known or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Document not found")

    if req.to_slug:
        meta = store._read_meta()
        if not store._find_entry(meta, req.to_slug):
            raise HTTPException(status_code=404, detail="Project not found")
        dest_dir = DIRECTORY_CHAT_HISTORIES / req.to_slug / "documents"
    else:
        dest_dir = DIRECTORY_NOTES
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / resolved.name
    if dest != resolved:
        if dest.exists():
            raise HTTPException(status_code=409, detail="A document with that name already exists there")
        resolved.replace(dest)

    if req.from_slug:
        store.remove_file(req.from_slug, str(resolved))
    if req.to_slug:
        store.add_file(req.to_slug, str(dest.resolve()))

    return FileMeta(**lib_docs.document_meta(dest))


@router.post("/attach", response_model=AttachResult)
async def attach_document(
    path: str = Query(...),
    chat_id: str = Query(...),
    slug: str | None = Query(default=None),
    store: ProjectStore = Depends(get_project_store),
):
    resolved = _validate_doc_path(store, path)
    chat_dir = chat_upload_dir(chat_id, slug)
    chat_dir.mkdir(parents=True, exist_ok=True)
    name = resolved.name
    dest = chat_dir / name
    dest.write_bytes(resolved.read_bytes())

    materialized = materialize(resolved)
    result = AttachResult(name=name, mime=materialized.mime, parsedMd=materialized.parsed_md, content=materialized.content)

    if resolved.suffix.lower() == ".pdf":
        try:
            lib_docs.organize_file(resolved)
        except Exception:
            log.exception("Failed to organize %s into PDF library", name)

    return result
