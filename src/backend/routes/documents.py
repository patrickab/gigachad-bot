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
from backend.routes.mineru import _parse_pdf
from config import DIRECTORY_OUTPUT_MINERU, chat_upload_dir
from lib import document_library as lib_docs
from lib.naming import dedup_filename
from lib.project_store import ProjectStore

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


class DocumentMeta(BaseModel):
    path: str
    name: str
    mime: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentMeta]


class DocumentAttachResult(BaseModel):
    name: str
    mime: str
    parsedMd: str | None = None
    content: str | None = None


class AddDocumentRequest(BaseModel):
    path: str


def _meta_list(paths: list[str]) -> list[DocumentMeta]:
    return [DocumentMeta(**lib_docs.document_meta(p)) for p in paths]


def _validate_doc_path(store: ProjectStore, path: str) -> Path:
    """Only allow paths the app already knows about (library or any project)."""
    resolved = Path(path).expanduser().resolve()
    library = lib_docs.LIBRARY_DIR.resolve()
    in_library = resolved.is_relative_to(library)
    known = in_library or str(resolved) in {str(Path(p).resolve()) for p in store.list_all_files()}
    if not known:
        raise HTTPException(status_code=403, detail="Unknown document path")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Document not found")
    return resolved


async def _parse_pdf_into_cache(pdf_path: Path) -> None:
    """Ensure a PDF has cached MinerU markdown so previews/attach work."""
    try:
        await _parse_pdf(pdf_path, DIRECTORY_OUTPUT_MINERU)
    except Exception:
        log.exception("MinerU parse failed for document %s", pdf_path.name)


@router.get("", response_model=DocumentListResponse)
async def list_documents(slug: str = Query(...), store: ProjectStore = Depends(get_project_store)):
    try:
        return DocumentListResponse(documents=_meta_list(store.list_files(slug)))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/all", response_model=DocumentListResponse)
async def list_all_documents(store: ProjectStore = Depends(get_project_store)):
    return DocumentListResponse(documents=_meta_list(store.list_all_files()))


@router.post("/upload", response_model=DocumentMeta)
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
        await _parse_pdf_into_cache(dest)

    store.add_file(slug, str(dest))
    return DocumentMeta(**lib_docs.document_meta(dest))


@router.post("/add", response_model=DocumentMeta)
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
    return DocumentMeta(**lib_docs.document_meta(resolved))


class RegisterUploadRequest(BaseModel):
    chat_id: str
    filename: str


@router.post("/register-upload", response_model=DocumentMeta)
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
    return DocumentMeta(**lib_docs.document_meta(dest))


@router.delete("")
async def remove_document(
    slug: str = Query(...),
    path: str = Query(...),
    store: ProjectStore = Depends(get_project_store),
):
    """Unassign a document from a project. The file itself stays in the library."""
    try:
        store.remove_file(slug, path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ok"}


@router.post("/attach", response_model=DocumentAttachResult)
async def attach_document(
    path: str = Query(...),
    chat_id: str = Query(...),
    slug: str | None = Query(default=None),
    store: ProjectStore = Depends(get_project_store),
):
    resolved = _validate_doc_path(store, path)
    chat_dir = chat_upload_dir(chat_id, slug)
    chat_dir.mkdir(parents=True, exist_ok=True)
    name = dedup_filename(chat_dir, resolved.name)
    dest = chat_dir / name
    dest.write_bytes(resolved.read_bytes())

    mime = lib_docs.mime_for(resolved)
    result = DocumentAttachResult(name=name, mime=mime)

    if resolved.suffix.lower() == ".pdf":
        try:
            md_path, _images = await _parse_pdf(dest, chat_dir)
            result.parsedMd = md_path.read_text(encoding="utf-8")
        except Exception:
            log.exception("MinerU parse failed attaching document %s", name)
    elif mime.startswith("text/"):
        try:
            result.content = dest.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            result.content = None

    return result
