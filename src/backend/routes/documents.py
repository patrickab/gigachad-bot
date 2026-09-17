"""Document routes — the vault-tree document feature.

A *document* is an arbitrary file a project references (stored as an absolute
path in ``project.json``'s ``files`` list). The add/upload flow tidies files
into the central library (``DIRECTORY_OUTPUT_PDF``); PDFs are sent to MinerU so a
parsed-markdown preview is available. Selecting a document *attaches* it — a
copy is materialised into the chat's upload directory so it behaves exactly like
any other attachment.

The API owns every filesystem access; the UI only ever passes back opaque path
strings the API previously handed it.

In Postgres mode those opaque strings are logical storage keys instead of
absolute paths: text documents are ``documents`` rows (``chat_history/_notes/x.md``),
bytes are ``assets`` rows, and every read/write is scoped to the request's user.
PDFs, MinerU markdown and drawings are additionally mirrored into the browsable
Nextcloud tree after their database write commits.
"""

import logging
from pathlib import Path, PurePosixPath
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from backend.routes.deps import get_asset_store, get_document_store, get_project_store
from backend.routes.schemas import AttachResult, FileListResponse, FileMeta
from config import (
    DIRECTORY_CHAT_HISTORIES,
    DIRECTORY_CHAT_UPLOADS,
    DIRECTORY_NOTES,
    DIRECTORY_OUTPUT_ARCHITECTURE_GRAPHS,
    DIRECTORY_OUTPUT_DRAWINGS,
    DIRECTORY_OUTPUT_LATEX,
    DIRECTORY_OUTPUT_MARKDOWN,
    DIRECTORY_OUTPUT_MINERU,
    DIRECTORY_OUTPUT_PDF,
    DOCUMENTS,
    chat_upload_dir,
)
from lib import document_library as lib_docs
from lib import extract_queue
from lib.asset_store import Asset, AssetStore
from lib.attachment_materialize import materialize
from lib.data_store import DataStore, DataStorePath, StorageNotFoundError, validate_key, write_text
from lib.project_store import ProjectStore

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _key_of(directory: Path) -> str:
    """The logical storage key of a configured ``Documents`` subdirectory."""
    return directory.relative_to(DOCUMENTS).as_posix()


KEY_CHATS = _key_of(DIRECTORY_CHAT_HISTORIES)
KEY_CHAT_UPLOADS = _key_of(DIRECTORY_CHAT_UPLOADS)
KEY_DRAWINGS = _key_of(DIRECTORY_OUTPUT_DRAWINGS)
KEY_GRAPHS = _key_of(DIRECTORY_OUTPUT_ARCHITECTURE_GRAPHS)
KEY_MINERU = _key_of(DIRECTORY_OUTPUT_MINERU)
KEY_NOTES = _key_of(DIRECTORY_NOTES)
KEY_PDFS = _key_of(DIRECTORY_OUTPUT_PDF)
# Cloud collections a saved document is mirrored into, by suffix.
KEY_MIRRORS = {".md": _key_of(DIRECTORY_OUTPUT_MARKDOWN), ".tex": _key_of(DIRECTORY_OUTPUT_LATEX)}
_UPLOADS_DIR = DIRECTORY_CHAT_UPLOADS.name


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


def _under(key: str, prefix: str) -> bool:
    return key == prefix or key.startswith(f"{prefix}/")


def _known_key(store: ProjectStore, path: str) -> str:
    """Storage-key twin of ``lib_docs.resolve_known_path``.

    Same allow-list — the document library, any document a project already
    references, and the canonical architecture graphs — expressed in logical
    keys. Strictly stricter than the filesystem version in two ways: a key can
    never leave the ``Documents`` root, and anything unknown answers 404 instead
    of 403 so another user's documents stay undiscoverable.

    # TODO(phase5): fold back into ``lib_docs.resolve_known_path`` once the
    # filesystem branch is gone, so one function owns the allow-list again.
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
    return f"{KEY_CHATS}/{slug}/documents"


def _uploads_key(slug: str | None) -> str:
    """Storage-key twin of ``config.uploads_dir_for``."""
    return f"{KEY_CHATS}/{slug}/{_UPLOADS_DIR}" if slug else KEY_CHAT_UPLOADS


def _chat_upload_key(chat_id: str, slug: str | None = None) -> str:
    """Storage-key twin of ``config.chat_upload_dir``."""
    return f"{_uploads_key(slug)}/{chat_id}"


def _mirror(assets: AssetStore, asset: Asset) -> Path | None:
    """Mirror a committed asset into the browsable Nextcloud tree.

    The database row is authoritative: a failed mirror is a missing convenience
    copy, logged and retried on the next write, never a lost document.
    """
    try:
        return assets.mirror(asset, DOCUMENTS)
    except OSError:
        log.exception("Failed to mirror %s into %s", asset.logical_path, DOCUMENTS)
        return None


def _library_asset(assets: AssetStore, name: str, content: bytes) -> Asset:
    """Storage twin of ``lib_docs.organize_file``: the library copy is a ``pdf`` asset."""
    if not name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return assets.write("pdf", f"{KEY_PDFS}/{name}", content)


def _parsed_markdown(assets: AssetStore, name: str) -> str | None:
    """Cached MinerU markdown for a PDF, mirroring ``attachment_materialize``'s lookup."""
    stem = name[:-4] if name.lower().endswith(".pdf") else name
    try:
        return (assets.read(f"{KEY_MINERU}/{stem}.md").content or b"").decode("utf-8")
    except StorageNotFoundError:
        return None


@router.get("", response_model=FileListResponse)
async def list_documents(slug: str = Query(...), store: ProjectStore = Depends(get_project_store)):
    try:
        return FileListResponse(documents=_meta_list(store.list_files(slug)))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/all", response_model=FileListResponse)
async def list_all_documents(
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore | None = Depends(get_document_store),
):
    if docs is None:
        graph_paths = [str(path) for path in DIRECTORY_OUTPUT_ARCHITECTURE_GRAPHS.glob("*.architecture.yaml") if path.is_file()]
    else:
        graph_paths = [e.key for e in docs.list(KEY_GRAPHS) if not e.is_dir and e.key.endswith(".architecture.yaml")]
    return FileListResponse(documents=_meta_list(list(dict.fromkeys([*store.list_all_files(), *graph_paths]))))


@router.get("/notes", response_model=FileListResponse)
async def list_notes(docs: DataStore | None = Depends(get_document_store)) -> FileListResponse:
    """List non-project canvases/notes (DIRECTORY_NOTES)."""
    if docs is None:
        paths = [str(p) for p in DIRECTORY_NOTES.iterdir() if p.is_file()]
    else:
        paths = [entry.key for entry in docs.list(KEY_NOTES) if not entry.is_dir]
    return FileListResponse(documents=_meta_list(paths))


def _write_document_key(
    req: WriteDocumentRequest, safe_name: str, store: ProjectStore, docs: DataStore, assets: AssetStore
) -> FileMeta:
    """Storage twin of ``write_document``'s filesystem body."""
    if req.slug == "":
        key = f"{KEY_NOTES}/{safe_name}"
        write_text(docs, key, req.content)
        return FileMeta(**lib_docs.document_meta(key))

    meta = store._read_meta()
    if not store._find_entry(meta, req.slug):
        raise HTTPException(status_code=404, detail="Project not found")

    key = f"{_project_docs_key(req.slug)}/{safe_name}"
    write_text(docs, key, req.content)
    store.add_file(req.slug, key)

    # keep _uploads copies in sync so attached context stays current
    content = req.content.encode("utf-8")
    for asset in assets.list(_uploads_key(req.slug), kind="upload"):
        if PurePosixPath(asset.logical_path).name == safe_name:
            assets.write("upload", asset.logical_path, content)

    # mirror into the browsable cloud collection (overwrite by name).
    # canvas → .jpg is rendered client-side; only md/tex mirror here.
    mirror_prefix = KEY_MIRRORS.get(PurePosixPath(safe_name).suffix.lower())
    if mirror_prefix:
        write_text(docs, f"{mirror_prefix}/{safe_name}", req.content)

    return FileMeta(**lib_docs.document_meta(key))


@router.post("/write", response_model=FileMeta)
async def write_document(
    req: WriteDocumentRequest,
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore | None = Depends(get_document_store),
    assets: AssetStore | None = Depends(get_asset_store),
):
    """Create or overwrite a document. Empty slug = non-project note (DIRECTORY_NOTES),
    otherwise the project's documents/ directory."""
    safe_name = Path(req.name).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if docs is not None and assets is not None:
        return _write_document_key(req, safe_name, store, docs, assets)

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
    docs: DataStore | None = Depends(get_document_store),
):
    """Write a binary file (e.g. PDF) to the project's documents/ directory."""
    meta = store._read_meta()
    if not store._find_entry(meta, slug):
        raise HTTPException(status_code=404, detail="Project not found")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    if docs is not None:
        safe_name = Path(file.filename).name
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid filename")
        key = f"{_project_docs_key(slug)}/{safe_name}"
        docs.write_bytes(key, await file.read())
        store.add_file(slug, key)
        return FileMeta(**lib_docs.document_meta(key))
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
async def mirror_drawing(
    file: UploadFile = File(...),
    assets: AssetStore | None = Depends(get_asset_store),
):
    """Mirror a rendered canvas (.jpg) into the browsable cloud Drawings dir,
    overwriting any same-named drawing. The raw .canvas never lands here."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if assets is not None:
        _mirror(assets, assets.write("drawing", f"{KEY_DRAWINGS}/{safe_name}", await file.read()))
        return {"status": "ok"}
    DIRECTORY_OUTPUT_DRAWINGS.mkdir(parents=True, exist_ok=True)
    (DIRECTORY_OUTPUT_DRAWINGS / safe_name).write_bytes(await file.read())
    return {"status": "ok"}


@router.post("/upload", response_model=FileMeta)
async def upload_document(
    file: UploadFile = File(...),
    slug: str = Query(...),
    store: ProjectStore = Depends(get_project_store),
    assets: AssetStore | None = Depends(get_asset_store),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    try:
        store.list_files(slug)  # validates project exists
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    content_b = await file.read()
    if assets is not None:
        asset = _library_asset(assets, Path(file.filename).name, content_b)
        mirrored = _mirror(assets, asset)
        # MinerU parses a real file, so it reads the mirror this write just refreshed.
        if mirrored is not None and mirrored.suffix.lower() == ".pdf":
            extract_queue.enqueue(mirrored)
        store.add_file(slug, asset.logical_path)
        return FileMeta(**lib_docs.document_meta(asset.logical_path))

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
    docs: DataStore | None = Depends(get_document_store),
    assets: AssetStore | None = Depends(get_asset_store),
):
    """Assign an existing document (from the global pool/library) to a project."""
    if docs is not None and assets is not None:
        key = _known_key(store, req.path)
        if not _exists_key(docs, assets, key):
            raise HTTPException(status_code=404, detail="Document not found")
        try:
            store.add_file(slug, key)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileMeta(**lib_docs.document_meta(key))

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
    assets: AssetStore | None = Depends(get_asset_store),
):
    """Organize a chat-uploaded file into the library and optionally register it in a project."""
    if assets is not None:
        source = f"{_chat_upload_key(req.chat_id, slug)}/{Path(req.filename).name}"
        try:
            upload = assets.read(source)
        except (StorageNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail="Uploaded file not found") from None
        asset = _library_asset(assets, PurePosixPath(upload.logical_path).name, upload.content or b"")
        _mirror(assets, asset)
        if slug:
            store.add_file(slug, asset.logical_path)
        return FileMeta(**lib_docs.document_meta(asset.logical_path))

    src = chat_upload_dir(req.chat_id, slug) / req.filename
    if not src.is_file():
        raise HTTPException(status_code=404, detail="Uploaded file not found")
    dest = lib_docs.organize_file(src)
    if slug:
        store.add_file(slug, str(dest))
    return FileMeta(**lib_docs.document_meta(dest))


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
    docs: DataStore | None = Depends(get_document_store),
):
    """Unassign a document from a project. If it lives in the project's
    documents/ directory, also delete the file itself. Empty slug = non-project
    note (DIRECTORY_NOTES), deleted directly since there's no project to unlink from."""
    if slug == "":
        if docs is not None:
            _delete_under(docs, path, KEY_NOTES)
            return {"status": "ok"}
        resolved = Path(path).expanduser().resolve()
        if resolved.is_relative_to(DIRECTORY_NOTES.resolve()) and resolved.is_file():
            resolved.unlink()
        return {"status": "ok"}

    try:
        store.remove_file(slug, path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if docs is not None:
        _delete_under(docs, path, _project_docs_key(slug))
        return {"status": "ok"}
    resolved = Path(path).expanduser().resolve()
    docs_dir = (DIRECTORY_CHAT_HISTORIES / slug / "documents").resolve()
    if resolved.is_relative_to(docs_dir) and resolved.is_file():
        resolved.unlink()
    return {"status": "ok"}


class MoveDocumentRequest(BaseModel):
    path: str
    from_slug: str = ""
    to_slug: str = ""


def _move_document_key(req: MoveDocumentRequest, store: ProjectStore, docs: DataStore) -> FileMeta:
    """Storage twin of ``move_document``'s filesystem body."""
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


@router.post("/move", response_model=FileMeta)
async def move_document(
    req: MoveDocumentRequest,
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore | None = Depends(get_document_store),
):
    """Move a document (e.g. a canvas) between projects, or to/from the
    unassigned root notes collection (empty slug = DIRECTORY_NOTES).
    Physically relocates the file and updates the source/destination
    project file registries — DIRECTORY_NOTES itself isn't registry-backed."""
    if req.from_slug == req.to_slug:
        raise HTTPException(status_code=400, detail="Source and destination are the same")

    if docs is not None:
        return _move_document_key(req, store, docs)

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


def _attach_key(
    path: str, chat_id: str, slug: str | None, store: ProjectStore, docs: DataStore, assets: AssetStore
) -> AttachResult:
    """Storage twin of ``attach_document``'s filesystem body."""
    key = _known_key(store, path)
    content = _read_key(docs, assets, key)
    name = PurePosixPath(key).name
    assets.write("upload", f"{_chat_upload_key(chat_id, slug)}/{name}", content)

    result = AttachResult(name=name, mime=lib_docs.mime_for(key))
    if PurePosixPath(key).suffix.lower() == ".pdf":
        result.parsedMd = _parsed_markdown(assets, name)
        try:
            mirrored = _mirror(assets, _library_asset(assets, name, content))
        except Exception:
            log.exception("Failed to organize %s into PDF library", name)
        else:
            if result.parsedMd is None and mirrored is not None:
                extract_queue.enqueue(mirrored)
    elif result.mime.startswith("text/"):
        try:
            result.content = content.decode("utf-8")
        except UnicodeDecodeError:
            result.content = None
    return result


@router.post("/attach", response_model=AttachResult)
async def attach_document(
    path: str = Query(...),
    chat_id: str = Query(...),
    slug: str | None = Query(default=None),
    store: ProjectStore = Depends(get_project_store),
    docs: DataStore | None = Depends(get_document_store),
    assets: AssetStore | None = Depends(get_asset_store),
):
    if docs is not None and assets is not None:
        return _attach_key(path, chat_id, slug, store, docs, assets)

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
