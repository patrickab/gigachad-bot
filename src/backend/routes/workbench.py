"""Workbench files on disk (`lib.workbench`) and the omp session of each, for logins with host access."""

from pathlib import Path, PurePosixPath

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.http_plumbing import http_errors
from backend.identity import host_access_only
from backend.routes.deps import get_asset_store, get_document_store, get_file_roots, get_project_store, get_relocate
from backend.routes.fileviewer import resolve_allowed
import config
from lib import document_service, latex
from lib.asset_store import AssetStore
from lib.data_store import DataStore, InvalidStorageKey, StorageConflictError, StorageNotFoundError, validate_key
from lib.document_service import DocumentNameConflict, DocumentNotFound, Relocate, under
from lib.file_roots import FileRoots
from lib.json_docs import JsonDocs
from lib.project_store import ProjectStore
from lib.storage_namespace import NOTE, PDFS, WORKBENCH_SESSIONS, library_pdf
from lib.workbench import Kind, document_folder, kind_directory, kind_of, new_folder, suffix_of

router = APIRouter(prefix="/api/workbench", tags=["workbench"], dependencies=[host_access_only("Workbench")])

_CONTENT: dict[Kind, str] = {"latex": latex.NEW_DOCUMENT, "markdown": ""}


class FilePath(BaseModel):
    """A project document key (`project/<slug>/document/<name>`) or an absolute disk path."""

    path: str


class PreparedFile(BaseModel):
    """The file on disk, the omp session cwd of its kind and the kind."""

    path: str
    cwd: str
    kind: Kind


class NewDocument(BaseModel):
    slug: str
    name: str


class SessionLink(BaseModel):
    """Make `session_id` the active omp session of the absolute file `path`, None starts a new one."""

    path: str
    session_id: str | None


class LinkedSession(BaseModel):
    """The active session (None: a new one) and every session the file had, first linked first."""

    session_id: str | None
    session_ids: list[str]


def _linked(value: object) -> LinkedSession:
    """The stored link of one file: `{"active", "sessions"}`, or a bare session id as stored before sessions were kept."""
    if isinstance(value, str):
        return LinkedSession(session_id=value, session_ids=[value])
    if not isinstance(value, dict):
        return LinkedSession(session_id=None, session_ids=[])
    sessions = [session for session in value.get("sessions", []) if isinstance(session, str)]
    active = value.get("active")
    return LinkedSession(session_id=active if isinstance(active, str) else None, session_ids=sessions)


def _kind(name: str) -> Kind:
    kind = kind_of(name)
    if kind is None:
        raise HTTPException(status_code=400, detail="The workbench edits only .tex and .md files")
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

    - A failure to list it removes the new file and the folders made for it
    """
    name = body.name.strip()
    kind = _kind(name)
    if kind == "pdf":
        raise HTTPException(status_code=400, detail="New documents are .md or .tex")
    with http_errors((FileNotFoundError, 404, "Project not found"), (ValueError, 400, None)):
        store.require(body.slug)
        folder = document_folder(config.WORKBENCH, kind, body.slug, name.removesuffix(Path(name).suffix))
    source = folder / f"{folder.name}{suffix_of(kind)}"
    with new_folder(folder):
        conflict = "A document with that name already exists"
        with http_errors((FileExistsError, 409, conflict)), source.open("x", encoding="utf-8") as file:
            file.write(_CONTENT[kind])
        try:
            store.add_file(body.slug, str(source))
        except BaseException:
            source.unlink(missing_ok=True)
            raise
    return FilePath(path=str(source))


@router.get("/session")
def get_session(path: str = Query(...), docs: DataStore = Depends(get_document_store)) -> LinkedSession:
    """The omp sessions linked to the absolute file `path`."""
    links = JsonDocs(docs).read(WORKBENCH_SESSIONS, {})
    return _linked(links.get(path) if isinstance(links, dict) else None)


@router.put("/session")
def put_session(body: SessionLink, docs: DataStore = Depends(get_document_store)) -> LinkedSession:
    """Make `session_id` the active session of the absolute file `path`, keeping every earlier one. None starts a new session."""
    if not PurePosixPath(body.path).is_absolute():
        raise HTTPException(status_code=400, detail="Path must be absolute")
    json_docs = JsonDocs(docs)
    try:
        links, revision = json_docs.read_with_revision(WORKBENCH_SESSIONS)
    except (StorageNotFoundError, ValueError):
        links, revision = {}, None
    links = links if isinstance(links, dict) else {}
    current = _linked(links.get(body.path))
    sessions = current.session_ids
    if body.session_id is not None and body.session_id not in sessions:
        sessions = [*sessions, body.session_id]
    linked = LinkedSession(session_id=body.session_id, session_ids=sessions)
    if linked != current or not isinstance(links.get(body.path), dict):
        stored = {"active": linked.session_id, "sessions": linked.session_ids}
        json_docs.write(WORKBENCH_SESSIONS, {**links, body.path: stored}, expected=revision)
    return linked
