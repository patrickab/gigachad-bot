"""Generic file-viewer routes backing the reusable ``FileViewer`` primitive.

The frontend ``FileViewer`` is handed a flat list of paths and previews each
by kind: images and PDFs stream their bytes from ``/raw``; markdown and
unknown (treated as text) files read their content from ``/text``. PDF
annotations live in a canvas keyed by the PDF's content hash, see ``/annotations``.

App-owned files (documents, canvases, PDFs, MinerU output) are read from the
request's storage. Postgres is authoritative, so serving from disk would
hand every caller the last filesystem write instead of the current one. Only
a live FileRoots reference, which was never stored in Postgres, is read off
disk, and only after storage has been checked and found not to own the path.

``/raw`` revalidates on every open: an unchanged file answers 304 from its
metadata alone, and a byte range (pdf.js fetches the pages it renders first)
reads only that slice. Handlers are plain ``def``, so their blocking storage
reads run in the threadpool instead of stalling the event loop.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from backend.http_plumbing import http_errors
from backend.routes.deps import get_asset_store, get_document_store, get_file_roots
from backend.routes.schemas import FileContent
from lib import document_library as lib_docs
from lib.architecture_workspace import ArchitectureError, ArchitectureStore, split_source_path
from lib.asset_store import Asset, AssetStore
from lib.data_store import DataStore, Entry, StorageNotFoundError, read_text
from lib.file_roots import FileRoots
from lib.mime import mime_for
from lib.references import annotation_canvas

router = APIRouter(prefix="/api/fileviewer", tags=["fileviewer"])

# Cacheable per browser, but revalidated on every use, so an edit shows on the next open.
_CACHE_CONTROL = "private, no-cache"
_SINGLE_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


def _resolve_allowed(path: str, roots: FileRoots) -> Path:
    try:
        return lib_docs.resolve_known_path(path, roots=roots)
    except lib_docs.PathNotAllowed as exc:
        if exc.outside_roots:
            raise HTTPException(status_code=403, detail="Unknown file path") from exc
        raise HTTPException(status_code=404, detail="File not found") from exc


def _stored(path: str, docs: DataStore) -> str | None:
    """The stored logical key for *path*, or None when storage does not hold it."""
    key = lib_docs.storage_key(path)
    return key if key is not None and docs.exists(key) else None


@router.get("/text", response_model=FileContent)
def read_text_content(
    path: str = Query(...),
    roots: FileRoots = Depends(get_file_roots),
    docs: DataStore = Depends(get_document_store),
) -> FileContent:
    if architecture := split_source_path(path):
        slug, relative = architecture
        with http_errors(((StorageNotFoundError, ArchitectureError), 404, "File not found")):
            return FileContent(path=path, content=ArchitectureStore.for_project(docs, slug, roots.roots()).read_source(relative))
    key = _stored(path, docs)
    if key is not None:
        with http_errors((StorageNotFoundError, 404, "File not found")):
            content, _ = read_text(docs, key)
        return FileContent(path=path, content=content)

    resolved = _resolve_allowed(path, roots)
    # Strict decode: a lossy one would let the editor save replacement characters over a binary file.
    with http_errors(
        (UnicodeDecodeError, 415, "Not a UTF-8 text file"),
        (PermissionError, 403, "Unknown file path"),
        (OSError, 500, None),
    ):
        content = resolved.read_text(encoding="utf-8")
    return FileContent(path=path, content=content)


def _not_modified(request: Request, etag: str) -> Response | None:
    """A 304 when the request's ``If-None-Match`` lists *etag* (weak comparison), else None."""
    header = request.headers.get("if-none-match")
    if header is None:
        return None
    tags = {tag.strip().removeprefix("W/") for tag in header.split(",")}
    if "*" not in tags and etag.removeprefix("W/") not in tags:
        return None
    return Response(status_code=304, headers={"ETag": etag, "Cache-Control": _CACHE_CONTROL})


def _byte_range(request: Request, etag: str, size: int) -> tuple[int, int] | None:
    """The request's ``Range`` as (start, end exclusive), None for the whole file, 416 when it starts past the end."""
    # No, several, malformed or stale (`If-Range`) ranges get the whole file, a valid answer to any range request.
    header = request.headers.get("range")
    if header is None or request.headers.get("if-range", etag) != etag:
        return None
    match = _SINGLE_RANGE.fullmatch(header.strip())
    if match is None or match.groups() == ("", ""):
        return None
    first, last = match.groups()
    if first:
        start, end = int(first), min(size, int(last) + 1) if last else size
    else:
        start, end = max(0, size - int(last)), size
    if start >= size:
        raise HTTPException(status_code=416, detail="Range not satisfiable", headers={"Content-Range": f"bytes */{size}"})
    return (start, end) if start < end else None


@dataclass(frozen=True)
class _StoredFile:
    """A file in storage, known by its metadata: a revalidation or a range never reads all of its bytes."""

    etag: str
    size: int
    media_type: str

    def read(self) -> tuple[bytes, str | None]:
        """The whole content and the ETag of what was read, None when it changed since the metadata was read."""
        raise NotImplementedError

    def read_range(self, start: int, length: int) -> bytes:
        raise NotImplementedError

    def sha256(self) -> str:
        raise NotImplementedError

    def response(self, request: Request) -> Response:
        if (not_modified := _not_modified(request, self.etag)) is not None:
            return not_modified
        headers = {"Content-Disposition": "inline", "Accept-Ranges": "bytes", "Cache-Control": _CACHE_CONTROL}
        span = _byte_range(request, self.etag, self.size)
        with http_errors((StorageNotFoundError, 404, "File not found")):
            if span is None:
                content, etag = self.read()
                if etag is not None:
                    headers["ETag"] = etag
                return Response(content=content, media_type=self.media_type, headers=headers)
            start, end = span
            content = self.read_range(start, end - start)
        headers |= {"ETag": self.etag, "Content-Range": f"bytes {start}-{start + len(content) - 1}/{self.size}"}
        return Response(content=content, status_code=206, media_type=self.media_type, headers=headers)


@dataclass(frozen=True)
class _StoredDocument(_StoredFile):
    """A file in the document store."""

    docs: DataStore
    key: str
    entry: Entry

    @classmethod
    def of(cls, docs: DataStore, key: str, entry: Entry) -> "_StoredDocument":
        """*entry* is a file's `stat`, which sets its size, revision and last write."""
        # Version and last write together: a deleted and re-created key restarts at version 1.
        etag = f'"{entry.revision.token}-{int(entry.modified.timestamp() * 1_000_000)}"'
        return cls(etag, entry.size, mime_for(key), docs, key, entry)

    def read(self) -> tuple[bytes, str | None]:
        content, revision = self.docs.read_bytes(self.key)
        return content, self.etag if revision == self.entry.revision else None

    def read_range(self, start: int, length: int) -> bytes:
        return self.docs.read_range(self.key, start, length)

    def sha256(self) -> str:
        with http_errors((StorageNotFoundError, 404, "File not found")):
            return self.docs.sha256(self.key)


@dataclass(frozen=True)
class _StoredAsset(_StoredFile):
    """A binary asset (PDFs, MinerU output), its ETag is the content hash the asset store keeps."""

    assets: AssetStore
    asset: Asset

    @classmethod
    def of(cls, assets: AssetStore, asset: Asset, media_type: str) -> "_StoredAsset":
        return cls(f'"{asset.sha256}"', asset.size_bytes, media_type, assets, asset)

    def read(self) -> tuple[bytes, str | None]:
        asset = self.assets.read(self.asset.logical_path)
        return asset.content or b"", f'"{asset.sha256}"'

    def read_range(self, start: int, length: int) -> bytes:
        return self.assets.read_range(self.asset.logical_path, start, length)

    def sha256(self) -> str:
        return self.asset.sha256


@dataclass(frozen=True)
class _HostFile:
    """An allowed host path that storage does not own."""

    path: Path

    def response(self, request: Request) -> Response:
        with http_errors((OSError, 404, "File not found")):
            stat = self.path.stat()
        # FileResponse serves ranges and sets ETag and Last-Modified, but never answers a matching ETag with 304.
        response = FileResponse(
            self.path,
            media_type=mime_for(self.path),
            content_disposition_type="inline",
            stat_result=stat,
            headers={"Cache-Control": _CACHE_CONTROL},
        )
        return _not_modified(request, response.headers["etag"]) or response

    def sha256(self) -> str:
        with http_errors((OSError, 404, "File not found")), self.path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()


def _raw_source(path: str, roots: FileRoots, docs: DataStore, assets: AssetStore) -> _StoredFile | _HostFile:
    """The stored file at *path* (documents first, then assets), or the allowed host path when storage does not own it."""
    key = lib_docs.storage_key(path)
    if key is not None:
        entry = docs.stat(key)
        if entry is not None:
            if entry.is_dir:
                raise HTTPException(status_code=404, detail="File not found")
            return _StoredDocument.of(docs, key, entry)
        # Binary assets live in the `assets` table under the same key `document_meta` returns.
        asset = assets.stat(key)
        if asset is not None:
            return _StoredAsset.of(assets, asset, mime_for(path))
    return _HostFile(_resolve_allowed(path, roots))


@router.get("/raw")
def read_raw(
    request: Request,
    path: str = Query(...),
    roots: FileRoots = Depends(get_file_roots),
    docs: DataStore = Depends(get_document_store),
    assets: AssetStore = Depends(get_asset_store),
) -> Response:
    return _raw_source(path, roots, docs, assets).response(request)


@router.post("/annotations")
def open_pdf_annotations(
    path: str = Query(...),
    roots: FileRoots = Depends(get_file_roots),
    docs: DataStore = Depends(get_document_store),
    assets: AssetStore = Depends(get_asset_store),
) -> dict[str, str]:
    """Return the annotation canvas key of the PDF at *path*, creating the empty canvas on first use.

    - Keyed by the SHA-256 of the PDF bytes: every copy shares annotations, renames keep them
    - Changed PDF content gets a new, empty canvas
    - A host file is recorded in the canvas's meta, so delete cleanup keeps the canvas while that file still has the hash
    """
    # Stored PDFs are not read here: assets and documents both keep the hash of their content.
    source = _raw_source(path, roots, docs, assets)
    host = str(source.path) if isinstance(source, _HostFile) else None
    return {"key": annotation_canvas(docs, source.sha256(), host)}
