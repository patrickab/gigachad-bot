"""What stored items reference, and the cleanup every delete runs to drop derived data nothing references any more.

Holders of references, read from `documents.meta` so no body is scanned per delete:
- chats: each attachment's `vaultPath`, plus the library copy `PDFs/<name>` of a PDF attachment (`ChatStore`)
- projects: the `files` of their `project.json`
- canvases: frame and attachment paths, nested inline canvases included (written by `CanvasMutationStore`).
  Notes canvases always hold theirs, project canvases only while a chat, project or held canvas references them
- chat uploads: a PDF upload holds its library copy

Derived data, deleted once unused: library PDFs `PDFs/<name>`, PDF annotation canvases `pdf-annotation/<sha256>.canvas`
(their mutation log and share rows follow by foreign key) and project documents `project/<slug>/document/...`. MinerU
output is never deleted. An annotation canvas stays while any stored or referenced host PDF has its hash, or a host file
recorded in its meta `hosts` (opened with it from the file tree) still does. Host files are only hashed, never written.

`python -m lib.references --dry-run` lists, per user, what earlier deletes left behind.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from psycopg_pool import ConnectionPool

from lib.asset_store import Asset, AssetStore
from lib.changes import UserScope
from lib.chat_store import ChatStore
from lib.data_store import DataStore, InvalidStorageKey, Meta, validate_key
from lib.db_schema import database_url
from lib.json_docs import JsonDocs
from lib.postgres_data_store import PostgresDataStore
from lib.storage_namespace import (
    ATTACHMENT,
    CHAT,
    NOTE,
    PDF_ANNOTATION,
    PDFS,
    PROJECT,
    PROJECT_JSON,
    chat_upload_prefix,
    library_pdf,
    pdf_annotations,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

log = logging.getLogger(__name__)

_CANVAS = ".canvas"
# Current canvases keep `frames` and `attachments`, older ones `pages`, `imageEmbeds` and `pdfEmbeds`.
_CANVAS_COLLECTIONS = ("frames", "attachments", "pages", "imageEmbeds", "pdfEmbeds")
_PDF_MIME = "application/pdf"


def canvas_refs(document: object) -> list[str]:
    """Paths the frames and attachments of a parsed canvas point at, nested inline canvases included."""
    found: set[str] = set()
    pending = [document]
    while pending:
        current = pending.pop()
        if not isinstance(current, dict):
            continue
        for collection in _CANVAS_COLLECTIONS:
            items = current.get(collection)
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict):
                    continue
                if isinstance(path := item.get("path"), str) and path:
                    found.add(path)
                pending.append(item.get("canvas"))
    return sorted(found)


def canvas_meta(document: object) -> Meta:
    """The meta stored beside a canvas: its references."""
    return {"refs": canvas_refs(document)}


def _project_meta(project: object) -> Meta:
    """The meta stored beside a `project.json`: its files, plain paths or `{"path", "name"}` entries."""
    files = project.get("files") if isinstance(project, dict) else None
    paths = {entry.get("path") if isinstance(entry, dict) else entry for entry in files} if isinstance(files, list) else set()
    return {"refs": sorted(path for path in paths if isinstance(path, str) and path)}


def _is_canvas(key: str) -> bool:
    return key.endswith(_CANVAS)


def _is_project_json(key: str) -> bool:
    return PurePosixPath(key).name == PROJECT_JSON


def _is_pdf(name: str) -> bool:
    return name.lower().endswith(".pdf")


def _is_host(path: str) -> bool:
    return path.startswith("/")


def _under(key: str, prefix: str) -> bool:
    return key.startswith(f"{prefix}/")


def _name(key: str) -> str:
    return PurePosixPath(key).name


def _is_project_document(key: str) -> bool:
    parts = key.split("/")
    return len(parts) >= 4 and parts[0] == PROJECT and parts[2] == "document"


def _deletable(key: str) -> bool:
    """Whether *key* is derived data a release may delete once nothing uses it."""
    return _under(key, PDFS) or _under(key, PDF_ANNOTATION) or _is_project_document(key)


def annotation_canvas(docs: DataStore, sha256: str, host: str | None = None) -> str:
    """The annotation canvas key of the PDF with *sha256*, created empty on first use, recording the *host* file opened with it."""
    key = pdf_annotations(sha256)
    if not docs.exists(key):
        docs.write_bytes(key, b"", meta={"refs": [], "hosts": [host]} if host else None)
        return key
    if host is None:
        return key
    # Read the hash first: meta written against it is dropped when the canvas changed since.
    content_sha256 = docs.sha256(key)
    meta = JsonDocs(docs).list_meta(PDF_ANNOTATION, lambda found: found == key, canvas_meta, require="refs").get(key) or {}
    hosts = meta.get("hosts", [])
    if host not in hosts:
        docs.write_meta({key: ({**meta, "hosts": sorted({*hosts, host})}, content_sha256)})
    return key


@dataclass
class Released:
    """What a delete let go of, cleaned up by `References.release` once the delete committed.

    `owned` keys are the deleted item's own data and always go, `refs` only once nothing uses them,
    `chats` are deleted chats `(chat_id, project slug)` whose uploads go with them.
    """

    owned: set[str] = field(default_factory=set)
    refs: set[str] = field(default_factory=set)
    chats: list[tuple[str, str | None]] = field(default_factory=list)

    def chat(self, chat_id: str, slug: str | None, refs: Iterable[str]) -> None:
        """Record a deleted chat and the references its attachments held."""
        self.chats.append((chat_id, slug))
        self.refs.update(refs)


@dataclass
class _Usage:
    """What the stored items use right now."""

    reachable: set[str]
    pdf_names: set[str]
    hashes: Callable[[], set[str]]
    hosted: Callable[[str], bool]

    def uses(self, key: str) -> bool:
        if _under(key, PDFS):
            return _name(key) in self.pdf_names
        if _under(key, PDF_ANNOTATION):
            return _name(key).removesuffix(_CANVAS) in self.hashes() or self.hosted(key)
        return key in self.reachable


class References:
    """One user's references, and the release of derived data nothing uses any more."""

    def __init__(self, docs: DataStore, assets: AssetStore) -> None:
        self._docs = docs
        self._json = JsonDocs(docs)
        self._chats = ChatStore(data_store=docs)
        self._assets = assets
        self._host_hashes: dict[str, tuple[tuple[int, int], str | None]] = {}

    def release(self, released: Released) -> list[str]:
        """Delete the owned keys, the deleted chats' uploads, then each released key nothing uses any more, returning the deleted keys.

        Callers run it after their delete committed, so a failed delete never loses shared data. A failing owned
        delete raises before anything else goes. Later failures are logged, not raised: the item itself is gone,
        and the dry-run report lists what stayed behind.
        """
        gone: set[str] = set()
        pending = set(released.refs)
        for key in sorted(released.owned):
            pending |= self._derived(key)
            self._remove(key, gone, dry_run=False)
        try:
            for chat_id, slug in released.chats:
                for asset in self._assets.list(chat_upload_prefix(chat_id, slug)):
                    pending |= self._derived(asset.logical_path)
                    self._remove(asset.logical_path, gone, dry_run=False)
            self._sweep(pending, gone, dry_run=False)
        except Exception:
            log.exception("Releasing the data of a deleted item failed")
        return sorted(gone)

    def orphans(self) -> list[str]:
        """Derived keys nothing uses any more, as a release of every one of them would delete them. Nothing is deleted."""
        candidates = {asset.logical_path for asset in self._assets.list(PDFS)}
        candidates |= {entry.key for entry in self._docs.list(PDF_ANNOTATION, recursive=True) if not entry.is_dir}
        candidates |= {e.key for e in self._docs.list(PROJECT, recursive=True) if not e.is_dir and _is_project_document(e.key)}
        gone: set[str] = set()
        self._sweep(candidates, gone, dry_run=True)
        return sorted(gone)

    def orphan_uploads(self) -> list[Asset]:
        """Chat uploads whose chat is not stored, such as uploads of omp composers or of chats never saved."""
        chat_ids = {meta["chat_id"] for meta in self._chats.tree_metas().values() if meta}
        return [asset for asset in self._assets.list(ATTACHMENT) if _upload_chat_id(asset.logical_path) not in chat_ids]

    def size(self, key: str) -> int:
        """Stored bytes of *key*, 0 when it is gone."""
        if assets := self._assets.list(key):
            return sum(asset.size_bytes for asset in assets)
        entry = self._docs.stat(key)
        return (entry.size or 0) if entry is not None and not entry.is_dir else 0

    def _sweep(self, pending: set[str], gone: set[str], *, dry_run: bool) -> None:
        """Delete (or with *dry_run* only collect into *gone*) every pending key nothing uses, and what that releases in turn."""
        expanded: set[str] = set()
        # Each round sees the previous round's deletions: a deleted canvas releases what it held, a library PDF its extraction.
        while pending:
            usage = self._usage(gone)
            batch, pending = pending, set()
            for key in sorted(batch):
                if key in gone or usage.uses(key):
                    continue
                if _deletable(key):
                    pending |= self._derived(key)
                    self._remove(key, gone, dry_run=dry_run)
                elif key not in expanded:
                    # A released host path or note leaves what was derived from it.
                    expanded.add(key)
                    pending |= self._derived(key)

    def _usage(self, gone: set[str]) -> _Usage:
        """What chats, projects, reachable canvases and uploads use, ignoring the keys in *gone*."""
        roots: set[str] = set()
        for meta in self._chats.tree_metas().values():
            roots.update(meta["refs"] if meta else ())
        for meta in self._json.list_meta(CHAT, _is_project_json, _project_meta, require="refs").values():
            roots.update(meta["refs"] if meta else ())
        canvases: dict[str, list[str]] = {}
        for prefix in (NOTE, PROJECT):
            for key, meta in self._json.list_meta(prefix, _is_canvas, canvas_meta, require="refs").items():
                if key not in gone:
                    canvases[key] = meta["refs"] if meta else []
        roots.update(ref for key, refs in canvases.items() if _under(key, NOTE) for ref in refs)
        reachable: set[str] = set()
        queue = list(roots)
        while queue:
            ref = queue.pop()
            if ref not in reachable:
                reachable.add(ref)
                queue.extend(canvases.get(ref, ()))

        uploads = [a for a in self._assets.list(ATTACHMENT, kind="upload") if a.logical_path not in gone]
        library = [a for a in self._assets.list(PDFS) if a.logical_path not in gone]
        pdf_names = {_name(ref) for ref in reachable if (_under(ref, PDFS) or _is_host(ref)) and _is_pdf(ref)}
        pdf_names |= {_name(a.logical_path) for a in uploads if _is_pdf(a.logical_path) or a.mime == _PDF_MIME}
        hashes: set[str] | None = None
        hosts: dict[str, list[str]] | None = None

        def pdf_hashes() -> set[str]:
            """Hashes of every remaining stored PDF and referenced host PDF, computed on first use."""
            nonlocal hashes
            if hashes is None:
                hashes = {a.sha256 for a in (*uploads, *library)}
                for prefix in (NOTE, PROJECT):
                    for entry in self._docs.list(prefix, recursive=True):
                        if not entry.is_dir and entry.key not in gone and _is_pdf(entry.key):
                            hashes.add(self._docs.sha256(entry.key))
                hashes |= {h for ref in reachable if _is_host(ref) and _is_pdf(ref) and (h := self._host_hash(ref))}
            return hashes

        def hosted(key: str) -> bool:
            """Whether a host file recorded as opened with annotation canvas *key* still has its hash."""
            nonlocal hosts
            if hosts is None:
                hosts = {k: meta.get("hosts", []) for k, meta in self._docs.list_meta(PDF_ANNOTATION).items() if meta}
            sha256 = _name(key).removesuffix(_CANVAS)
            return any(self._host_hash(path) == sha256 for path in hosts.get(key, ()))

        return _Usage(reachable, pdf_names, pdf_hashes, hosted)

    def _derived(self, key: str) -> set[str]:
        """Keys derived from *key*, read before it is deleted: a PDF's library copy and annotations, a canvas's references."""
        name = _name(key)
        if _is_host(key):
            return {library_pdf(name), *self._annotations(self._host_hash(key))} if _is_pdf(name) else set()
        try:
            key = validate_key(key)
        except InvalidStorageKey:
            return set()
        asset = self._assets.stat(key)
        if _under(key, PDFS):
            return self._annotations(asset.sha256 if asset else None)
        if asset is not None:
            pdf_upload = _under(key, ATTACHMENT) and (_is_pdf(name) or asset.mime == _PDF_MIME)
            return {library_pdf(name), *self._annotations(asset.sha256)} if pdf_upload else set()
        if not self._docs.is_file(key):
            return set()
        derived = set(canvas_refs(self._json.read(key))) if _is_canvas(key) else set()
        if _is_pdf(name):
            derived |= self._annotations(self._docs.sha256(key))
        return derived

    @staticmethod
    def _annotations(sha256: str | None) -> set[str]:
        return {pdf_annotations(sha256)} if sha256 else set()

    def _host_hash(self, path: str) -> str | None:
        """SHA-256 of the host file at *path*, None when it is missing or unreadable, hashed again only after its stat changed."""
        try:
            stat = Path(path).stat()
        except OSError:
            return None
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._host_hashes.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]
        try:
            with Path(path).open("rb") as file:
                digest: str | None = hashlib.file_digest(file, "sha256").hexdigest()
        except OSError:
            digest = None
        self._host_hashes[path] = (signature, digest)
        return digest

    def _remove(self, key: str, gone: set[str], *, dry_run: bool) -> None:
        """Delete the stored data of *key* and add it to *gone*."""
        try:
            key = validate_key(key)
        except InvalidStorageKey:
            return
        assets = [asset.logical_path for asset in self._assets.list(key) if asset.logical_path == key]
        document = not assets and self._docs.is_file(key)
        for stored in assets:
            if not dry_run:
                self._assets.delete(stored)
            gone.add(stored)
        if document:
            if not dry_run:
                self._docs.delete(key)
            gone.add(key)


def _upload_chat_id(key: str) -> str | None:
    """The chat ID of an upload `attachment/chat/<id>/...` or `attachment/project/<slug>/chat/<id>/...`."""
    parts = key.split("/")
    if len(parts) > 3 and parts[1] == "chat":
        return parts[2]
    if len(parts) > 5 and parts[1] == "project" and parts[3] == "chat":
        return parts[4]
    return None


def _megabytes(size: int) -> str:
    return f"{size / 1_000_000:.2f} MB"


def main() -> None:
    """Print, per user, the derived data and uploads no stored item uses any more, with their sizes."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--database-url")
    args = parser.parse_args()
    with ConnectionPool(database_url(args.database_url), min_size=1, max_size=2) as pool:
        with pool.connection() as connection:
            users = connection.execute("SELECT id, tailscale_login FROM users ORDER BY tailscale_login").fetchall()
        for user_id, login in users:
            scope = UserScope(pool, user_id)
            references = References(PostgresDataStore(scope), AssetStore(scope))
            rows = [(key, references.size(key)) for key in references.orphans()]
            rows += [(asset.logical_path, asset.size_bytes) for asset in references.orphan_uploads()]
            print(f"{login}: {len(rows)} unused, {_megabytes(sum(size for _, size in rows))}")
            for key, size in rows:
                print(f"  {_megabytes(size):>12}  {key}")


if __name__ == "__main__":
    main()
