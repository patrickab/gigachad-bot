"""Chat attachments: uploads with their parsed-markdown sidecars, and mounted files attached as live references.

PDFs also go into the shared library (``attachment_materialize``), which caches their MinerU markdown.
"""

import logging
from pathlib import Path, PurePosixPath
import tempfile
from typing import Literal

from pydantic import BaseModel

from lib.asset_store import AssetStore
from lib.attachment_materialize import cached_markdown, materialize, store_library_output, store_library_pdf, sync_library_pdf
from lib.data_store import StorageNotFoundError, validate_key
from lib.mime import mime_for
from lib.mineru import parse_pdf, should_cancel
from lib.storage_namespace import chat_upload_prefix, library_pdf

log = logging.getLogger(__name__)

AttachmentKind = Literal["pdf", "image", "text", "other"]

_TEXT_EXTS = {".csv", ".json", ".yaml", ".yml", ".xml", ".toml", ".ini", ".cfg", ".conf", ".log", ".md", ".rst", ".svg"}


class AttachResult(BaseModel):
    name: str
    mime: str
    # Set by file-root attaches only, stored on the Attachment and used to re-read content
    path: str | None = None
    content: str | None = None
    parsedMd: str | None = None


def decode_text(content: bytes) -> str | None:
    """The UTF-8 text of *content*, `None` when it is not valid UTF-8."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return None


def read_text_or_none(path: Path) -> str | None:
    """The UTF-8 text of *path* (universal newlines), `None` when it is not valid UTF-8."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def kind_of(name: str, mime: str = "") -> AttachmentKind:
    """What an attachment is, by *mime* when given, else by the suffix of *name* alone."""
    suffix = Path(name).suffix.lower()
    if mime == "application/pdf" or (not mime and suffix == ".pdf"):
        return "pdf"
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("text/") or suffix in _TEXT_EXTS:
        return "text"
    return "other"


def sync_host_pdf(assets: AssetStore, file: Path) -> str | None:
    """Sync a mounted PDF into the shared library and queue its extraction, returning the parsed markdown when already cached."""
    content = file.read_bytes()
    sync_library_pdf(assets, file.name, content)
    return materialize(file.name, content, assets, enqueue_on_miss=True)


def attach_host_file(assets: AssetStore, file: Path) -> AttachResult:
    """A mounted *file* as a live reference: a PDF with its cached markdown, anything else with its UTF-8 text."""
    result = AttachResult(name=file.name, mime=mime_for(file), path=str(file))
    if kind_of(file.name) == "pdf":
        result.parsedMd = sync_host_pdf(assets, file)
    else:
        result.content = read_text_or_none(file)
    return result


def attach_library_pdf(assets: AssetStore, path: str) -> AttachResult:
    """A library PDF key (`PDFs/<name>`, what older chats stored) with its cached markdown. Raises StorageNotFoundError."""
    # The PDF's bytes are read only to queue an extraction that is not cached yet.
    name = Path(path).name
    key = library_pdf(name)
    if assets.stat(key) is None:
        raise StorageNotFoundError(key)
    parsed_md = cached_markdown(assets, name)
    if parsed_md is None:
        parsed_md = materialize(name, assets.read(key).content or b"", assets, enqueue_on_miss=True)
    return AttachResult(name=name, mime=mime_for(path), path=key, parsedMd=parsed_md)


def _sidecar_key(key: str) -> str:
    """Parsed-markdown sibling of an attachment: ``a/foo.pdf`` -> ``a/foo.md``, as on disk."""
    path = PurePosixPath(key)
    return str(path.with_name(f"{path.stem}.md"))


class ChatUploads:
    """The files uploaded into one chat. A filename that would escape the chat's prefix raises InvalidStorageKey."""

    def __init__(self, assets: AssetStore, chat_id: str, slug: str | None = None) -> None:
        self._assets = assets
        self._prefix = chat_upload_prefix(chat_id, slug)

    def _key(self, filename: str) -> str:
        return validate_key(f"{self._prefix}/{filename}")

    def _free_name(self, filename: str) -> str:
        """*filename*, or the first free ``stem (N).suffix`` when an upload already has it."""
        taken = {PurePosixPath(asset.logical_path).name for asset in self._assets.list(self._prefix)}
        if filename not in taken:
            return filename
        stem, suffix = Path(filename).stem, Path(filename).suffix
        n = 1
        while f"{stem} ({n}){suffix}" in taken:
            n += 1
        return f"{stem} ({n}){suffix}"

    def store(self, filename: str, content: bytes, mime: str, *, overwrite: bool) -> AttachResult:
        """Store an upload under *filename* (deduplicated unless *overwrite*), returning text content inline.

        A PDF is mirrored into the library at once: it must reach Nextcloud whether or not
        the user ever sends or parses it, and MinerU needs a real file to read.
        """
        name = filename if overwrite else self._free_name(filename)
        self._assets.write("upload", self._key(name), content, mime=mime)
        result = AttachResult(name=name, mime=mime)
        kind = kind_of(name, mime)
        if kind == "text":
            result.content = decode_text(content)
        elif kind == "pdf":
            store_library_pdf(self._assets, name, content)
        return result

    async def parse(self, filenames: list[str]) -> list[AttachResult]:
        """Parse uploads to markdown, stopping the batch when MinerU is cancelled.

        The PDF library row is already written by `store`, MinerU only needs a real file in a temp dir.
        """
        results: list[AttachResult] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            for filename in filenames:
                if should_cancel():
                    log.info("Parse cancelled, stopping batch")
                    break
                results.append(await self._parse_one(filename, work))
        return results

    async def _parse_one(self, filename: str, work: Path) -> AttachResult:
        """One upload with its markdown: a PDF's (cached or freshly extracted, stored as sidecar), a text file's content."""
        mime = mime_for(filename)
        key = self._key(filename)
        try:
            asset = self._assets.read(key)
        except StorageNotFoundError:
            return AttachResult(name=filename, mime=mime)
        content = asset.content or b""
        kind = kind_of(filename)
        if kind != "pdf":
            return AttachResult(name=filename, mime=mime, parsedMd=decode_text(content) if kind == "text" else None)

        parsed_md = materialize(filename, content, self._assets, enqueue_on_miss=False)
        if parsed_md is None:
            pdf_path = work / Path(filename).name
            pdf_path.write_bytes(content)
            try:
                md_path, images_dir = await parse_pdf(pdf_path, work / "mineru")
            except Exception:
                log.exception("MinerU parse failed for %s", filename)
                return AttachResult(name=filename, mime=mime)
            store_library_output(self._assets, pdf_path.stem, md_path, images_dir)
            parsed_md = md_path.read_text(encoding="utf-8")
        self._assets.write("upload", _sidecar_key(key), parsed_md.encode("utf-8"), mime="text/markdown")
        return AttachResult(name=filename, mime=mime, parsedMd=parsed_md)

    def delete(self, filename: str) -> bool:
        """Delete an upload and its markdown sidecar, False when there is no such upload."""
        key = self._key(filename)
        if not any(asset.logical_path == key for asset in self._assets.list(key)):
            return False
        self._assets.delete(key)
        self._assets.delete(_sidecar_key(key))
        return True

    def delete_all(self) -> None:
        """Delete every upload of the chat."""
        for asset in self._assets.list(self._prefix):
            self._assets.delete(asset.logical_path)
