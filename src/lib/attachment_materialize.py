"""Shared attach/parse core for the documents/files/file_roots routes.

A PDF and its MinerU output (markdown + images) are written once per user into
the shared library namespace (``PDFs/<name>``, ``Mineru/<stem>.md``,
``Mineru/images/<stem>/...``) and mirrored to Nextcloud as a copy the app never
reads back. This module is the one place that writes those namespaces.
"""

import logging
from pathlib import Path

from config import DOCUMENTS
from lib import extract_queue
from lib.asset_store import Asset, AssetStore
from lib.data_store import StorageNotFoundError
from lib.storage_namespace import MINERU, PDFS

log = logging.getLogger(__name__)


def library_markdown_key(stem: str) -> str:
    return f"{MINERU}/{stem}.md"


def library_images_prefix(stem: str) -> str:
    return f"{MINERU}/images/{stem}"


def _mirror_quietly(assets: AssetStore, asset: Asset) -> None:
    """Mirror to Nextcloud, logging failure instead of raising.

    The database row is authoritative, so a failed mirror only loses the convenience copy.
    """
    try:
        assets.mirror(asset, DOCUMENTS)
    except OSError:
        log.exception("Failed to mirror %s into Nextcloud", asset.logical_path)


def store_library_pdf(assets: AssetStore, name: str, content: bytes) -> Asset:
    """Store (or refresh) a PDF in the shared per-user library and mirror it.

    The filename is the document's identity, so re-storing a name overwrites it."""
    asset = assets.write("pdf", f"{PDFS}/{name}", content, mime="application/pdf")
    _mirror_quietly(assets, asset)
    return asset


def _library_pdf(assets: AssetStore, name: str) -> Asset | None:
    """The library copy of the PDF *name*, or None when it is not stored."""
    try:
        return assets.read(f"{PDFS}/{name}")
    except StorageNotFoundError:
        return None


def sync_library_pdf(assets: AssetStore, name: str, content: bytes) -> None:
    """Make the library copy of the PDF *name* equal *content*, dropping its cached extraction when it changed, so the next `materialize` re-extracts."""
    stored = _library_pdf(assets, name)
    if stored is not None and stored.content == content:
        return
    store_library_pdf(assets, name, content)
    images = assets.list(f"{library_images_prefix(_stem(name))}/")
    for key in (library_markdown_key(_stem(name)), *(image.logical_path for image in images)):
        assets.delete(key)


def is_superseded(assets: AssetStore, name: str, content: bytes) -> bool:
    """True when the library copy of the PDF *name* now holds other bytes than *content*, making an extraction of *content* stale."""
    stored = _library_pdf(assets, name)
    return stored is not None and stored.content != content


def store_library_output(assets: AssetStore, stem: str, md_path: Path, images_dir: Path) -> None:
    """Persist a completed MinerU parse into this user's shared library cache."""
    md_asset = assets.write("mineru_markdown", library_markdown_key(stem), md_path.read_bytes(), mime="text/markdown")
    _mirror_quietly(assets, md_asset)
    if images_dir.is_dir():
        for image in sorted(images_dir.iterdir()):
            # The images directory may hold other documents' images, so keep only this stem's.
            if image.is_file() and image.name.startswith(stem):
                key = f"{library_images_prefix(stem)}/{image.name}"
                _mirror_quietly(assets, assets.write("mineru_image", key, image.read_bytes()))


def _stem(name: str) -> str:
    return name[:-4] if name.lower().endswith(".pdf") else name


def materialize(name: str, content: bytes, assets: AssetStore, *, enqueue_on_miss: bool = True) -> str | None:
    """This user's cached MinerU markdown for the PDF *name*, or None on a cache miss.

    A miss queues background extraction unless *enqueue_on_miss* is False
    (for callers that parse synchronously). *content* is all extraction needs.
    """
    try:
        return (assets.read(library_markdown_key(_stem(name))).content or b"").decode("utf-8")
    except StorageNotFoundError:
        if enqueue_on_miss:
            extract_queue.enqueue(name, content, assets)
        return None
