"""Shared attach/parse core for the documents/files/file_vaults routes.

A PDF and its MinerU output (markdown + images) are each written once per
user into the shared library namespace (``PDFs/<name>``, ``Mineru/<stem>.md``,
``Mineru/images/<stem>/...``) and mirrored to Nextcloud as a convenience copy
other tools can read — the app itself never reads that mirror back. Route-
specific behavior (vault promotion, chat-upload parsing, document attach)
stays in the routes; this module is the one place that writes those two
shared namespaces.
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
    """The database row is authoritative: a failed mirror is a missing
    convenience copy, logged and retried on the next write, never a lost document."""
    try:
        assets.mirror(asset, DOCUMENTS)
    except OSError:
        log.exception("Failed to mirror %s into Nextcloud", asset.logical_path)


def store_library_pdf(assets: AssetStore, name: str, content: bytes) -> Asset:
    """Store (or refresh) a PDF in the shared per-user library and mirror it
    to Nextcloud. Filename is the document's identity — re-storing the same
    name overwrites the existing library copy."""
    asset = assets.write("pdf", f"{PDFS}/{name}", content, mime="application/pdf")
    _mirror_quietly(assets, asset)
    return asset


def store_library_output(assets: AssetStore, stem: str, md_path: Path, images_dir: Path) -> None:
    """Persist a completed MinerU parse into this user's shared library cache."""
    md_asset = assets.write("mineru_markdown", library_markdown_key(stem), md_path.read_bytes(), mime="text/markdown")
    _mirror_quietly(assets, md_asset)
    if images_dir.is_dir():
        for image in sorted(images_dir.iterdir()):
            if image.is_file() and image.name.startswith(stem):
                key = f"{library_images_prefix(stem)}/{image.name}"
                _mirror_quietly(assets, assets.write("mineru_image", key, image.read_bytes()))


def _stem(name: str) -> str:
    return name[:-4] if name.lower().endswith(".pdf") else name


def materialize(name: str, content: bytes, assets: AssetStore, *, enqueue_on_miss: bool = True) -> str | None:
    """This user's cached MinerU markdown for the PDF *name*, or None on a cache miss.

    Queues background extraction on a miss unless *enqueue_on_miss* is False
    (pass False when the caller parses synchronously instead). Extraction
    never depends on a pre-existing disk file — *content* is all it needs.
    """
    try:
        return (assets.read(library_markdown_key(_stem(name))).content or b"").decode("utf-8")
    except StorageNotFoundError:
        if enqueue_on_miss:
            extract_queue.enqueue(name, content, assets)
        return None
