"""Tests for ``lib.attachment_materialize`` — the shared attach/parse core.

A PDF's MinerU markdown is cached once per user in Postgres
(``Mineru/<stem>.md``), not on disk: ``materialize`` must read that row, and
only ever touch the filesystem to write a Nextcloud mirror (a convenience
copy, never read back).
"""

import pytest

from lib.attachment_materialize import materialize, store_library_output, store_library_pdf
from lib.storage_namespace import library_markdown

pytestmark = pytest.mark.usefixtures("clean_database", "documents_root")


def test_store_library_pdf_writes_the_asset_and_mirrors_it(asset_store, documents_root):
    asset = store_library_pdf(asset_store, "paper.pdf", b"%PDF-1.7 body")

    assert (asset.kind, asset.logical_path) == ("pdf", "PDFs/paper.pdf")
    assert asset_store.read("PDFs/paper.pdf").content == b"%PDF-1.7 body"
    assert (documents_root / "PDFs/paper.pdf").read_bytes() == b"%PDF-1.7 body"


def test_store_library_output_writes_markdown_and_matching_images(tmp_path, asset_store, documents_root):
    md_path = tmp_path / "paper.md"
    md_path.write_text("# extracted", encoding="utf-8")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "paper-1.png").write_bytes(b"\x89PNG image one")
    (images_dir / "paper-2.png").write_bytes(b"\x89PNG image two")
    (images_dir / "other-1.png").write_bytes(b"\x89PNG unrelated stem")

    store_library_output(asset_store, "paper", md_path, images_dir)

    assert asset_store.read(library_markdown("paper")).content == b"# extracted"
    image_paths = sorted(a.logical_path for a in asset_store.list("Mineru/images/paper"))
    assert image_paths == ["Mineru/images/paper/paper-1.png", "Mineru/images/paper/paper-2.png"]
    assert (documents_root / "Mineru/paper.md").read_text(encoding="utf-8") == "# extracted"
    assert (documents_root / "Mineru/images/paper/paper-1.png").read_bytes() == b"\x89PNG image one"


def test_materialize_cache_hit_returns_the_stored_markdown(asset_store, no_enqueue):
    asset_store.write("mineru_markdown", library_markdown("paper"), b"# cached", mime="text/markdown")

    result = materialize("paper.pdf", b"%PDF-1.7 body", asset_store)

    assert result == "# cached"
    assert no_enqueue == []


def test_materialize_cache_miss_enqueues_by_default(asset_store, no_enqueue):
    result = materialize("unseen.pdf", b"%PDF-1.7 body", asset_store)

    assert result is None
    assert no_enqueue == ["unseen.pdf"]


def test_materialize_cache_miss_skips_enqueue_when_disabled(asset_store, no_enqueue):
    """``enqueue_on_miss=False`` is the path ``files.parse_attachments`` uses to parse synchronously."""
    result = materialize("sync.pdf", b"%PDF-1.7 body", asset_store, enqueue_on_miss=False)

    assert result is None
    assert no_enqueue == []


def test_materialize_strips_only_the_pdf_suffix(asset_store, no_enqueue):
    """A dotted name like ``paper.v1.pdf`` must not be truncated to ``paper`` under ``Path.stem`` semantics."""
    asset_store.write("mineru_markdown", library_markdown("paper.v1"), b"# v1", mime="text/markdown")

    assert materialize("paper.v1.pdf", b"body", asset_store) == "# v1"
