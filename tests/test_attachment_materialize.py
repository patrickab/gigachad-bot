"""Tests for ``lib.attachment_materialize`` — the shared attach/parse core.

A PDF's MinerU markdown is cached once per user in Postgres
(``Mineru/<stem>.md``), not on disk: ``materialize`` must read that row, and
only ever touch the filesystem to write a Nextcloud mirror (a convenience
copy, never read back).
"""

import os
from uuid import uuid4

from psycopg_pool import ConnectionPool
import pytest

from lib.asset_store import AssetStore
from lib.attachment_materialize import library_markdown_key, materialize, store_library_output, store_library_pdf
from lib.db_schema import upgrade


@pytest.fixture(scope="module")
def postgres_pool():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required")
    upgrade(url)
    pool = ConnectionPool(url, min_size=1, max_size=4)
    yield pool
    pool.close()


@pytest.fixture(autouse=True)
def clean_database(postgres_pool):
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute("TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE")


@pytest.fixture
def assets(postgres_pool):
    device_id = uuid4()
    with postgres_pool.connection() as connection, connection.transaction():
        user_id = connection.execute(
            "INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", ("alice@example.test",)
        ).fetchone()[0]
        connection.execute("INSERT INTO devices (id, user_id) VALUES (%s, %s)", (device_id, user_id))
    return AssetStore(postgres_pool, user_id, device_id=device_id)


@pytest.fixture(autouse=True)
def documents_root(tmp_path, monkeypatch):
    """Point the Nextcloud mirror root at a tmp dir; the mirror is a
    convenience copy this module writes but never reads back."""
    import lib.attachment_materialize as attachment_materialize

    root = tmp_path / "Documents"
    monkeypatch.setattr(attachment_materialize, "DOCUMENTS", root, raising=True)
    return root


def test_store_library_pdf_writes_the_asset_and_mirrors_it(assets, documents_root):
    asset = store_library_pdf(assets, "paper.pdf", b"%PDF-1.7 body")

    assert (asset.kind, asset.logical_path) == ("pdf", "PDFs/paper.pdf")
    assert assets.read("PDFs/paper.pdf").content == b"%PDF-1.7 body"
    assert (documents_root / "PDFs/paper.pdf").read_bytes() == b"%PDF-1.7 body"


def test_store_library_output_writes_markdown_and_matching_images(tmp_path, assets, documents_root):
    md_path = tmp_path / "paper.md"
    md_path.write_text("# extracted", encoding="utf-8")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "paper-1.png").write_bytes(b"\x89PNG image one")
    (images_dir / "paper-2.png").write_bytes(b"\x89PNG image two")
    (images_dir / "other-1.png").write_bytes(b"\x89PNG unrelated stem")

    store_library_output(assets, "paper", md_path, images_dir)

    assert assets.read(library_markdown_key("paper")).content == b"# extracted"
    image_paths = sorted(a.logical_path for a in assets.list("Mineru/images/paper"))
    assert image_paths == ["Mineru/images/paper/paper-1.png", "Mineru/images/paper/paper-2.png"]
    assert (documents_root / "Mineru/paper.md").read_text(encoding="utf-8") == "# extracted"
    assert (documents_root / "Mineru/images/paper/paper-1.png").read_bytes() == b"\x89PNG image one"


def test_materialize_cache_hit_returns_the_stored_markdown(assets, no_enqueue):
    assets.write("mineru_markdown", library_markdown_key("paper"), b"# cached", mime="text/markdown")

    result = materialize("paper.pdf", b"%PDF-1.7 body", assets)

    assert result == "# cached"
    assert no_enqueue == []


def test_materialize_cache_miss_enqueues_by_default(assets, no_enqueue):
    result = materialize("unseen.pdf", b"%PDF-1.7 body", assets)

    assert result is None
    assert no_enqueue == ["unseen.pdf"]


def test_materialize_cache_miss_skips_enqueue_when_disabled(assets, no_enqueue):
    """``enqueue_on_miss=False`` is the path ``files.parse_attachments`` uses to parse synchronously."""
    result = materialize("sync.pdf", b"%PDF-1.7 body", assets, enqueue_on_miss=False)

    assert result is None
    assert no_enqueue == []


def test_materialize_strips_only_the_pdf_suffix(assets, no_enqueue):
    """A dotted name like ``paper.v1.pdf`` must not be truncated to ``paper`` under ``Path.stem`` semantics."""
    assets.write("mineru_markdown", library_markdown_key("paper.v1"), b"# v1", mime="text/markdown")

    assert materialize("paper.v1.pdf", b"body", assets) == "# v1"
