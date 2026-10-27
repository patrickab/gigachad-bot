"""Tests for ``lib.extract_queue`` — the background MinerU extraction worker.

Extraction is decoupled from any pre-existing disk file: ``enqueue`` takes
the PDF's bytes directly and the worker spools its own temp file. A finished
parse is persisted into the requesting user's Postgres library cache; the
on-disk Nextcloud mirror is a write-only convenience copy, never read back.
"""

from pathlib import Path

import pytest

import lib.extract_queue as extract_queue
from lib.asset_store import AssetStore
from lib.attachment_materialize import library_markdown_key
from tests.conftest import create_user_and_device

pytestmark = pytest.mark.usefixtures("clean_database", "documents_root")


@pytest.fixture
def assets(postgres_pool):
    user_id, device_id = create_user_and_device(postgres_pool)
    return AssetStore(postgres_pool, user_id, device_id=device_id)


async def test_worker_persists_a_finished_parse_into_postgres(tmp_path, assets, monkeypatch):
    """The worker spools its own temp file — no disk file needs to exist beforehand."""

    async def fake_parse_pdf(pdf_path: Path, output_dir: Path, backend: str = "pipeline"):
        assert pdf_path.read_bytes() == b"%PDF body"
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / f"{pdf_path.stem}.md"
        md_path.write_text("# extracted", encoding="utf-8")
        (images_dir / f"{pdf_path.stem}-1.png").write_bytes(b"\x89PNG")
        return md_path, images_dir

    monkeypatch.setattr(extract_queue.mineru, "parse_pdf", fake_parse_pdf)

    extract_queue.enqueue("report.pdf", b"%PDF body", assets)
    await extract_queue.start()
    await extract_queue.stop()

    assert assets.read(library_markdown_key("report")).content == b"# extracted"
    assert [a.logical_path for a in assets.list("Mineru/images/report")] == ["Mineru/images/report/report-1.png"]


async def test_worker_survives_a_failed_parse(assets, monkeypatch):
    async def failing_parse_pdf(pdf_path: Path, output_dir: Path, backend: str = "pipeline"):
        raise RuntimeError("MinerU exploded")

    monkeypatch.setattr(extract_queue.mineru, "parse_pdf", failing_parse_pdf)

    extract_queue.enqueue("broken.pdf", b"%PDF body", assets)
    await extract_queue.start()
    await extract_queue.stop()

    assert assets.list("Mineru") == []
