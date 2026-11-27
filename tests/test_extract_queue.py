"""Tests for ``lib.extract_queue`` — the background MinerU extraction worker.

Extraction is decoupled from any pre-existing disk file: ``enqueue`` takes
the PDF's bytes directly and the worker spools its own temp file. A finished
parse is persisted into the requesting user's Postgres library cache; the
on-disk Nextcloud mirror is a write-only convenience copy, never read back.
"""

import asyncio
from pathlib import Path
import threading
import time

import pytest

from lib.attachment_materialize import materialize
import lib.extract_queue as extract_queue
from lib.storage_namespace import library_markdown

pytestmark = pytest.mark.usefixtures("clean_database", "documents_root")


async def test_worker_persists_a_finished_parse_into_postgres(tmp_path, asset_store, monkeypatch):
    """The worker spools its own temp file — no disk file needs to exist beforehand."""

    async def fake_parse_pdf(pdf_path: Path, output_dir: Path):
        assert pdf_path.read_bytes() == b"%PDF body"
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / f"{pdf_path.stem}.md"
        md_path.write_text("# extracted", encoding="utf-8")
        (images_dir / f"{pdf_path.stem}-1.png").write_bytes(b"\x89PNG")
        return md_path, images_dir

    monkeypatch.setattr(extract_queue.mineru, "parse_pdf", fake_parse_pdf)

    assert materialize("report.pdf", b"%PDF body", asset_store) is None
    await extract_queue.start()
    await extract_queue.stop()

    assert asset_store.read(library_markdown("report")).content == b"# extracted"
    assert [a.logical_path for a in asset_store.list("Mineru/images/report")] == ["Mineru/images/report/report-1.png"]


async def test_worker_survives_a_failed_parse(asset_store, monkeypatch):
    async def failing_parse_pdf(pdf_path: Path, output_dir: Path):
        raise RuntimeError("MinerU exploded")

    monkeypatch.setattr(extract_queue.mineru, "parse_pdf", failing_parse_pdf)

    materialize("broken.pdf", b"%PDF body", asset_store)
    await extract_queue.start()
    await extract_queue.stop()

    assert asset_store.list("Mineru") == []


async def test_enqueue_from_a_thread_wakes_the_idle_worker(monkeypatch):
    """Sync routes enqueue from threadpool threads; the idle worker must start without other loop activity."""
    parsed = threading.Event()

    async def fake_parse_pdf(pdf_path: Path, output_dir: Path):
        parsed.set()
        raise RuntimeError("stop here")

    def enqueue_and_wait() -> bool:
        time.sleep(0.1)  # the loop is asleep in select
        extract_queue.enqueue("thread.pdf", b"%PDF thread", lambda *_: None)
        return parsed.wait(timeout=2)

    monkeypatch.setattr(extract_queue.mineru, "parse_pdf", fake_parse_pdf)
    await extract_queue.start()
    await asyncio.sleep(0)  # the worker reaches its idle wait on the queue
    try:
        # The loop idles until the thread returns: only the enqueue itself can wake the worker.
        assert await asyncio.get_running_loop().run_in_executor(None, enqueue_and_wait)
    finally:
        await extract_queue.stop()
