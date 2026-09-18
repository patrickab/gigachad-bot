"""Tests for ``lib.extract_queue`` — the background MinerU extraction worker.

Extraction is decoupled from any pre-existing disk file: ``enqueue`` takes
the PDF's bytes directly and the worker spools its own temp file. A finished
parse is persisted into the requesting user's Postgres library cache; the
on-disk Nextcloud mirror is a write-only convenience copy, never read back.
"""

import os
from pathlib import Path
from uuid import uuid4

from psycopg_pool import ConnectionPool
import pytest

import lib.extract_queue as extract_queue
from lib.asset_store import AssetStore
from lib.attachment_materialize import library_markdown_key
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
    import lib.attachment_materialize as attachment_materialize

    root = tmp_path / "Documents"
    monkeypatch.setattr(attachment_materialize, "DOCUMENTS", root, raising=True)
    return root


def test_enqueue_puts_name_content_and_store_on_the_queue(assets, monkeypatch):
    put_calls: list[tuple] = []
    monkeypatch.setattr(extract_queue._queue, "put_nowait", lambda item: put_calls.append(item))

    extract_queue.enqueue("report.pdf", b"%PDF body", assets)

    assert put_calls == [("report.pdf", b"%PDF body", assets)]


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
    assert extract_queue.status() == {"in_progress": None, "queued": 0}
