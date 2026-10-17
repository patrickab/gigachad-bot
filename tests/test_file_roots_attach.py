"""Route-level regression tests for ``file_roots`` file root attach.

A file root PDF is promoted into the shared Postgres library (``PDFs/<name>``,
via ``store_library_pdf``) on first attach, never onto disk only. The library
copy takes precedence over same-named file-root files thereafter, and the
endpoint hands back that library's logical key — never a raw filesystem path
— so the frontend's "re-attach at send time" refresh keeps working.
"""

import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg_pool import ConnectionPool
import pytest

from backend.routes import file_roots
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


@pytest.fixture(autouse=True)
def no_real_extraction(monkeypatch):
    """These tests assert on promotion/dedup, not extraction itself."""
    calls: list[str] = []
    monkeypatch.setattr(
        "lib.attachment_materialize.extract_queue.enqueue",
        lambda name, content, assets: calls.append(name),
    )
    return calls


class FakeFileRoots:
    """Minimal FileRoots stub: resolve() maps a virtual path to a tmp file."""

    def __init__(self, mapping: dict[str, Path]):
        self._mapping = mapping

    def resolve(self, path: str) -> Path:
        if path not in self._mapping:
            raise FileNotFoundError(path)
        return self._mapping[path]

    def read(self, path: str) -> str:
        return self._mapping[path].read_text(encoding="utf-8")


def _build_app(roots: FakeFileRoots, assets: AssetStore) -> FastAPI:
    app = FastAPI()
    app.include_router(file_roots.router)
    app.dependency_overrides[file_roots.get_file_roots] = lambda: roots
    app.dependency_overrides[file_roots.get_asset_store] = lambda: assets
    return app


def test_attach_root_pdf_cache_hit_returns_cached_markdown(tmp_path, assets, no_real_extraction):
    pdf = tmp_path / "vault.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    assets.write("mineru_markdown", library_markdown_key("vault"), b"# already extracted", mime="text/markdown")

    roots = FakeFileRoots({str(pdf): pdf})
    client = TestClient(_build_app(roots, assets))

    r = client.post("/api/fileroots/attach", params={"path": str(pdf)})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["parsedMd"] == "# already extracted"
    assert body["name"] == "vault.pdf"
    assert body["path"] == "PDFs/vault.pdf"
    # A first attach still promotes the PDF into the library, even on a markdown cache hit.
    assert assets.read("PDFs/vault.pdf").content == b"%PDF-1.4 fake"
    assert no_real_extraction == []


def test_attach_root_pdf_cache_miss_promotes_and_enqueues(tmp_path, assets, no_real_extraction):
    pdf = tmp_path / "report.v1.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    roots = FakeFileRoots({str(pdf): pdf})
    client = TestClient(_build_app(roots, assets))

    r = client.post("/api/fileroots/attach", params={"path": str(pdf)})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["parsedMd"] is None
    assert body["path"] == "PDFs/report.v1.pdf"
    assert assets.read("PDFs/report.v1.pdf").content == b"%PDF-1.4 fake"
    assert no_real_extraction == ["report.v1.pdf"]


def test_attach_root_pdf_library_copy_takes_precedence(tmp_path, assets, no_real_extraction):
    """A same-named library PDF (already parsed) is never clobbered by a same-named file-root file."""
    pdf = tmp_path / "vault.pdf"
    pdf.write_bytes(b"file root bytes, never promoted")
    assets.write("pdf", "PDFs/vault.pdf", b"library bytes", mime="application/pdf")
    assets.write("mineru_markdown", library_markdown_key("vault"), b"# library markdown", mime="text/markdown")

    roots = FakeFileRoots({str(pdf): pdf})
    client = TestClient(_build_app(roots, assets))

    r = client.post("/api/fileroots/attach", params={"path": str(pdf)})

    assert r.status_code == 200, r.text
    assert r.json()["parsedMd"] == "# library markdown"
    assert assets.read("PDFs/vault.pdf").content == b"library bytes"


def test_attach_root_pdf_refresh_by_logical_key(assets, no_real_extraction):
    """Send-time refresh re-calls attach with the logical key this endpoint returned, no file root lookup needed."""
    assets.write("pdf", "PDFs/vault.pdf", b"library bytes", mime="application/pdf")
    assets.write("mineru_markdown", library_markdown_key("vault"), b"# ready", mime="text/markdown")

    roots = FakeFileRoots({})  # empty: proves roots.resolve() is never consulted for a logical-key refresh
    client = TestClient(_build_app(roots, assets))

    r = client.post("/api/fileroots/attach", params={"path": "PDFs/vault.pdf"})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["path"] == "PDFs/vault.pdf"
    assert body["parsedMd"] == "# ready"


def test_attach_root_pdf_refresh_unknown_key_is_not_found(assets):
    roots = FakeFileRoots({})
    client = TestClient(_build_app(roots, assets))

    r = client.post("/api/fileroots/attach", params={"path": "PDFs/ghost.pdf"})

    assert r.status_code == 404


def test_attach_root_text_file_returns_content(tmp_path, assets, no_real_extraction):
    """Non-PDF file root attach stays copy-free and returns the file's text content."""
    txt = tmp_path / "notes.md"
    txt.write_text("# file root note", encoding="utf-8")

    roots = FakeFileRoots({str(txt): txt})
    client = TestClient(_build_app(roots, assets))

    r = client.post("/api/fileroots/attach", params={"path": str(txt)})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["content"] == "# file root note"
    assert body["parsedMd"] is None
    assert assets.list() == []
    assert no_real_extraction == []
