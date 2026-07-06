"""Route-level regression tests for ``file_vaults.attach_file``.

The refactor swapped the inline ``DIRECTORY_OUTPUT_MINERU / f"{stem}.md"`` lookup
for ``materialize(canonical).parsed_md``. The two behaviors the refactor must
preserve on the vault PDF attach path are:

  1. **Cache hit** — an already-extracted PDF returns the cached markdown.
  2. **Cache miss** — the PDF is *promoted* into the cloud library via
     ``organize_file`` (the inline copy) AND background extraction is queued
     (the default ``enqueue_on_miss=True`` of ``materialize``).

The pre-refactor code did both; this test pins them so the extraction cannot
silently regress to "attach returns parsedMd=None forever".
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routes import file_vaults
import lib.document_library as lib_docs


class FakeVault:
    """Minimal FileVault stub: resolve() maps a virtual path to a tmp file."""

    def __init__(self, mapping: dict[str, Path]):
        self._mapping = mapping

    def resolve(self, path: str) -> Path:
        if path not in self._mapping:
            raise FileNotFoundError(path)
        return self._mapping[path]

    def read(self, path: str) -> str:
        return self._mapping[path].read_text(encoding="utf-8")


def _build_app(vault: FakeVault) -> FastAPI:
    app = FastAPI()
    app.include_router(file_vaults.router)
    app.dependency_overrides[file_vaults.get_file_vault] = lambda: vault
    return app


def test_attach_vault_pdf_cache_hit_returns_md(tmp_path: Path, mineru_cache_dir: Path, monkeypatch):
    pdf = tmp_path / "vault.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    (mineru_cache_dir / "vault.md").write_text("# already extracted", encoding="utf-8")

    # Library dir empty so promotion runs — but the cache hit is what we assert.
    lib_dir = tmp_path / "library"
    lib_dir.mkdir()
    monkeypatch.setattr(lib_docs, "LIBRARY_DIR", lib_dir, raising=True)

    enqueued: list[Path] = []
    monkeypatch.setattr(
        "lib.attachment_materialize.extract_queue.enqueue",
        lambda p: enqueued.append(p),
    )

    vault = FakeVault({"v/vault.pdf": pdf})
    client = TestClient(_build_app(vault))

    r = client.post("/api/filevaults/attach", params={"path": "v/vault.pdf"})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["parsedMd"] == "# already extracted"
    assert body["name"] == "vault.pdf"
    # Cache hit → no enqueue.
    assert enqueued == []


def test_attach_vault_pdf_cache_miss_promotes_and_enqueues(tmp_path: Path, mineru_cache_dir: Path, monkeypatch):
    pdf = tmp_path / "report.v1.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    # Library dir is real so organize_file copies the PDF there.
    lib_dir = tmp_path / "library"
    lib_dir.mkdir()
    monkeypatch.setattr(lib_docs, "LIBRARY_DIR", lib_dir, raising=True)

    enqueued: list[Path] = []
    monkeypatch.setattr(
        "lib.attachment_materialize.extract_queue.enqueue",
        lambda p: enqueued.append(p),
    )

    vault = FakeVault({"vaults/report.v1.pdf": pdf})
    client = TestClient(_build_app(vault))

    r = client.post("/api/filevaults/attach", params={"path": "vaults/report.v1.pdf"})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["parsedMd"] is None  # cache miss → no inline markdown
    # Promotion: the library now contains the promoted PDF.
    assert (lib_dir / "report.v1.pdf").is_file()
    # The canonical path returned points at the promoted library copy.
    assert Path(body["path"]).name == "report.v1.pdf"
    # Background extraction was queued for the canonical (promoted) path.
    assert len(enqueued) == 1
    assert enqueued[0].name == "report.v1.pdf"


def test_attach_vault_text_file_returns_content(tmp_path: Path, mineru_cache_dir: Path, monkeypatch):
    """Non-PDF vault attach stays copy-free and returns the file's text content."""
    txt = tmp_path / "notes.md"
    txt.write_text("# vault note", encoding="utf-8")

    enqueued: list[Path] = []
    monkeypatch.setattr(
        "lib.attachment_materialize.extract_queue.enqueue",
        lambda p: enqueued.append(p),
    )

    vault = FakeVault({"vaults/notes.md": txt})
    client = TestClient(_build_app(vault))

    r = client.post("/api/filevaults/attach", params={"path": "vaults/notes.md"})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["content"] == "# vault note"
    assert body["parsedMd"] is None
    assert enqueued == []
