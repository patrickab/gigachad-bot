"""Tests for ``lib.attachment_materialize`` — the shared attach/parse core
extracted during the refactor.

These cover:
  - the latent ``Path.stem`` bug the refactor fixes (dotted PDF names like
    ``paper.v1.pdf`` were truncated to ``paper``; now only ``.pdf`` is stripped).
  - PDF cache hit returns parsed markdown; cache miss enqueues background
    extraction by default, and skips enqueueing when ``enqueue_on_miss=False``
    (the path ``files.parse_attachments`` uses for synchronous parsing).
  - text/* files are read as utf-8; undecodable files yield ``content=None``
    instead of raising.
"""

from pathlib import Path

from lib.attachment_materialize import Materialized, materialize, mineru_cache_path

# --- mineru_cache_path -----------------------------------------------------


def test_cache_path_strips_only_pdf_suffix(mineru_cache_dir: Path):
    """``Path.stem`` would turn ``paper.v1.pdf`` into ``paper`` — the helper preserves the dotted stem."""
    assert mineru_cache_path("paper.v1.pdf").name == "paper.v1.md"
    assert mineru_cache_path("paper v1.2.pdf").name == "paper v1.2.md"
    assert mineru_cache_path("plain.pdf").name == "plain.md"


def test_cache_path_accepts_full_path(tmp_path: Path, mineru_cache_dir: Path):
    pdf = tmp_path / "report.v3.pdf"
    assert mineru_cache_path(pdf).name == "report.v3.md"


def test_cache_path_passes_through_stem_without_pdf_suffix(mineru_cache_dir: Path):
    """``enqueue`` calls this with bare stems too; ensure non-.pdf names are unchanged."""
    assert mineru_cache_path("report").name == "report.md"
    assert mineru_cache_path("already.md").name == "already.md.md"


# --- materialize: PDF cache hit -------------------------------------------


def test_materialize_pdf_cache_hit_returns_md(tmp_path: Path, mineru_cache_dir: Path, no_enqueue):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    (mineru_cache_dir / "doc.md").write_text("# cached", encoding="utf-8")

    result = materialize(pdf)

    assert isinstance(result, Materialized)
    assert result.mime == "application/pdf"
    assert result.parsed_md == "# cached"
    assert result.content is None
    # A cache hit must never enqueue — extraction is already done.
    assert no_enqueue == []


def test_materialize_pdf_cache_miss_enqueues(tmp_path: Path, mineru_cache_dir: Path, no_enqueue):
    pdf = tmp_path / "unseen.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    result = materialize(pdf)

    assert result.parsed_md is None
    assert no_enqueue == [pdf]


def test_materialize_pdf_cache_miss_no_enqueue_when_disabled(tmp_path: Path, mineru_cache_dir: Path, no_enqueue):
    """``enqueue_on_miss=False`` is the path used by ``files.parse_attachments`` which parses synchronously."""
    pdf = tmp_path / "sync.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    result = materialize(pdf, enqueue_on_miss=False)

    assert result.parsed_md is None
    assert no_enqueue == []  # caller handles the miss itself


# --- materialize: text files ----------------------------------------------


def test_materialize_text_file_reads_utf8(tmp_path: Path, mineru_cache_dir: Path, no_enqueue):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello world", encoding="utf-8")

    result = materialize(txt)

    assert result.content == "hello world"
    assert result.parsed_md is None
    assert no_enqueue == []


def test_materialize_text_file_non_utf8_returns_none(tmp_path: Path, mineru_cache_dir: Path, no_enqueue):
    """Refactor preserved the silent-None contract on ``UnicodeDecodeError``."""
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"\xff\xfe\x00\x01 not utf-8")

    result = materialize(binary)

    assert result.content is None
    assert no_enqueue == []


def test_materialize_markdown_uses_text_branch(tmp_path: Path, mineru_cache_dir: Path, no_enqueue):
    md = tmp_path / "spec.md"
    md.write_text("## heading", encoding="utf-8")

    result = materialize(md)

    assert result.mime == "text/markdown"
    assert result.content == "## heading"


# --- materialize: non-text, non-pdf --------------------------------------


def test_materialize_unknown_mime_returns_empty_result(tmp_path: Path, mineru_cache_dir: Path, no_enqueue, monkeypatch):
    """A binary file (e.g. .png) has no parsed_md and no content; neither branch fires."""
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n fake png")

    result = materialize(img)

    assert result.parsed_md is None
    assert result.content is None
    assert no_enqueue == []
    # mime_for is suffix-driven; a .png yields image/png.
    assert result.mime == "image/png"
