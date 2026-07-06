"""Tests for ``lib.extract_queue`` enqueue path.

The refactor routes the cache lookup through ``mineru_cache_path``. The pre-refactor
code did ``DIRECTORY_OUTPUT_MINERU / f"{pdf_path.stem}.md"`` — for a PDF named
``paper.v1.pdf`` that produced ``paper.md`` instead of ``paper.v1.md``, so a
cache hit was missed and the file got re-queued. The post-refactor ``enqueue``
must respect the same helper the cache write uses, so an already-cached PDF is
a true no-op.
"""

from pathlib import Path

import lib.extract_queue as extract_queue


def test_enqueue_is_noop_when_cache_present(tmp_path: Path, mineru_cache_dir: Path, monkeypatch):
    """A PDF whose cache already exists must not be put on the queue."""
    pdf = tmp_path / "report.v1.pdf"
    pdf.write_bytes(b"%PDF")
    (mineru_cache_dir / "report.v1.md").write_text("# cached", encoding="utf-8")

    put_calls: list[Path] = []
    monkeypatch.setattr(extract_queue._queue, "put_nowait", lambda p: put_calls.append(p))

    extract_queue.enqueue(pdf)

    assert put_calls == []


def test_enqueue_queues_when_cache_absent(tmp_path: Path, mineru_cache_dir: Path, monkeypatch):
    pdf = tmp_path / "fresh.v2.pdf"
    pdf.write_bytes(b"%PDF")

    put_calls: list[Path] = []
    monkeypatch.setattr(extract_queue._queue, "put_nowait", lambda p: put_calls.append(p))

    extract_queue.enqueue(pdf)

    assert put_calls == [pdf]


def test_enqueue_uses_mineru_cache_path_helper(tmp_path: Path, mineru_cache_dir: Path, monkeypatch):
    """Enqueue must look at the *same* path ``mineru_cache_path`` returns — otherwise a cache
    written by the worker would never be seen by a subsequent enqueue (the pre-refactor bug)."""
    pdf = tmp_path / "x.y.z.pdf"
    pdf.write_bytes(b"%PDF")
    # Write the cache at the *helper's* path (not the buggy ``stem`` path).
    from lib.attachment_materialize import mineru_cache_path

    cached = mineru_cache_path(pdf)
    cached.write_text("# cached", encoding="utf-8")

    put_calls: list[Path] = []
    monkeypatch.setattr(extract_queue._queue, "put_nowait", lambda p: put_calls.append(p))

    extract_queue.enqueue(pdf)

    assert put_calls == []  # cache hit at the helper path, enqueue is a no-op
