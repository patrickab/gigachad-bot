"""Background MinerU extraction queue.

Single worker processes PDFs one at a time so MinerU servers don't fight
for resources. Each queued PDF carries the callback that persists its finished
parse (``attachment_materialize`` stores it in the requesting user's library).
"""

import asyncio
from collections.abc import Callable
import hashlib
import logging
from pathlib import Path
import tempfile

from config import DIRECTORY_OUTPUT_MINERU

# Bound as a module so tests can replace ``mineru.parse_pdf``.
from lib import mineru

log = logging.getLogger(__name__)

# Persists a finished parse, given the PDF stem, the markdown file and the images directory.
StoreParse = Callable[[str, Path, Path], None]

_queue: asyncio.Queue[tuple[str, bytes, StoreParse] | None] = asyncio.Queue()
_worker_task: asyncio.Task | None = None
# PDFs queued or extracting, by name and content hash: live references re-attach the same PDF on every send.
_pending: set[tuple[str, str]] = set()


def _pending_key(name: str, content: bytes) -> tuple[str, str]:
    return name, hashlib.sha256(content).hexdigest()


async def _worker() -> None:
    while True:
        item = await _queue.get()
        if item is None:  # shutdown sentinel from stop()
            _queue.task_done()
            break
        name, content, store = item
        try:
            log.info("Extracting %s via MinerU", name)
            with tempfile.TemporaryDirectory() as tmp:
                pdf_path = Path(tmp) / name
                pdf_path.write_bytes(content)
                md_path, images_dir = await mineru.parse_pdf(pdf_path, DIRECTORY_OUTPUT_MINERU)
                store(pdf_path.stem, md_path, images_dir)
            log.info("Extraction complete: %s", name)
        except Exception:
            log.exception("Background MinerU extraction failed for %s", name)
        finally:
            _pending.discard(_pending_key(name, content))
            _queue.task_done()


def enqueue(name: str, content: bytes, store: StoreParse) -> None:
    """Queue a PDF for background extraction, unless these bytes are already queued or extracting. *store* persists the parse."""
    key = _pending_key(name, content)
    if key in _pending:
        return
    _pending.add(key)
    _queue.put_nowait((name, content, store))
    log.info("Queued for extraction: %s (queue depth: %d)", name, _queue.qsize())


async def start() -> None:
    global _worker_task
    _worker_task = asyncio.create_task(_worker(), name="mineru-extract-worker")


async def stop() -> None:
    # None is the sentinel that makes _worker exit after draining earlier items.
    _queue.put_nowait(None)
    if _worker_task:
        try:
            await asyncio.wait_for(_worker_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            _worker_task.cancel()
