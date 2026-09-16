"""Background MinerU extraction queue.

Single worker processes PDFs one at a time so MinerU servers don't fight
for resources. Enqueue at upload or promote-to-context time; extraction
runs in the background without blocking the response.
"""

import asyncio
import logging
from pathlib import Path

from config import DIRECTORY_OUTPUT_MINERU

# ``lib.mineru`` and ``lib.attachment_materialize`` both reach back here, so
# bind the modules rather than their members and read through them at call time.
from lib import attachment_materialize
from lib import mineru

log = logging.getLogger(__name__)

_queue: asyncio.Queue[Path | None] = asyncio.Queue()
_worker_task: asyncio.Task | None = None
_in_progress: Path | None = None


async def _worker() -> None:
    global _in_progress
    while True:
        pdf_path = await _queue.get()
        if pdf_path is None:
            _queue.task_done()
            break
        _in_progress = pdf_path
        try:
            cached = attachment_materialize.mineru_cache_path(pdf_path)
            if not cached.is_file():
                log.info("Extracting %s via MinerU", pdf_path.name)
                await mineru.parse_pdf(pdf_path, DIRECTORY_OUTPUT_MINERU)
                log.info("Extraction complete: %s", pdf_path.name)
        except Exception:
            log.exception("Background MinerU extraction failed for %s", pdf_path.name)
        finally:
            _in_progress = None
            _queue.task_done()


def enqueue(pdf_path: Path) -> None:
    """Queue a PDF for background extraction. No-op if already cached."""
    cached = attachment_materialize.mineru_cache_path(pdf_path)
    if cached.is_file():
        return
    _queue.put_nowait(pdf_path)
    log.info("Queued for extraction: %s (queue depth: %d)", pdf_path.name, _queue.qsize())


def status() -> dict:
    return {
        "in_progress": _in_progress.name if _in_progress else None,
        "queued": _queue.qsize(),
    }


async def start() -> None:
    global _worker_task
    _worker_task = asyncio.create_task(_worker(), name="mineru-extract-worker")


async def stop() -> None:
    _queue.put_nowait(None)
    if _worker_task:
        try:
            await asyncio.wait_for(_worker_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            _worker_task.cancel()
