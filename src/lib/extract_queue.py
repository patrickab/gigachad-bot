"""Background MinerU extraction queue.

Single worker processes PDFs one at a time so MinerU servers don't fight
for resources. Enqueue at upload or promote-to-context time; extraction
runs in the background without blocking the response. A finished parse is
persisted into the requesting user's Postgres library cache — the on-disk
Nextcloud mirror is never read back by the app.
"""

import asyncio
import logging
import tempfile
from pathlib import Path

from config import DIRECTORY_OUTPUT_MINERU
from lib.asset_store import AssetStore

# ``lib.mineru`` and ``lib.attachment_materialize`` both reach back here, so
# bind the modules rather than their members and read through them at call time.
from lib import attachment_materialize
from lib import mineru

log = logging.getLogger(__name__)

_queue: asyncio.Queue[tuple[str, bytes, AssetStore] | None] = asyncio.Queue()
_worker_task: asyncio.Task | None = None
_in_progress: str | None = None


async def _worker() -> None:
    global _in_progress
    while True:
        item = await _queue.get()
        if item is None:
            _queue.task_done()
            break
        name, content, assets = item
        _in_progress = name
        try:
            log.info("Extracting %s via MinerU", name)
            with tempfile.TemporaryDirectory() as tmp:
                pdf_path = Path(tmp) / name
                pdf_path.write_bytes(content)
                md_path, images_dir = await mineru.parse_pdf(pdf_path, DIRECTORY_OUTPUT_MINERU)
                attachment_materialize.store_library_output(assets, pdf_path.stem, md_path, images_dir)
            log.info("Extraction complete: %s", name)
        except Exception:
            log.exception("Background MinerU extraction failed for %s", name)
        finally:
            _in_progress = None
            _queue.task_done()


def enqueue(name: str, content: bytes, assets: AssetStore) -> None:
    """Queue a PDF for background extraction and Postgres persistence."""
    _queue.put_nowait((name, content, assets))
    log.info("Queued for extraction: %s (queue depth: %d)", name, _queue.qsize())



def status() -> dict:
    return {
        "in_progress": _in_progress,
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
