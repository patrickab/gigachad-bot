"""Blocking store and file work never stalls the event loop or starves short thread offloads."""

import asyncio
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
import httpx
import pytest

from backend.identity import get_request_identity
from backend.routes import config, documents, files, models, projects
from backend.routes.deps import (
    get_asset_store,
    get_document_store,
    get_model_provider_store,
    get_project_store,
    get_prompt_store,
    get_user_preferences,
)
from backend.server import widen_default_executor
import lib.extract_queue as extract_queue
from tests.conftest import request_identity

# Long enough to notice a stalled loop, short enough that a regression fails fast.
_BLOCK_SECONDS = 2


class BlockingStore:
    """A store whose every method blocks until released (or timed out), then answers from *results*."""

    def __init__(self, results: dict[str, Any]) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.timed_out = False
        self._results = results

    def __getattr__(self, name: str) -> Any:
        result = self._results[name]

        def method(*_args: Any, **_kwargs: Any) -> Any:
            self.entered.set()
            if not self.release.wait(_BLOCK_SECONDS):
                self.timed_out = True
            return result

        return method


@pytest.mark.parametrize(
    ("path", "dependency", "results"),
    [
        ("/api/config", get_model_provider_store, {"load_defaults": {}}),
        ("/api/models", get_model_provider_store, {"load": {}, "load_defaults": {}, "load_tab_order": []}),
        ("/api/projects", get_project_store, {"list_projects": []}),
        ("/api/prompts", get_prompt_store, {"prompt_map": {}}),
        ("/api/prompts/list", get_prompt_store, {"list_prompts": []}),
        ("/api/documents/notes", get_document_store, {"list": []}),
    ],
)
async def test_a_blocked_store_read_does_not_stall_other_requests(path: str, dependency: Any, results: dict[str, Any]) -> None:
    await _assert_off_the_loop(BlockingStore(results), dependency, "GET", path)


@pytest.mark.parametrize(
    ("path", "upload", "results"),
    [
        ("/api/files/upload?chat_id=c1", {"file": ("a.txt", b"x", "text/plain")}, {"list": [], "write": None}),
        ("/api/files/parse?chat_id=c1&filenames=a.txt", None, {"read": SimpleNamespace(content=b"x")}),
    ],
)
async def test_a_blocked_chat_upload_store_does_not_stall_other_requests(path: str, upload: Any, results: dict[str, Any]) -> None:
    await _assert_off_the_loop(BlockingStore(results), get_asset_store, "POST", path, files=upload)


async def _assert_off_the_loop(store: BlockingStore, dependency: Any, method: str, path: str, **kwargs: Any) -> None:
    app = FastAPI()
    for router in (config.router, documents.router, files.router, models.router, projects.router):
        app.include_router(router)
    app.dependency_overrides[dependency] = lambda: store
    app.dependency_overrides[get_request_identity] = lambda: request_identity(has_host_access=False)
    app.dependency_overrides[get_user_preferences] = lambda: SimpleNamespace(animated_background=lambda: True)

    @app.get("/ping")
    async def ping() -> dict[str, bool]:
        return {"ok": True}

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        pending = asyncio.create_task(client.request(method, path, **kwargs))
        assert await asyncio.to_thread(store.entered.wait, _BLOCK_SECONDS)
        assert (await asyncio.wait_for(client.get("/ping"), _BLOCK_SECONDS / 2)).status_code == 200
        store.release.set()
        assert (await pending).status_code == 200
    assert not store.timed_out, f"{path} ran its store call on the event loop"


async def test_a_blocked_extraction_store_does_not_stall_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_parse_pdf(pdf_path: Path, output_dir: Path) -> tuple[Path, Path]:
        return pdf_path, output_dir

    monkeypatch.setattr(extract_queue.mineru, "parse_pdf", fake_parse_pdf)
    store = BlockingStore({"store": None})
    extract_queue.enqueue("blocked.pdf", b"%PDF blocked", store.store)
    await extract_queue.start()
    try:
        assert await asyncio.to_thread(store.entered.wait, _BLOCK_SECONDS)
        await asyncio.wait_for(asyncio.sleep(0), _BLOCK_SECONDS / 2)
    finally:
        store.release.set()
        await extract_queue.stop()
    assert not store.timed_out, "the extraction worker ran its store callback on the event loop"


async def test_long_lived_blocking_work_leaves_executor_workers_for_short_calls() -> None:
    widen_default_executor()
    release = threading.Event()
    # More holders than the stdlib default executor has workers, e.g. streaming chats and Neovim sessions.
    holders = [asyncio.create_task(asyncio.to_thread(release.wait, _BLOCK_SECONDS)) for _ in range(min(32, (os.cpu_count() or 1) + 4))]
    try:
        assert await asyncio.wait_for(asyncio.to_thread(lambda: "saved"), _BLOCK_SECONDS / 2) == "saved"
    finally:
        release.set()
        await asyncio.gather(*holders)
