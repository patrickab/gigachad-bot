"""Route-level tests for the notebook HTTP API.

Stages real SandboxService state (memory stores, fake runner) so revision
arbitration and the run seam are exercised end-to-end without Docker or
Postgres. `run_notebook_cells` is stubbed at the route module boundary:
its execution logic belongs to the harness tests.
"""

from __future__ import annotations
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routes import notebook
from backend.routes.deps import get_chat_store, get_sandbox_service
from lib.agent_sandbox_adapter import FakeSandboxRunner
from lib.sandbox_service import SandboxRuntimeState, SandboxService
from tests._memory_stores import MemoryAssetStore, MemoryDataStore

SOURCE = "# %% [markdown]\n# Title\n# %%\nprint('hi')\n"


def _service() -> SandboxService:
    return SandboxService(
        data_store=MemoryDataStore(),
        asset_store=MemoryAssetStore(),
        runner=FakeSandboxRunner(),
        runtime_state=SandboxRuntimeState(),
        model="test-model",
    )


@pytest.fixture
def client(monkeypatch):
    """App with a per-test SandboxService; identity deps never trigger (no overrides used)."""
    service = _service()
    app = FastAPI()
    app.include_router(notebook.router)
    app.dependency_overrides[get_sandbox_service] = lambda: service
    app.dependency_overrides[get_chat_store] = lambda: SimpleNamespace(find_by_chat_id=lambda _chat_id: None)

    async def fake_run(sandbox, chat_id, upto, *, timeout=120.0):
        current = await sandbox.read_notebook(chat_id)
        if current is None:
            raise LookupError(chat_id)
        # Mimic the harness: same source, new revision, outputs keyed by cell sha.
        revision_id = f"run-{upto}-{len(fake_run.calls)}"
        fake_run.calls.append((chat_id, upto))
        await sandbox.stage_notebook(chat_id, revision_id, current["source"], {"outputs": [{"type": "stdout", "text": "hi"}]})
        return {"revision_id": revision_id, "outputs": {"outputs": [{"type": "stdout", "text": "hi"}]}}

    fake_run.calls = []
    monkeypatch.setattr(notebook, "run_notebook_cells", fake_run)
    return TestClient(app)


def test_get_before_put_is_404(client):
    assert client.get("/api/notebook/chat-a").status_code == 404


def test_put_then_get_round_trips(client):
    put = client.put("/api/notebook/chat-a", json={"source": SOURCE, "outputs": {}, "base_revision": ""})
    assert put.status_code == 200
    revision_id = put.json()["revision_id"]
    assert revision_id

    got = client.get("/api/notebook/chat-a")
    assert got.status_code == 200
    body = got.json()
    assert body["source"] == SOURCE
    assert body["outputs"] == {}
    assert body["revision_id"] == revision_id


def test_put_with_stale_base_revision_is_409(client):
    first = client.put("/api/notebook/chat-a", json={"source": SOURCE, "outputs": {}, "base_revision": ""}).json()
    stale = client.put(
        "/api/notebook/chat-a",
        json={"source": SOURCE + "# %%\nx = 2\n", "outputs": {}, "base_revision": "not-the-current-revision"},
    )
    assert stale.status_code == 409
    # The rejected write must not have replaced the current revision.
    assert client.get("/api/notebook/chat-a").json()["revision_id"] == first["revision_id"]


def test_put_with_current_base_revision_replaces(client):
    first = client.put("/api/notebook/chat-a", json={"source": SOURCE, "outputs": {}, "base_revision": ""}).json()
    second_source = "# %%\nprint('bye')\n"
    second = client.put(
        "/api/notebook/chat-a", json={"source": second_source, "outputs": {}, "base_revision": first["revision_id"]}
    )
    assert second.status_code == 200
    body = client.get("/api/notebook/chat-a").json()
    assert body["source"] == second_source
    assert body["revision_id"] == second.json()["revision_id"] != first["revision_id"]


def test_run_through_real_harness_stages_outputs(client, monkeypatch):
    """Route -> real run_notebook_cells -> harness script, run in a local Python subprocess."""
    import subprocess

    class LocalPythonRunner:
        async def run_script(self, script, *, profile, interpreter, timeout):
            del profile, interpreter, timeout
            return subprocess.run(["python", "-c", script], capture_output=True, text=True, check=True).stdout

    # Swap in one shared service whose runner executes the harness locally.
    local_service = SandboxService(
        data_store=MemoryDataStore(),
        asset_store=MemoryAssetStore(),
        runner=LocalPythonRunner(),
        runtime_state=SandboxRuntimeState(),
        model="test-model",
    )
    client.app.dependency_overrides[get_sandbox_service] = lambda: local_service
    import backend.routes.notebook as nb
    import lib.sandbox_notebook as harness

    monkeypatch.setattr(nb, "run_notebook_cells", harness.run_notebook_cells)

    source = "# %%\nprint('hello')\n42\n"
    client.put("/api/notebook/chat-b", json={"source": source, "outputs": {}, "base_revision": ""})
    run = client.post("/api/notebook/chat-b/run", json={"upto": 1})
    assert run.status_code == 200
    body = run.json()

    assert body["outputs"], "run must produce output records"
    records = next(iter(body["outputs"].values()))
    assert records[0] == {"type": "stdout", "text": "hello\n"}
    assert records[1] == {"type": "result", "text": "42"}

    stored = client.get("/api/notebook/chat-b").json()
    assert stored["revision_id"] == body["revision_id"]
    assert stored["outputs"] == body["outputs"]


def test_run_returns_new_revision_with_outputs(client):
    client.put("/api/notebook/chat-a", json={"source": SOURCE, "outputs": {}, "base_revision": ""})
    run = client.post("/api/notebook/chat-a/run", json={"upto": 1})
    assert run.status_code == 200
    revision_id = run.json()["revision_id"]

    body = client.get("/api/notebook/chat-a").json()
    assert body["revision_id"] == revision_id
    assert body["outputs"] != {}


def test_run_without_notebook_is_404(client):
    assert client.post("/api/notebook/chat-a/run", json={"upto": 1}).status_code == 404