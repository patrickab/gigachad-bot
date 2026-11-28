"""Ollama discovery and its Pi provider in `models.json`, against a fake Ollama API."""

from collections.abc import Callable
import functools
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest

from backend.identity import get_request_identity
from backend.routes import models as models_route
from backend.routes.deps import get_model_provider_store
from config import OLLAMA_BASE_URL
from lib import ollama_models
from tests.conftest import request_identity

QWEN = {
    "capabilities": ["completion", "tools", "thinking"],
    "model_info": {"general.architecture": "qwen3", "qwen3.context_length": 40960},
}
LLAVA = {"capabilities": ["completion", "vision"], "model_info": {"general.architecture": "llama"}}
EMBEDDING = {"capabilities": ["embedding"], "model_info": {"general.architecture": "nomic-bert"}}


class FakeOllama:
    """Ollama's `/api/tags` and `/api/show` over *installed* (tag -> details), counting the calls of both."""

    def __init__(self, installed: dict[str, dict[str, Any]]) -> None:
        self.installed = installed
        self.listed = 0
        self.shown: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            self.listed += 1
            return httpx.Response(200, json={"models": [{"name": tag, "digest": f"sha-{tag}"} for tag in self.installed]})
        tag = json.loads(request.content)["model"]
        self.shown.append(tag)
        return httpx.Response(200, json=self.installed[tag])


@pytest.fixture(autouse=True)
def no_cached_details(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama_models, "_details_by_digest", {})


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Callable[[Callable[[httpx.Request], httpx.Response]], None]:
    """Route the module's Ollama client to *handler*."""

    def install(handler: Callable[[httpx.Request], httpx.Response]) -> None:
        client = functools.partial(httpx.Client, transport=httpx.MockTransport(handler))
        monkeypatch.setattr(ollama_models.httpx, "Client", client)

    return install


def _ollama_down(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def models_json(agent_dir: Path) -> dict[str, Any]:
    return json.loads((agent_dir / "models.json").read_text())


def test_lists_chat_models_without_embedding_models(serve, scratch_pi_agent_dir: Path) -> None:
    serve(FakeOllama({"qwen3:0.6b": QWEN, "nomic-embed-text:latest": EMBEDDING, "llava:7b": LLAVA}))
    assert ollama_models.sync_pi_provider() == ["qwen3:0.6b", "llava:7b"]


def test_ollama_down_lists_nothing_and_writes_nothing(serve, scratch_pi_agent_dir: Path) -> None:
    serve(_ollama_down)
    assert ollama_models.sync_pi_provider() == []
    assert not (scratch_pi_agent_dir / "models.json").exists()


def test_sync_writes_the_provider_and_keeps_the_rest(serve, scratch_pi_agent_dir: Path) -> None:
    other = {"baseUrl": "http://vllm:8000/v1", "api": "openai-completions", "apiKey": "x", "models": [{"id": "m"}]}
    (scratch_pi_agent_dir / "models.json").write_text(
        json.dumps({"$schema": "https://pi.dev/schemas/models.schema.json", "providers": {"vllm": other}})
    )
    serve(FakeOllama({"qwen3:0.6b": QWEN, "llava:7b": LLAVA, "bge-m3:latest": EMBEDDING}))

    ollama_models.sync_pi_provider()

    written = models_json(scratch_pi_agent_dir)
    assert written["$schema"] == "https://pi.dev/schemas/models.schema.json"
    assert written["providers"]["vllm"] == other
    assert written["providers"]["ollama"] == {
        "baseUrl": f"{OLLAMA_BASE_URL.rstrip('/')}/v1",
        "api": "openai-completions",
        "apiKey": "ollama",
        "models": [
            {
                "id": "qwen3:0.6b",
                "input": ["text"],
                "reasoning": True,
                "thinkingLevelMap": {"off": "none", "minimal": None},
                "contextWindow": 40960,
            },
            {"id": "llava:7b", "input": ["text", "image"], "reasoning": False},
        ],
    }


def test_unchanged_sync_neither_rewrites_nor_asks_ollama_again(serve, scratch_pi_agent_dir: Path) -> None:
    ollama = FakeOllama({"qwen3:0.6b": QWEN})
    serve(ollama)
    ollama_models.sync_pi_provider()
    written = scratch_pi_agent_dir / "models.json"
    before = written.stat()

    assert ollama_models.sync_pi_provider() == ["qwen3:0.6b"]

    after = written.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert ollama.shown == ["qwen3:0.6b"]


def test_a_new_pull_and_a_removal_follow_the_installed_models(serve, scratch_pi_agent_dir: Path) -> None:
    ollama = FakeOllama({"qwen3:0.6b": QWEN})
    serve(ollama)
    ollama_models.sync_pi_provider()

    ollama.installed["llava:7b"] = LLAVA
    ollama_models.sync_pi_provider()
    assert [model["id"] for model in models_json(scratch_pi_agent_dir)["providers"]["ollama"]["models"]] == [
        "qwen3:0.6b",
        "llava:7b",
    ]

    ollama.installed.clear()
    assert ollama_models.sync_pi_provider() == []
    assert models_json(scratch_pi_agent_dir)["providers"]["ollama"]["models"] == []


def test_no_chat_models_and_no_provider_writes_nothing(serve, scratch_pi_agent_dir: Path) -> None:
    serve(FakeOllama({"nomic-embed-text:latest": EMBEDDING}))
    assert ollama_models.sync_pi_provider() == []
    assert not (scratch_pi_agent_dir / "models.json").exists()


def test_user_keys_on_the_provider_survive(serve, scratch_pi_agent_dir: Path) -> None:
    compat = {"supportsDeveloperRole": False}
    (scratch_pi_agent_dir / "models.json").write_text(json.dumps({"providers": {"ollama": {"compat": compat}}}))
    serve(FakeOllama({"qwen3:0.6b": QWEN}))
    ollama_models.sync_pi_provider()
    assert models_json(scratch_pi_agent_dir)["providers"]["ollama"]["compat"] == compat


def test_an_invalid_file_is_never_overwritten(serve, scratch_pi_agent_dir: Path) -> None:
    (scratch_pi_agent_dir / "models.json").write_text("{not json")
    serve(FakeOllama({"qwen3:0.6b": QWEN}))
    assert ollama_models.sync_pi_provider() == ["qwen3:0.6b"]
    assert (scratch_pi_agent_dir / "models.json").read_text() == "{not json"


class StaticStore:
    """The model provider store reads `/api/models` makes, empty."""

    def load(self) -> dict:
        return {}

    def load_defaults(self) -> dict:
        return {}

    def load_tab_order(self) -> list[str]:
        return []


def test_models_route_lists_selectors_and_syncs_pi_from_one_listing(serve, scratch_pi_agent_dir: Path) -> None:
    ollama = FakeOllama({"qwen3:0.6b": QWEN, "nomic-embed-text:latest": EMBEDDING})
    serve(ollama)
    app = FastAPI()
    app.include_router(models_route.router)
    app.dependency_overrides[get_model_provider_store] = StaticStore
    app.dependency_overrides[get_request_identity] = request_identity

    response = TestClient(app).get("/api/models")

    assert response.status_code == 200
    assert response.json()["ollama"] == ["ollama/qwen3:0.6b"]
    assert ollama.listed == 1
    assert [model["id"] for model in models_json(scratch_pi_agent_dir)["providers"]["ollama"]["models"]] == ["qwen3:0.6b"]
