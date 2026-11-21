"""Ollama discovery against a fake Ollama API."""

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
from lib import ollama_models
from tests.conftest import request_identity

QWEN = {
    "capabilities": ["completion", "tools", "thinking"],
    "model_info": {"general.architecture": "qwen3", "qwen3.context_length": 40960},
}
LLAVA = {"capabilities": ["completion", "vision"], "model_info": {"general.architecture": "llama"}}
EMBEDDING = {"capabilities": ["embedding"], "model_info": {"general.architecture": "nomic-bert"}}


class FakeOllama:
    """Ollama's `/api/tags` and `/api/show` over *installed* (tag -> details), counting `/api/show` calls."""

    def __init__(self, installed: dict[str, dict[str, Any]]) -> None:
        self.installed = installed
        self.shown: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
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


def test_lists_chat_models_without_embedding_models(serve) -> None:
    serve(FakeOllama({"qwen3:0.6b": QWEN, "nomic-embed-text:latest": EMBEDDING, "llava:7b": LLAVA}))
    assert ollama_models.installed_chat_models() == ["qwen3:0.6b", "llava:7b"]


def test_ollama_down_lists_nothing(serve) -> None:
    serve(_ollama_down)
    assert ollama_models.installed_chat_models() == []


class StaticStore:
    """The model provider store reads `/api/models` makes, empty."""

    def load(self) -> dict:
        return {}

    def load_defaults(self) -> dict:
        return {}

    def load_tab_order(self) -> list[str]:
        return []


def test_models_route_lists_selectors_and_leaves_pi_alone(serve, scratch_pi_agent_dir: Path) -> None:
    serve(FakeOllama({"qwen3:0.6b": QWEN, "nomic-embed-text:latest": EMBEDDING}))
    app = FastAPI()
    app.include_router(models_route.router)
    app.dependency_overrides[get_model_provider_store] = StaticStore
    app.dependency_overrides[get_request_identity] = request_identity

    response = TestClient(app).get("/api/models")

    assert response.status_code == 200
    assert response.json()["ollama"] == ["ollama/qwen3:0.6b"]
    assert list(scratch_pi_agent_dir.iterdir()) == []
