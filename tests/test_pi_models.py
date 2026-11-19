"""The Pi model source: catalog from pi-host's gateway, effort ladders, LiteLLM routing and stored row folding."""

import os

import httpx
import pytest

from lib import pi_host, pi_models
from lib.model_provider_store import providers_for_ui

ENTRIES = [
    {
        "id": "anthropic/claude-opus-4-5",
        "object": "model",
        "owned_by": "anthropic",
        "name": "Claude Opus 4.5",
        "context_window": 200000,
        "max_tokens": 64000,
        "input": ["text", "image"],
        "reasoning": True,
        "thinking_levels": ["off", "minimal", "low", "medium", "high"],
    },
    {"id": "anthropic/claude-3-haiku", "owned_by": "anthropic", "name": "Claude Haiku 3", "input": ["text"], "thinking_levels": []},
    {"id": "openrouter/vendor/model", "owned_by": "openrouter"},
    {"id": "", "owned_by": "anthropic"},
]


@pytest.fixture(autouse=True)
def _isolated_pi_host(monkeypatch, tmp_path):
    monkeypatch.setenv("PI_HOST_TOKEN_FILE", str(tmp_path / "pi-host.token"))
    monkeypatch.setenv("GIGACHAD_PI_HOST_URL", "http://127.0.0.1:8399")
    pi_models.reset_cache()
    yield
    pi_models.reset_cache()


def _start_pi_host(tmp_path, token: str = "host-token") -> None:
    (tmp_path / "pi-host.token").write_text(f"{token}\n", encoding="utf-8")


def _serve(monkeypatch, handler) -> list[httpx.Request]:
    """Answer the gateway's HTTP calls with *handler*, recording each request."""
    requests: list[httpx.Request] = []
    transport = httpx.MockTransport(lambda request: requests.append(request) or handler(request))
    monkeypatch.setattr(pi_models.httpx, "get", lambda url, **kwargs: httpx.Client(transport=transport).get(url, **kwargs))
    return requests


def test_catalog_without_a_started_pi_host_does_not_probe(monkeypatch):
    def fail() -> list[dict]:
        raise AssertionError("must not probe a gateway pi-host never started")

    monkeypatch.setattr(pi_models, "catalog_entries", fail)
    catalog = pi_models.catalog()

    assert (catalog["installed"], catalog["online"], catalog["providers"]) == (False, False, [])
    assert "pi-host" in catalog["error"]


def test_catalog_groups_gateway_models_by_provider(tmp_path, monkeypatch):
    _start_pi_host(tmp_path)
    requests = _serve(monkeypatch, lambda _request: httpx.Response(200, json={"object": "list", "data": ENTRIES}))

    catalog = pi_models.catalog()

    assert str(requests[0].url) == "http://127.0.0.1:8399/v1/models"
    assert requests[0].headers["Authorization"] == "Bearer host-token"
    assert (catalog["installed"], catalog["online"], catalog["litellm_id"]) == (True, True, "litellm_proxy")
    anthropic, openrouter = catalog["providers"]
    assert (anthropic["label"], openrouter["label"]) == ("Anthropic", "OpenRouter")
    # Sorted by id, blank ids dropped, vision from the input modalities, a bare row still has a usable name.
    assert [model["id"] for model in anthropic["models"]] == ["anthropic/claude-3-haiku", "anthropic/claude-opus-4-5"]
    assert [model["vision"] for model in anthropic["models"]] == [False, True]
    assert anthropic["models"][1]["context_window"] == 200000
    assert openrouter["models"][0]["name"] == "openrouter/vendor/model"


def test_catalog_reports_an_offline_gateway_instead_of_raising(tmp_path, monkeypatch):
    _start_pi_host(tmp_path)

    def refuse(_request):
        raise httpx.ConnectError("connection refused")

    _serve(monkeypatch, refuse)
    catalog = pi_models.catalog()

    assert (catalog["installed"], catalog["online"]) == (True, False)
    assert "connection refused" in catalog["error"]


def test_thinking_levels_are_pis_ladder_and_unknown_reads_as_none(tmp_path, monkeypatch):
    _start_pi_host(tmp_path)
    _serve(monkeypatch, lambda _request: httpx.Response(200, json={"data": ENTRIES}))

    assert pi_models.thinking_levels("anthropic/claude-opus-4-5") == ["off", "minimal", "low", "medium", "high"]
    assert pi_models.thinking_levels("anthropic/claude-3-haiku") == []
    assert pi_models.thinking_levels("anthropic/unlisted") is None


def test_thinking_registry_raises_where_thinking_levels_reads_unknown(tmp_path, monkeypatch):
    _start_pi_host(tmp_path)
    _serve(monkeypatch, lambda _request: httpx.Response(503, json={"error": {"message": "down"}}))

    with pytest.raises(pi_models.GatewayUnavailable):
        pi_models.thinking_registry()
    assert pi_models.thinking_levels("anthropic/claude-opus-4-5") is None


def test_configure_litellm_proxy_points_the_prefix_at_the_gateway(tmp_path, monkeypatch):
    _start_pi_host(tmp_path)
    monkeypatch.delenv("LITELLM_PROXY_API_BASE", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)

    pi_models.configure_litellm_proxy()

    assert os.environ["LITELLM_PROXY_API_BASE"] == pi_host.gateway_url() == "http://127.0.0.1:8399/v1"
    assert os.environ["LITELLM_PROXY_API_KEY"] == "host-token"


def test_configure_litellm_proxy_without_a_token_leaves_the_key_unset(monkeypatch):
    monkeypatch.delenv("LITELLM_PROXY_API_BASE", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)

    pi_models.configure_litellm_proxy()

    assert "LITELLM_PROXY_API_KEY" not in os.environ


def test_configure_litellm_proxy_keeps_an_operator_override(tmp_path, monkeypatch):
    _start_pi_host(tmp_path)
    monkeypatch.setenv("LITELLM_PROXY_API_BASE", "http://elsewhere/v1")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "operator-key")

    pi_models.configure_litellm_proxy()

    assert os.environ["LITELLM_PROXY_API_BASE"] == "http://elsewhere/v1"
    assert os.environ["LITELLM_PROXY_API_KEY"] == "operator-key"


def test_legacy_per_login_rows_fold_into_one(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    stored = {
        "Gemini": {"litellm_id": "gemini", "models": ["gemini-3.1-pro"]},
        "OMP · Anthropic": {"litellm_id": "litellm_proxy", "models": ["anthropic/claude-opus-5"], "source": "omp"},
        "OMP · OpenAI Codex": {
            "litellm_id": "litellm_proxy",
            "models": ["openai-codex/gpt-5.2", "anthropic/claude-opus-5"],
            "source": "omp",
        },
    }

    visible = providers_for_ui(stored)

    assert list(visible) == ["Gemini", "OMP"]
    # Merged in row order, deduped, and the other provider is untouched.
    assert visible["OMP"] == {
        "litellm_id": "litellm_proxy",
        "models": ["anthropic/claude-opus-5", "openai-codex/gpt-5.2"],
        "source": "omp",
    }
    assert visible["Gemini"] is stored["Gemini"]
