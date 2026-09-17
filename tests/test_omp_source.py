import json
import os
from pathlib import Path
from uuid import uuid4

from psycopg_pool import ConnectionPool
import pytest

from lib import omp_source
from lib.db_schema import upgrade
from lib.model_provider_store import ModelProviderStore, providers_for_ui
from lib.postgres_data_store import PostgresDataStore


@pytest.fixture(autouse=True)
def _isolated_omp(monkeypatch, tmp_path):
    monkeypatch.setenv("PI_CONFIG_DIR", str(tmp_path / "omp"))
    monkeypatch.delenv("GIGACHAD_OMP_GATEWAY_URL", raising=False)
    monkeypatch.delenv("GIGACHAD_OMP_GATEWAY_TOKEN", raising=False)
    omp_source.reset_cache()
    yield
    omp_source.reset_cache()


@pytest.fixture(scope="module")
def postgres_pool():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for Postgres-backed omp source tests")
    upgrade(url)
    pool = ConnectionPool(url, min_size=1, max_size=4)
    yield pool
    pool.close()


@pytest.fixture
def postgres_store(postgres_pool):
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute("TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE")
        user_id = connection.execute(
            "INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", (f"test-{uuid4()}@example.test",)
        ).fetchone()[0]
    return PostgresDataStore(postgres_pool, user_id)


def _install_omp(tmp_path: Path, token: str | None = None) -> None:
    (tmp_path / "omp" / "agent").mkdir(parents=True)
    (tmp_path / "omp" / "agent" / "agent.db").write_bytes(b"")
    if token is not None:
        (tmp_path / "omp" / "auth-gateway.token").write_text(f"{token}\n", encoding="utf-8")


def test_catalog_reports_missing_install_without_probing(tmp_path, monkeypatch):
    def fail() -> list[dict]:
        raise AssertionError("must not probe a gateway when OMP is not installed")

    monkeypatch.setattr(omp_source, "_fetch_catalog_entries", fail)
    catalog = omp_source.catalog()

    assert catalog["installed"] is False
    assert catalog["online"] is False
    assert catalog["providers"] == []
    assert "credential store" in catalog["error"]


def test_catalog_groups_gateway_models_by_authenticated_provider(tmp_path, monkeypatch):
    _install_omp(tmp_path)
    entries = [
        {
            "id": "anthropic/claude-opus-5",
            "owned_by": "anthropic",
            "display_name": "Claude Opus 5",
            "input_modalities": ["text", "image"],
            "context_length": 200000,
        },
        {"id": "anthropic/claude-haiku-4-5", "owned_by": "anthropic", "display_name": "Claude Haiku 4.5", "input_modalities": []},
        {"id": "openai-codex/gpt-5.2", "owned_by": "openai-codex"},
        {"id": "", "owned_by": "anthropic"},
    ]
    monkeypatch.setattr(omp_source, "_fetch_catalog_entries", lambda: entries)
    catalog = omp_source.catalog()

    assert catalog["installed"] is True
    assert catalog["online"] is True
    assert catalog["litellm_id"] == "litellm_proxy"
    assert [provider["id"] for provider in catalog["providers"]] == ["anthropic", "openai-codex"]

    anthropic, codex = catalog["providers"]
    assert anthropic["label"] == "Anthropic"
    assert codex["label"] == "OpenAI Codex"
    # Sorted by id, blank ids dropped, and vision derived from the modalities.
    assert [model["id"] for model in anthropic["models"]] == ["anthropic/claude-haiku-4-5", "anthropic/claude-opus-5"]
    assert [model["vision"] for model in anthropic["models"]] == [False, True]
    # A model with no metadata still resolves a usable name.
    assert codex["models"][0]["name"] == "openai-codex/gpt-5.2"


def test_catalog_reports_offline_gateway_instead_of_raising(tmp_path, monkeypatch):
    _install_omp(tmp_path)

    def refuse() -> list[dict]:
        raise OSError("connection refused")

    monkeypatch.setattr(omp_source, "_fetch_catalog_entries", refuse)
    catalog = omp_source.catalog()

    assert catalog["installed"] is True
    assert catalog["online"] is False
    assert "connection refused" in catalog["error"]


def test_catalog_is_cached_until_refresh(tmp_path, monkeypatch):
    _install_omp(tmp_path)
    calls: list[int] = []

    def probe() -> list[dict]:
        calls.append(1)
        return [{"id": "anthropic/claude-opus-5", "owned_by": "anthropic"}]

    monkeypatch.setattr(omp_source, "_fetch_catalog_entries", probe)
    omp_source.catalog()
    omp_source.catalog()
    assert len(calls) == 1

    omp_source.catalog(refresh=True)
    assert len(calls) == 2


def test_gateway_token_prefers_env_then_file(tmp_path, monkeypatch):
    _install_omp(tmp_path, token="file-token")
    assert omp_source.gateway_token() == "file-token"

    monkeypatch.setenv("GIGACHAD_OMP_GATEWAY_TOKEN", "env-token")
    assert omp_source.gateway_token() == "env-token"


def test_gateway_token_falls_back_when_no_token_exists(tmp_path):
    _install_omp(tmp_path)
    assert omp_source.gateway_token() == omp_source.FALLBACK_TOKEN


def test_configure_litellm_proxy_scopes_the_override_to_omp(tmp_path, monkeypatch):
    _install_omp(tmp_path, token="file-token")
    monkeypatch.delenv("LITELLM_PROXY_API_BASE", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
    monkeypatch.setenv("GIGACHAD_OMP_GATEWAY_URL", "http://127.0.0.1:4321/v1/")

    omp_source.configure_litellm_proxy()

    import os

    assert os.environ["LITELLM_PROXY_API_BASE"] == "http://127.0.0.1:4321/v1"
    assert os.environ["LITELLM_PROXY_API_KEY"] == "file-token"


def test_configure_litellm_proxy_keeps_an_operator_override(tmp_path, monkeypatch):
    _install_omp(tmp_path)
    monkeypatch.setenv("LITELLM_PROXY_API_BASE", "http://elsewhere/v1")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "operator-key")

    omp_source.configure_litellm_proxy()

    import os

    assert os.environ["LITELLM_PROXY_API_BASE"] == "http://elsewhere/v1"
    assert os.environ["LITELLM_PROXY_API_KEY"] == "operator-key"


def test_model_selector_prefixes_the_proxy_hop():
    assert omp_source.model_selector("anthropic/claude-opus-5") == "litellm_proxy/anthropic/claude-opus-5"


def test_fetch_catalog_entries_sends_the_gateway_bearer(tmp_path, monkeypatch):
    _install_omp(tmp_path, token="file-token")
    seen: dict[str, object] = {}

    class _Response:
        def read(self) -> bytes:
            return json.dumps({"object": "list", "data": [{"id": "anthropic/claude-opus-5", "owned_by": "anthropic"}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(omp_source.urllib.request, "urlopen", fake_urlopen)
    entries = omp_source._fetch_catalog_entries()

    assert seen["url"] == "http://127.0.0.1:4000/v1/models"
    assert seen["auth"] == "Bearer file-token"
    assert entries[0]["id"] == "anthropic/claude-opus-5"


def test_provider_catalog_round_trips_an_omp_source(postgres_store):
    catalog = ModelProviderStore(postgres_store)
    providers = {
        "OMP": {"litellm_id": "litellm_proxy", "models": ["anthropic/claude-opus-5"], "source": "omp"},
        "Gemini": {"litellm_id": "gemini", "models": ["gemini-3.1-pro"]},
    }

    saved = catalog.save(providers)

    assert saved["OMP"]["source"] == "omp"
    # A hand-entered provider keeps its exact old shape — no null source key.
    assert saved["Gemini"] == {"litellm_id": "gemini", "models": ["gemini-3.1-pro"]}
    assert catalog.load() == saved


def test_legacy_per_login_omp_rows_fold_into_one(monkeypatch):
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
    # Merged in row order, deduped, and the non-OMP provider is untouched.
    assert visible["OMP"] == {
        "litellm_id": "litellm_proxy",
        "models": ["anthropic/claude-opus-5", "openai-codex/gpt-5.2"],
        "source": "omp",
    }
    assert visible["Gemini"] is stored["Gemini"]


def test_folding_is_a_no_op_for_a_single_omp_row(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    stored = {"OMP": {"litellm_id": "litellm_proxy", "models": ["anthropic/claude-opus-5"], "source": "omp"}}

    assert providers_for_ui(stored) is stored


def test_folding_leaves_a_catalog_without_omp_rows_alone(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    stored = {"Gemini": {"litellm_id": "gemini", "models": ["gemini-3.1-pro"]}}

    assert providers_for_ui(stored) is stored


def test_provider_catalog_rejects_a_blank_source(postgres_store):
    catalog = ModelProviderStore(postgres_store)

    with pytest.raises(ValueError, match="invalid source"):
        catalog.save({"OMP": {"litellm_id": "litellm_proxy", "models": ["anthropic/claude-opus-5"], "source": "  "}})
