"""The LearnHarness model generator on a fixture catalog: proxy and snapshot agree, capabilities, tiers, untouched keys,
failures that keep the last configuration, concurrency, and the reused proxy check."""

import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import stat
import threading
import urllib.error

import litellm
import pytest
import yaml

from lib import learnharness, omp_source, reasoning_levels

PROVIDERS = {
    "Gemini": {"litellm_id": "gemini", "models": ["gemini-3.5-flash"]},
    "DeepSeek": {"litellm_id": "deepseek", "models": ["deepseek-chat"]},
    "OpenRouter": {"litellm_id": "openrouter", "models": []},
    "OMP": {"litellm_id": "litellm_proxy", "models": ["anthropic/claude-sonnet-5-5", "ollama-cloud/devstral-2:123b"], "source": "omp"},
}
OLLAMA = ["ollama/gemma3:4b"]
OMP_ENTRIES = {
    "anthropic/claude-sonnet-5-5": {
        "id": "anthropic/claude-sonnet-5-5",
        "input_modalities": ["text", "image"],
        "context_length": 200000,
        "max_output_tokens": 64000,
    },
    "ollama-cloud/devstral-2:123b": {"id": "ollama-cloud/devstral-2:123b", "input_modalities": ["text"], "context_length": 262144},
}
OLLAMA_DETAILS = {
    "gemma3:4b": {
        "capabilities": ["completion", "vision"],
        "model_info": {"general.architecture": "gemma3", "gemma3.context_length": 131072},
    },
}
REGISTRY = {
    "gemini/gemini-3.5-flash": {"max_input_tokens": 1048576, "max_output_tokens": 65536, "vision": True},
    "deepseek/deepseek-chat": {"max_input_tokens": 131072, "max_tokens": 8192, "vision": False},
}
REASONING = {"gemini/gemini-3.5-flash", "litellm_proxy/anthropic/claude-sonnet-5-5"}
DEFAULTS = {
    "default_model": "litellm_proxy/anthropic/claude-sonnet-5-5",
    "small_model": "gemini/gemini-3.5-flash",
    "vision_model": "openrouter/not-in-the-catalog",
    "memory_model": "gemini/gemini-3.5-flash",
    "omp_model": "gemini/gemini-3.5-flash",
}
BASE_URL = "http://127.0.0.1:30183"


@pytest.fixture
def models(monkeypatch):
    """The fixture catalog served, with LiteLLM's registry and the OMP gateway faked."""

    def model_info(model):
        if model not in REGISTRY:
            raise ValueError(f"unknown {model}")
        return {key: value for key, value in REGISTRY[model].items() if key != "vision"}

    monkeypatch.setattr(litellm, "get_model_info", model_info)
    monkeypatch.setattr(litellm, "supports_vision", lambda model: REGISTRY.get(model, {}).get("vision", False))
    monkeypatch.setenv("GIGACHAD_OMP_GATEWAY_URL", "http://127.0.0.1:4000/v1")
    monkeypatch.setenv("GIGACHAD_OMP_GATEWAY_TOKEN", "gateway-token")
    return [
        learnharness.served_model(selector, reasoning=selector in REASONING, omp_entries=OMP_ENTRIES, ollama_details=OLLAMA_DETAILS)
        for selector in learnharness.catalog_selectors(PROVIDERS, OLLAMA)
    ]


def write(models, tmp_path, defaults=DEFAULTS):
    learnharness.write_config(models, defaults, proxy_dir=tmp_path / "proxy", learnharness_dir=tmp_path / "lh", base_url=BASE_URL)
    config = yaml.safe_load((tmp_path / "proxy" / "config.yaml").read_text())
    settings = json.loads((tmp_path / "lh" / "data" / "settings.json").read_text())
    auth = json.loads((tmp_path / "lh" / "auth.json").read_text())
    return config, settings, auth


def tier_ids(settings):
    return {tier: slot["model"]["id"] for tier, slot in settings["models"].items()}


def test_proxy_and_learnharness_serve_the_same_selector_strings(models, tmp_path):
    config, settings, auth = write(models, tmp_path)

    expected = [
        "gemini/gemini-3.5-flash",
        "deepseek/deepseek-chat",
        "litellm_proxy/anthropic/claude-sonnet-5-5",
        "litellm_proxy/ollama-cloud/devstral-2:123b",
        "ollama/gemma3:4b",
    ]
    assert [route["model_name"] for route in config["model_list"]] == expected
    assert [row["id"] for row in settings["customProvider"]["models"]] == expected
    routes = {route["model_name"]: route["litellm_params"] for route in config["model_list"]}
    # Secrets are named, the proxy launcher exports them: config.yaml holds none.
    assert routes["litellm_proxy/anthropic/claude-sonnet-5-5"] == {
        "model": "litellm_proxy/anthropic/claude-sonnet-5-5",
        "api_base": "http://127.0.0.1:4000/v1",
        "api_key": "os.environ/GIGACHAD_OMP_GATEWAY_TOKEN",
    }
    assert routes["ollama/gemma3:4b"] == {"model": "ollama/gemma3:4b", "api_base": learnharness.OLLAMA_BASE_URL}
    # Provider keys stay in the proxy's environment, where LiteLLM looks them up like gigachad's own calls.
    assert routes["gemini/gemini-3.5-flash"] == {"model": "gemini/gemini-3.5-flash"}
    assert config["litellm_settings"] == {"drop_params": True}
    assert config["general_settings"] == {"master_key": "os.environ/LITELLM_MASTER_KEY"}

    key = (tmp_path / "proxy" / "master.key").read_text().strip()
    assert key.startswith("sk-")
    assert key == learnharness.master_key(tmp_path / "proxy")
    config_text = (tmp_path / "proxy" / "config.yaml").read_text()
    assert key not in config_text
    assert "gateway-token" not in config_text
    for private in (tmp_path / "proxy" / "master.key", tmp_path / "proxy" / "config.yaml", tmp_path / "lh" / "auth.json"):
        assert stat.S_IMODE(private.stat().st_mode) == 0o600

    provider = settings["customProvider"]
    assert (provider["baseUrl"], provider["authMode"]) == (BASE_URL, "apiKey")
    binding = json.loads(auth[learnharness.CUSTOM_AUTH_BINDING])
    assert binding == {"version": 1, "id": provider["credentialBindingId"], "baseUrl": BASE_URL, "authMode": "apiKey"}
    assert len(binding["id"]) == 32
    assert key not in (tmp_path / "lh" / "auth.json").read_text() + (tmp_path / "lh" / "data" / "settings.json").read_text()
    # Nothing is left staged next to the files.
    assert not list(tmp_path.rglob("*.tmp"))


def test_capabilities_come_from_the_omp_gateway_ollama_and_litellm(models, tmp_path):
    _, settings, _ = write(models, tmp_path)

    assert {row["id"]: row for row in settings["customProvider"]["models"]} == {
        "gemini/gemini-3.5-flash": {
            "id": "gemini/gemini-3.5-flash",
            "image": True,
            "reasoning": True,
            "context": 1048576,
            "maxTokens": 65536,
        },
        "deepseek/deepseek-chat": {
            "id": "deepseek/deepseek-chat",
            "image": False,
            "reasoning": False,
            "context": 131072,
            "maxTokens": 8192,
        },
        "litellm_proxy/anthropic/claude-sonnet-5-5": {
            "id": "litellm_proxy/anthropic/claude-sonnet-5-5",
            "image": True,
            "reasoning": True,
            "context": 200000,
            "maxTokens": 64000,
        },
        # The gateway omits its max output, Ollama never reports one: assumed, never LearnHarness's 4096.
        "litellm_proxy/ollama-cloud/devstral-2:123b": {
            "id": "litellm_proxy/ollama-cloud/devstral-2:123b",
            "image": False,
            "reasoning": False,
            "context": 262144,
            "maxTokens": learnharness.ASSUMED_MAX_TOKENS,
        },
        "ollama/gemma3:4b": {
            "id": "ollama/gemma3:4b",
            "image": True,
            "reasoning": False,
            "context": 131072,
            "maxTokens": learnharness.ASSUMED_MAX_TOKENS,
        },
    }
    assert [model.id for model in models if model.assumed_limits] == ["litellm_proxy/ollama-cloud/devstral-2:123b", "ollama/gemma3:4b"]


def test_a_model_no_source_describes_gets_assumed_limits_not_8k(models):
    unknown = learnharness.served_model("openrouter/new/model", reasoning=False, omp_entries={}, ollama_details={})
    assert unknown.assumed_limits
    assert unknown.snapshot() == {
        "id": "openrouter/new/model",
        "image": False,
        "reasoning": False,
        "context": learnharness.ASSUMED_CONTEXT,
        "maxTokens": learnharness.ASSUMED_MAX_TOKENS,
    }
    # A small known window keeps half of it for the answer.
    small = learnharness.ServedModel("ollama/tiny", {"model": "ollama/tiny"}, context=8192)
    assert small.snapshot()["maxTokens"] == 4096


def test_first_launch_seeds_every_tier_from_gigachad_defaults(models, tmp_path):
    _, settings, _ = write(models, tmp_path)

    # The vision default is not served, so vision falls to the first image model.
    assert tier_ids(settings) == {
        "strong": "litellm_proxy/anthropic/claude-sonnet-5-5",
        "cheap": "gemini/gemini-3.5-flash",
        "vision": "gemini/gemini-3.5-flash",
    }
    assert {slot["model"]["provider"] for slot in settings["models"].values()} == {"custom"}
    assert settings["version"] == 1


def test_vision_is_seeded_only_with_an_image_model(models, tmp_path, capsys):
    text_only = [model for model in models if not model.image]
    # gigachad's vision default is served but cannot see: it is not seeded either.
    _, settings, _ = write(text_only, tmp_path, defaults={**DEFAULTS, "vision_model": "deepseek/deepseek-chat"})

    assert tier_ids(settings) == {"strong": "deepseek/deepseek-chat", "cheap": "deepseek/deepseek-chat"}
    assert "no image model in the catalog" in capsys.readouterr().err


def test_rewrites_keep_chosen_tiers_pins_and_every_other_key(models, tmp_path):
    settings_path = tmp_path / "lh" / "data" / "settings.json"
    auth_path = tmp_path / "lh" / "auth.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "version": 1,
                "appearance": {"accent": "teal"},
                "agent": {"models": {"researcher": {"model": {"provider": "custom", "id": "deepseek/deepseek-chat"}}}},
                "customProvider": {"baseUrl": "http://127.0.0.1:4000/v1", "overrides": {"deepseek/deepseek-chat": {"context": 64000}}},
                # Only cheap is chosen, with its level: strong and vision are seeded, all three are written.
                "models": {"cheap": {"model": {"provider": "custom", "id": "deepseek/deepseek-chat"}, "thinkingLevel": "off"}},
            }
        )
    )
    auth_path.write_text(json.dumps({"brave": "file-key"}))

    _, settings, auth = write(models, tmp_path)
    assert tier_ids(settings) == {
        "strong": "litellm_proxy/anthropic/claude-sonnet-5-5",
        "cheap": "deepseek/deepseek-chat",
        "vision": "gemini/gemini-3.5-flash",
    }
    assert settings["models"]["cheap"]["thinkingLevel"] == "off"
    assert settings["appearance"] == {"accent": "teal"}
    assert settings["agent"] == {"models": {"researcher": {"model": {"provider": "custom", "id": "deepseek/deepseek-chat"}}}}
    assert settings["customProvider"]["overrides"] == {"deepseek/deepseek-chat": {"context": 64000}}
    assert settings["customProvider"]["baseUrl"] == BASE_URL
    assert auth["brave"] == "file-key"
    first_binding = settings["customProvider"]["credentialBindingId"]
    first_key = (tmp_path / "proxy" / "master.key").read_text()

    # The user picks other tiers in LearnHarness: the next launch keeps them, the binding and the master key.
    settings["models"]["strong"] = {"model": {"provider": "custom", "id": "ollama/gemma3:4b"}, "thinkingLevel": "high"}
    settings["models"]["vision"] = {"model": {"provider": "custom", "id": "litellm_proxy/anthropic/claude-sonnet-5-5"}}
    settings_path.write_text(json.dumps(settings))
    _, again, auth_again = write(models, tmp_path)
    assert again["models"] == settings["models"]
    assert again["customProvider"]["credentialBindingId"] == first_binding
    assert auth_again == auth
    assert (tmp_path / "proxy" / "master.key").read_text() == first_key


def test_a_tier_nothing_serves_is_reseeded_without_its_level(models, tmp_path, capsys):
    settings_path = tmp_path / "lh" / "data" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    chosen = {
        "strong": {"model": {"provider": "custom", "id": "gone/model"}, "thinkingLevel": "xhigh"},
        "cheap": {"model": {"provider": "openai", "id": "deepseek/deepseek-chat"}},
        "vision": {"model": {"provider": "custom", "id": "ollama/gemma3:4b"}, "thinkingLevel": "off"},
    }
    settings_path.write_text(json.dumps({"models": chosen}))

    _, settings, _ = write(models, tmp_path)
    assert settings["models"] == {
        "strong": {"model": {"provider": "custom", "id": "litellm_proxy/anthropic/claude-sonnet-5-5"}},
        "cheap": {"model": {"provider": "custom", "id": "gemini/gemini-3.5-flash"}},
        "vision": chosen["vision"],
    }
    assert "strong tier gone/model is not served" in capsys.readouterr().err


def test_an_empty_catalog_writes_nothing(tmp_path):
    settings_path = tmp_path / "lh" / "data" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    last_good = json.dumps(
        {"models": {"strong": {"model": {"provider": "custom", "id": "kept"}}}, "customProvider": {"models": [{"id": "kept"}]}}
    )
    settings_path.write_text(last_good)
    (tmp_path / "proxy").mkdir()
    (tmp_path / "proxy" / "config.yaml").write_text("model_list: [kept]\n")

    with pytest.raises(learnharness.ConfigUnavailable, match="no model"):
        write([], tmp_path)
    assert settings_path.read_text() == last_good
    assert (tmp_path / "proxy" / "config.yaml").read_text() == "model_list: [kept]\n"
    assert not (tmp_path / "lh" / "auth.json").exists()


def test_a_settings_file_that_is_not_an_object_is_never_overwritten(models, tmp_path):
    settings_path = tmp_path / "lh" / "data" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("[]")

    with pytest.raises(ValueError, match="not a JSON object"):
        write(models, tmp_path)
    assert settings_path.read_text() == "[]"


def test_a_settings_write_by_learnharness_meanwhile_is_merged_not_lost(models, tmp_path, monkeypatch):
    settings_path = tmp_path / "lh" / "data" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(json.dumps({"appearance": {"accent": "teal"}}))
    merge = learnharness.merged_settings
    calls = []

    def learner_picks_meanwhile(current, **kwargs):
        calls.append(current)
        if len(calls) == 1:
            settings_path.write_text(json.dumps({"appearance": {"accent": "teal"}, "general": {"picked": "while we merged"}}))
        return merge(current, **kwargs)

    monkeypatch.setattr(learnharness, "merged_settings", learner_picks_meanwhile)
    _, settings, _ = write(models, tmp_path)

    assert len(calls) == 2
    assert settings["general"] == {"picked": "while we merged"}
    assert not list(settings_path.parent.glob(".*.tmp"))


def test_a_failed_rename_leaves_no_staging_file(tmp_path, monkeypatch):
    def failing_replace(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(learnharness.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        learnharness._write_private(tmp_path / "settings.json", "{}")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("stored", ["", "sk-", "sk-12ab\n"])
def test_an_empty_or_cut_master_key_is_replaced(tmp_path, stored):
    (tmp_path / "master.key").write_text(stored)
    key = learnharness.master_key(tmp_path)
    assert len(key) == 67
    assert key.startswith("sk-")
    assert (tmp_path / "master.key").read_text() == f"{key}\n"
    assert learnharness.master_key(tmp_path) == key


def test_concurrent_runs_agree_on_one_master_key_and_one_binding(models, tmp_path):
    keys = []
    errors = []

    def run():
        try:
            keys.append(learnharness.master_key(tmp_path / "proxy"))
            write(models, tmp_path)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(set(keys)) == 1
    assert (tmp_path / "proxy" / "master.key").read_text() == f"{keys[0]}\n"
    _, settings, auth = write(models, tmp_path)
    assert json.loads(auth[learnharness.CUSTOM_AUTH_BINDING])["id"] == settings["customProvider"]["credentialBindingId"]


# ── Live catalog reads, faked at their sources ───────────────────────────────────────────────────────────────────────


@pytest.fixture
def catalog(monkeypatch):
    """The fixture catalog without Ollama, its OMP sources answering, counted."""
    calls = {"gateway": 0, "registry": 0}

    def gateway():
        calls["gateway"] += 1
        return list(OMP_ENTRIES.values())

    def registry():
        calls["registry"] += 1
        return {"anthropic/claude-sonnet-5-5": ["low", "medium", "high"], "ollama-cloud/devstral-2:123b": []}

    async def resolve(model):
        return ["none", "high"] if model in REASONING else []

    monkeypatch.setattr(learnharness, "_host_catalog", lambda login: (PROVIDERS, DEFAULTS))
    monkeypatch.setattr(learnharness, "discover_ollama_models", list)
    monkeypatch.setattr(omp_source, "catalog_entries", gateway)
    monkeypatch.setattr(omp_source, "thinking_registry", registry)
    monkeypatch.setattr(reasoning_levels, "resolve", resolve)
    monkeypatch.setattr(litellm, "get_model_info", lambda model: {"max_input_tokens": 100000, "max_output_tokens": 8000})
    monkeypatch.setattr(litellm, "supports_vision", lambda model: False)
    return calls


def test_omp_models_are_described_from_one_gateway_and_one_registry_read(catalog):
    models, defaults = asyncio.run(learnharness.served_catalog("host"))

    assert catalog == {"gateway": 1, "registry": 1}
    by_id = {model.id: model for model in models}
    assert by_id["litellm_proxy/anthropic/claude-sonnet-5-5"].reasoning
    assert by_id["litellm_proxy/anthropic/claude-sonnet-5-5"].image
    assert not by_id["litellm_proxy/ollama-cloud/devstral-2:123b"].reasoning
    assert defaults == DEFAULTS


@pytest.mark.parametrize(
    ("source", "failure", "message"),
    [
        ("catalog_entries", urllib.error.URLError("refused"), "no OMP gateway"),
        ("thinking_registry", OSError("omp timed out"), "OMP's model registry"),
    ],
)
def test_an_omp_source_down_fails_instead_of_degrading(catalog, monkeypatch, source, failure, message):
    def down():
        raise failure

    monkeypatch.setattr(omp_source, source, down)
    with pytest.raises(learnharness.ConfigUnavailable, match=message):
        asyncio.run(learnharness.served_catalog("host"))


def test_an_unanswered_reasoning_lookup_fails_instead_of_counting_as_no(catalog, monkeypatch):
    async def unknown(model):
        return None

    monkeypatch.setattr(reasoning_levels, "resolve", unknown)
    with pytest.raises(learnharness.ConfigUnavailable, match=r"no reasoning answer for gemini/gemini-3\.5-flash"):
        asyncio.run(learnharness.served_catalog("host"))


def test_an_omp_model_the_gateway_does_not_list_is_warned_about(catalog, monkeypatch, capsys):
    monkeypatch.setattr(omp_source, "catalog_entries", lambda: [OMP_ENTRIES["anthropic/claude-sonnet-5-5"]])
    asyncio.run(learnharness.served_catalog("host"))
    err = capsys.readouterr().err
    assert "the OMP gateway does not list ollama-cloud/devstral-2:123b" in err
    assert "no context or max output figure for litellm_proxy/ollama-cloud/devstral-2:123b" in err


def test_an_empty_catalog_fails(catalog, monkeypatch):
    monkeypatch.setattr(
        learnharness, "_host_catalog", lambda login: ({"OpenRouter": {"litellm_id": "openrouter", "models": []}}, DEFAULTS)
    )
    with pytest.raises(learnharness.ConfigUnavailable, match="host's catalog has no model"):
        asyncio.run(learnharness.served_catalog("host"))


class _NoRows:
    """A pool whose users table has no row for any login, and that records what it ran."""

    def __init__(self):
        self.statements = []

    def connection(self):
        pool = self

        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, statement, params):
                pool.statements.append(statement)
                return self

            def fetchone(self):
                return None

        return Connection()


def test_an_unknown_login_fails_without_registering_a_user():
    pool = _NoRows()
    with pytest.raises(learnharness.ConfigUnavailable, match="no gigachad user 'typo'"):
        learnharness._user_id(pool, "typo")
    assert [statement.split()[0] for statement in pool.statements] == ["SELECT"]


def test_write_config_against_a_dead_gateway_keeps_the_last_good_files(catalog, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GIGACHAD_LITELLM_PROXY_HOME", str(tmp_path / "proxy"))
    monkeypatch.setenv("GIGACHAD_LEARNHARNESS_HOME", str(tmp_path / "lh"))
    monkeypatch.setenv("GIGACHAD_LITELLM_PROXY_BIND", BASE_URL.removeprefix("http://"))
    assert learnharness.main(["write-config", "--login", "host"]) == 0
    good = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert "host's catalog: 4 (1 image, 2 reasoning, 1 with assumed limits)" in capsys.readouterr().out

    def refused():
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(omp_source, "catalog_entries", refused)
    assert learnharness.main(["write-config", "--login", "host"]) == 1
    assert "no OMP gateway" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == good


# ── The reused proxy ─────────────────────────────────────────────────────────────────────────────────────────────────


class _Proxy(BaseHTTPRequestHandler):
    def do_GET(self):
        authorized = self.headers.get("Authorization") == f"Bearer {self.server.key}"
        body = {"data": [{"id": model} for model in self.server.served]} if authorized else {"error": "Invalid key"}
        payload = json.dumps(body).encode()
        self.send_response(200 if authorized else 400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def proxy():
    server = HTTPServer(("127.0.0.1", 0), _Proxy)
    server.key = "sk-right"
    server.served = ["a/one", "b/two"]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


def test_a_reused_proxy_is_checked_against_this_homes_config_and_key(proxy, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(learnharness.proxy_config([learnharness.ServedModel(m, {"model": m}) for m in ["a/one", "b/two"]]))
    )
    url = f"http://127.0.0.1:{proxy.server_port}"

    assert learnharness.proxy_drift(url, "sk-right", config) is None
    assert "refuses the master key" in learnharness.proxy_drift(url, "sk-other", config)
    proxy.served = ["a/one"]
    assert "missing: b/two, extra: none" in learnharness.proxy_drift(url, "sk-right", config)
    assert "there is no" in learnharness.proxy_drift(url, "sk-right", tmp_path / "absent.yaml")


def test_the_proxy_tool_pin_follows_the_locked_litellm(tmp_path, monkeypatch):
    lock = tmp_path / "uv.lock"
    lock.write_text(
        """version = 1

[[package]]
name = "litellm"
version = "1.98.0"
sdist = { url = "x", upload-time = "2026-08-22T22:19:21.931Z" }
wheels = [{ url = "y", upload-time = "2026-08-22T22:20:00Z" }]
"""
    )
    monkeypatch.setattr(learnharness.importlib.metadata, "version", lambda name: "1.98.0")
    requirement, python, exclude_newer = learnharness.proxy_tool_pin(lock).split()
    assert requirement == "litellm[proxy]==1.98.0"
    assert python.count(".") == 1
    assert exclude_newer == "2026-08-23T22:20:00Z"

    monkeypatch.setattr(learnharness.importlib.metadata, "version", lambda name: "1.99.0")
    with pytest.raises(learnharness.ConfigUnavailable, match=r"does not lock litellm 1\.99\.0"):
        learnharness.proxy_tool_pin(lock)
