"""The LearnHarness model generator on a fixture catalog: gateway ids and capabilities, tiers, untouched keys, failures
that keep the last configuration, and concurrency."""

import asyncio
import json
import stat
import threading

import pytest

from lib import learnharness_config, ollama_models, pi_models

PROVIDERS = {
    "Gemini": {"provider_id": "gemini", "models": ["gemini-3.5-flash"]},
    "DeepSeek": {"provider_id": "deepseek", "models": ["deepseek-chat", "deepseek-unserved"]},
    "OpenRouter": {"provider_id": "openrouter", "models": []},
    "Pi": {"provider_id": "litellm_proxy", "models": ["anthropic/claude-sonnet-5-5", "anthropic/claude-3-haiku"], "source": "pi"},
}
OLLAMA_TAGS = ["gemma3:4b"]
GATEWAY_ROWS = {
    "google/gemini-3.5-flash": {
        "id": "google/gemini-3.5-flash",
        "input": ["text", "image"],
        "context_window": 1048576,
        "max_tokens": 65536,
        "thinking_levels": ["minimal", "low", "medium", "high"],
    },
    # Thinking can only be off: no reasoning.
    "deepseek/deepseek-chat": {
        "id": "deepseek/deepseek-chat",
        "input": ["text"],
        "context_window": 131072,
        "max_tokens": 8192,
        "thinking_levels": ["off"],
    },
    "anthropic/claude-sonnet-5-5": {
        "id": "anthropic/claude-sonnet-5-5",
        "input": ["text", "image"],
        "context_window": 200000,
        "max_tokens": 64000,
        "thinking_levels": ["off", "low", "medium", "high"],
    },
    "anthropic/claude-3-haiku": {"id": "anthropic/claude-3-haiku", "input": ["text"], "context_window": 262144, "thinking_levels": []},
    "ollama/gemma3:4b": {"id": "ollama/gemma3:4b", "input": ["text", "image"], "context_window": 131072, "thinking_levels": []},
    "openai/not-in-the-catalog": {"id": "openai/not-in-the-catalog", "input": ["text"]},
}
SERVED_IDS = [
    "google/gemini-3.5-flash",
    "deepseek/deepseek-chat",
    "anthropic/claude-sonnet-5-5",
    "anthropic/claude-3-haiku",
    "ollama/gemma3:4b",
]
# Stored gigachad selectors, as the defaults hold them.
DEFAULTS = {
    "default_model": "litellm_proxy/anthropic/claude-sonnet-5-5",
    "small_model": "gemini/gemini-3.5-flash",
    "vision_model": "openrouter/not-in-the-catalog",
    "sandbox_model": "gemini/gemini-3.5-flash",
}
BASE_URL = "http://127.0.0.1:8300/v1"


@pytest.fixture
def models():
    """The fixture catalog as the gateway describes it."""
    return [learnharness_config.served_model(GATEWAY_ROWS[model_id]) for model_id in SERVED_IDS]


def write(models, tmp_path, defaults=DEFAULTS):
    learnharness_config.write_config(models, defaults, learnharness_dir=tmp_path / "lh", base_url=BASE_URL)
    settings = json.loads((tmp_path / "lh" / "data" / "settings.json").read_text())
    auth = json.loads((tmp_path / "lh" / "auth.json").read_text())
    return settings, auth


def tier_ids(settings):
    return {tier: slot["model"]["id"] for tier, slot in settings["models"].items()}


def test_catalog_selectors_become_gateway_ids_in_the_selectors_order():
    assert learnharness_config.catalog_model_ids(PROVIDERS, OLLAMA_TAGS) == [
        "google/gemini-3.5-flash",
        "deepseek/deepseek-chat",
        "deepseek/deepseek-unserved",
        "anthropic/claude-sonnet-5-5",
        "anthropic/claude-3-haiku",
        "ollama/gemma3:4b",
    ]


def test_learnharness_calls_the_gateway_by_gateway_id_without_a_stored_key(models, tmp_path):
    settings, auth = write(models, tmp_path)

    provider = settings["customProvider"]
    assert [row["id"] for row in provider["models"]] == SERVED_IDS
    assert (provider["baseUrl"], provider["authMode"]) == (BASE_URL, "apiKey")
    binding = json.loads(auth[learnharness_config.CUSTOM_AUTH_BINDING])
    assert binding == {"version": 1, "id": provider["credentialBindingId"], "baseUrl": BASE_URL, "authMode": "apiKey"}
    assert len(binding["id"]) == 32
    # The gateway token reaches LearnHarness by environment only (run-learnharness.sh).
    assert "fileKeyBaseUrl" not in binding
    assert "CUSTOM_OPENAI_API_KEY" not in auth
    for private in (tmp_path / "lh" / "auth.json", tmp_path / "lh" / "data" / "settings.json"):
        assert stat.S_IMODE(private.stat().st_mode) == 0o600
    assert not list(tmp_path.rglob("*.tmp"))


def test_capabilities_come_from_the_gateway_rows(models, tmp_path):
    settings, _ = write(models, tmp_path)

    assert {row["id"]: row for row in settings["customProvider"]["models"]} == {
        "google/gemini-3.5-flash": {
            "id": "google/gemini-3.5-flash",
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
        "anthropic/claude-sonnet-5-5": {
            "id": "anthropic/claude-sonnet-5-5",
            "image": True,
            "reasoning": True,
            "context": 200000,
            "maxTokens": 64000,
        },
        # No max output in the row: assumed, never LearnHarness's 4096.
        "anthropic/claude-3-haiku": {
            "id": "anthropic/claude-3-haiku",
            "image": False,
            "reasoning": False,
            "context": 262144,
            "maxTokens": learnharness_config.ASSUMED_MAX_TOKENS,
        },
        "ollama/gemma3:4b": {
            "id": "ollama/gemma3:4b",
            "image": True,
            "reasoning": False,
            "context": 131072,
            "maxTokens": learnharness_config.ASSUMED_MAX_TOKENS,
        },
    }
    assert [model.id for model in models if model.assumed_limits] == ["anthropic/claude-3-haiku", "ollama/gemma3:4b"]


def test_a_row_without_figures_gets_assumed_limits_not_8k():
    unknown = learnharness_config.served_model({"id": "openrouter/new/model"})
    assert unknown.assumed_limits
    assert unknown.snapshot() == {
        "id": "openrouter/new/model",
        "image": False,
        "reasoning": False,
        "context": learnharness_config.ASSUMED_CONTEXT,
        "maxTokens": learnharness_config.ASSUMED_MAX_TOKENS,
    }
    # A small known window keeps half of it for the answer.
    small = learnharness_config.ServedModel("ollama/tiny", context=8192)
    assert small.snapshot()["maxTokens"] == 4096


def test_first_launch_seeds_every_tier_from_gigachad_defaults_as_gateway_ids(models, tmp_path):
    settings, _ = write(models, tmp_path)

    # The vision default is not served, so vision falls to the first image model.
    assert tier_ids(settings) == {
        "strong": "anthropic/claude-sonnet-5-5",
        "cheap": "google/gemini-3.5-flash",
        "vision": "google/gemini-3.5-flash",
    }
    assert {slot["model"]["provider"] for slot in settings["models"].values()} == {"custom"}
    assert settings["version"] == 1


def test_vision_is_seeded_only_with_an_image_model(models, tmp_path, capsys):
    text_only = [model for model in models if not model.image]
    # gigachad's vision default is served but cannot see: it is not seeded either.
    settings, _ = write(text_only, tmp_path, defaults={**DEFAULTS, "vision_model": "deepseek/deepseek-chat"})

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
                "customProvider": {"baseUrl": "http://127.0.0.1:30183", "overrides": {"deepseek/deepseek-chat": {"context": 64000}}},
                # Only cheap is chosen, with its level: strong and vision are seeded, all three are written.
                "models": {"cheap": {"model": {"provider": "custom", "id": "deepseek/deepseek-chat"}, "thinkingLevel": "off"}},
            }
        )
    )
    auth_path.write_text(json.dumps({"brave": "file-key"}))

    settings, auth = write(models, tmp_path)
    assert tier_ids(settings) == {
        "strong": "anthropic/claude-sonnet-5-5",
        "cheap": "deepseek/deepseek-chat",
        "vision": "google/gemini-3.5-flash",
    }
    assert settings["models"]["cheap"]["thinkingLevel"] == "off"
    assert settings["appearance"] == {"accent": "teal"}
    assert settings["agent"] == {"models": {"researcher": {"model": {"provider": "custom", "id": "deepseek/deepseek-chat"}}}}
    assert settings["customProvider"]["overrides"] == {"deepseek/deepseek-chat": {"context": 64000}}
    assert settings["customProvider"]["baseUrl"] == BASE_URL
    assert auth["brave"] == "file-key"
    first_binding = settings["customProvider"]["credentialBindingId"]

    # The user picks other tiers in LearnHarness: the next launch keeps them and the binding.
    settings["models"]["strong"] = {"model": {"provider": "custom", "id": "ollama/gemma3:4b"}, "thinkingLevel": "high"}
    settings["models"]["vision"] = {"model": {"provider": "custom", "id": "anthropic/claude-sonnet-5-5"}}
    settings_path.write_text(json.dumps(settings))
    again, auth_again = write(models, tmp_path)
    assert again["models"] == settings["models"]
    assert again["customProvider"]["credentialBindingId"] == first_binding
    assert auth_again == auth


def test_a_tier_naming_a_stored_selector_keeps_its_pick_under_the_gateway_id(models, tmp_path, capsys):
    settings_path = tmp_path / "lh" / "data" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "models": {
                    "strong": {
                        "model": {"provider": "custom", "id": "litellm_proxy/anthropic/claude-3-haiku"},
                        "thinkingLevel": "high",
                    },
                    "cheap": {"model": {"provider": "custom", "id": "gemini/gemini-3.5-flash"}, "thinkingLevel": "low"},
                }
            }
        )
    )

    settings, _ = write(models, tmp_path)
    assert settings["models"]["strong"] == {"model": {"provider": "custom", "id": "anthropic/claude-3-haiku"}, "thinkingLevel": "high"}
    assert settings["models"]["cheap"] == {"model": {"provider": "custom", "id": "google/gemini-3.5-flash"}, "thinkingLevel": "low"}
    assert "not served" not in capsys.readouterr().err


def test_a_tier_nothing_serves_is_reseeded_without_its_level(models, tmp_path, capsys):
    settings_path = tmp_path / "lh" / "data" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    chosen = {
        "strong": {"model": {"provider": "custom", "id": "gone/model"}, "thinkingLevel": "xhigh"},
        "cheap": {"model": {"provider": "openai", "id": "deepseek/deepseek-chat"}},
        "vision": {"model": {"provider": "custom", "id": "ollama/gemma3:4b"}, "thinkingLevel": "off"},
    }
    settings_path.write_text(json.dumps({"models": chosen}))

    settings, _ = write(models, tmp_path)
    assert settings["models"] == {
        "strong": {"model": {"provider": "custom", "id": "anthropic/claude-sonnet-5-5"}},
        "cheap": {"model": {"provider": "custom", "id": "google/gemini-3.5-flash"}},
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

    with pytest.raises(learnharness_config.ConfigUnavailable, match="no model"):
        write([], tmp_path)
    assert settings_path.read_text() == last_good
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
    merge = learnharness_config.merged_settings
    calls = []

    def learner_picks_meanwhile(current, **kwargs):
        calls.append(current)
        if len(calls) == 1:
            settings_path.write_text(json.dumps({"appearance": {"accent": "teal"}, "general": {"picked": "while we merged"}}))
        return merge(current, **kwargs)

    monkeypatch.setattr(learnharness_config, "merged_settings", learner_picks_meanwhile)
    settings, _ = write(models, tmp_path)

    assert len(calls) == 2
    assert settings["general"] == {"picked": "while we merged"}
    assert not list(settings_path.parent.glob(".*.tmp"))


def test_concurrent_runs_agree_on_one_binding(models, tmp_path):
    errors = []

    def run():
        try:
            write(models, tmp_path)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    settings, auth = write(models, tmp_path)
    assert json.loads(auth[learnharness_config.CUSTOM_AUTH_BINDING])["id"] == settings["customProvider"]["credentialBindingId"]


# ── Live catalog reads, faked at their sources ───────────────────────────────────────────────────────────────────────


@pytest.fixture
def catalog(monkeypatch):
    """The fixture catalog with Ollama installed and the Pi gateway answering, in call order."""
    calls = []

    def gateway():
        calls.append("gateway")
        return list(GATEWAY_ROWS.values())

    def sync():
        calls.append("sync")
        return False

    monkeypatch.setattr(learnharness_config, "_host_catalog", lambda login: (PROVIDERS, DEFAULTS))
    monkeypatch.setattr(ollama_models, "installed_chat_models", lambda: list(OLLAMA_TAGS))
    monkeypatch.setattr(ollama_models, "sync_pi_provider", sync)
    monkeypatch.setattr(pi_models, "catalog_entries", gateway)
    return calls


def test_the_catalog_is_described_from_one_gateway_read_after_the_ollama_sync(catalog, capsys):
    models, defaults = asyncio.run(learnharness_config.served_catalog("host"))

    assert catalog == ["sync", "gateway"]
    assert [model.id for model in models] == SERVED_IDS
    by_id = {model.id: model for model in models}
    assert by_id["anthropic/claude-sonnet-5-5"].reasoning
    assert by_id["ollama/gemma3:4b"].image
    assert defaults == DEFAULTS
    # Pi knows no such model: LearnHarness never offers it.
    assert "the Pi gateway does not serve deepseek/deepseek-unserved: left out of LearnHarness" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        (pi_models.GatewayUnavailable("no Pi gateway at http://127.0.0.1:8300/v1 (refused)"), "no Pi gateway"),
        ([], "lists no model"),
        ([{"id": "openai/other"}], "serves none of host's models"),
    ],
)
def test_a_pi_gateway_down_empty_or_serving_none_fails_instead_of_degrading(catalog, monkeypatch, answer, message):
    def down():
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(pi_models, "catalog_entries", down)
    with pytest.raises(learnharness_config.ConfigUnavailable, match=message):
        asyncio.run(learnharness_config.served_catalog("host"))


def test_ollama_down_leaves_only_its_models_out(catalog, monkeypatch):
    monkeypatch.setattr(ollama_models, "installed_chat_models", list)
    models, _ = asyncio.run(learnharness_config.served_catalog("host"))
    assert "ollama/gemma3:4b" not in [model.id for model in models]


def test_an_empty_catalog_fails(catalog, monkeypatch):
    monkeypatch.setattr(
        learnharness_config, "_host_catalog", lambda login: ({"OpenRouter": {"provider_id": "openrouter", "models": []}}, DEFAULTS)
    )
    monkeypatch.setattr(ollama_models, "installed_chat_models", list)
    with pytest.raises(learnharness_config.ConfigUnavailable, match="host's catalog has no model"):
        asyncio.run(learnharness_config.served_catalog("host"))


def test_write_config_against_a_dead_gateway_keeps_the_last_good_files(catalog, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GIGACHAD_LEARNHARNESS_HOME", str(tmp_path / "lh"))
    monkeypatch.setenv("GIGACHAD_PI_HOST_URL", "http://127.0.0.1:8300")
    assert learnharness_config.main(["write-config", "--login", "host"]) == 0
    good = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    out = capsys.readouterr().out
    assert "host's catalog: 5 (3 image, 2 reasoning, 2 with assumed limits) via http://127.0.0.1:8300/v1" in out
    settings = json.loads((tmp_path / "lh" / "data" / "settings.json").read_text())
    assert settings["customProvider"]["baseUrl"] == "http://127.0.0.1:8300/v1"

    def refused():
        raise pi_models.GatewayUnavailable("no Pi gateway at http://127.0.0.1:8300/v1 (connection refused)")

    monkeypatch.setattr(pi_models, "catalog_entries", refused)
    assert learnharness_config.main(["write-config", "--login", "host"]) == 1
    assert "no Pi gateway" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == good
