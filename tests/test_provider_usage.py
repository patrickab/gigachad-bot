"""`omp usage --json` becomes one row per account with its worst limit per window."""

import json

import pytest

from lib.provider_usage import UsageUnavailable, parse_usage

NOW_MS = 1_000_000_000_000


def _limit(window_id: str, fraction: float, *, resets_in_ms: int = 600_000, **scope) -> dict:
    return {
        "scope": {"windowId": window_id, **scope},
        "window": {"durationMs": 0, "resetsAt": NOW_MS + resets_in_ms},
        "amount": {"usedFraction": fraction},
    }


def test_each_account_keeps_its_worst_limit_per_window_and_the_reset_in_minutes():
    output = json.dumps({"generatedAt": 5, "reports": [
        {"provider": "anthropic", "metadata": {"email": "me@example.com", "accountId": "uuid"}, "limits": [
            _limit("5h", 0.22),
            _limit("7d", 0.10),
            _limit("7d", 0.24, resets_in_ms=3_600_000, modelId="opus"),
        ]},
    ]})

    [report] = parse_usage(output, NOW_MS)["reports"]

    assert report["account"] == "me@example.com"
    assert report["no_limits"] is False
    assert [(w["id"], round(w["percent"]), w["resets_in_minutes"]) for w in report["windows"]] == [("5h", 22, 10), ("7d", 24, 60)]


def test_windows_are_recognised_by_duration_when_the_scope_does_not_name_them():
    limit = {"scope": {}, "window": {"durationMs": 7 * 86_400_000}, "amount": {"usedFraction": 0.5}}

    [report] = parse_usage(json.dumps({"reports": [{"provider": "p", "limits": [limit]}]}), NOW_MS)["reports"]

    assert [w["id"] for w in report["windows"]] == ["7d"]


def test_an_account_without_limits_is_listed_as_unlimited_without_a_label():
    [report] = parse_usage(json.dumps({"reports": [{"provider": "ollama", "limits": []}]}), NOW_MS)["reports"]

    assert report["no_limits"] is True
    assert report["account"] is None
    assert report["windows"] == []


def test_unreadable_output_is_unavailable_not_empty():
    with pytest.raises(UsageUnavailable):
        parse_usage("not json", NOW_MS)


def test_a_provider_dropped_by_a_failed_fetch_keeps_its_last_report(monkeypatch):
    import lib.provider_usage as usage

    def payload(*providers):
        return json.dumps({"generatedAt": NOW_MS, "reports": [
            {"provider": provider, "limits": [_limit("5h", 0.4, resets_in_ms=120 * 60_000)]} for provider in providers
        ]})

    outputs = iter([payload("openai-codex", "anthropic"), payload("openai-codex")[:-1] + ', "accountsWithoutUsage": [{"provider": "anthropic"}]}'])
    clock = iter([0.0, 30 * 60.0])
    monkeypatch.setattr(usage, "_last_reports", {})
    monkeypatch.setattr(usage, "_output", lambda refresh: next(outputs))
    monkeypatch.setattr(usage.time, "time", lambda: NOW_MS / 1000)
    monkeypatch.setattr(usage.time, "monotonic", lambda: next(clock))

    usage.provider_usage()
    reports = {report["provider"]: report for report in usage.provider_usage()["reports"]}

    assert set(reports) == {"openai-codex", "anthropic"}
    assert reports["anthropic"]["windows"] == [{"id": "5h", "percent": 40.0, "resets_in_minutes": 90}]


def test_an_account_without_usage_is_listed_as_unavailable_until_it_has_a_report(monkeypatch):
    import lib.provider_usage as usage

    output = json.dumps({"generatedAt": NOW_MS, "reports": [], "accountsWithoutUsage": [{"provider": "anthropic", "email": "a@b.c"}]})
    monkeypatch.setattr(usage, "_last_reports", {})
    monkeypatch.setattr(usage, "_output", lambda refresh: output)

    assert usage.provider_usage()["reports"] == [
        {"provider": "anthropic", "account": "a@b.c", "no_limits": False, "unavailable": True, "windows": []},
    ]
