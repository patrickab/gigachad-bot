"""pi-host's `/usage` becomes one row per login with its worst limit per window, the last good one outliving failures."""

from datetime import UTC, datetime

import httpx
import pytest

import lib.provider_usage as usage
from lib.provider_usage import UsageUnavailable, parse_usage

NOW = 1_000_000_000.0


def _iso(seconds_from_now: float) -> str:
    return datetime.fromtimestamp(NOW + seconds_from_now, UTC).isoformat().replace("+00:00", "Z")


def _payload(*limits: dict) -> dict:
    return {"provider": "anthropic", "limits": list(limits), "fetchedAt": _iso(0)}


def test_the_login_keeps_its_worst_limit_per_window_and_the_reset_in_minutes():
    payload = _payload(
        {"window": "5h", "usedPercent": 22, "resetsAt": _iso(600)},
        {"window": "7d", "usedPercent": 10, "resetsAt": _iso(600)},
        {"window": "7d", "label": "Opus", "usedPercent": 24, "resetsAt": _iso(3600)},
    )

    parsed = parse_usage(payload, NOW)

    assert parsed["generated_at"] == _iso(0)
    [report] = parsed["reports"]
    assert (report["provider"], report["account"], report["no_limits"]) == ("anthropic", None, False)
    assert [(w["id"], round(w["percent"]), w["resets_in_minutes"]) for w in report["windows"]] == [("5h", 22, 10), ("7d", 24, 60)]


def test_a_login_without_limits_is_listed_as_unlimited_and_an_unknown_reset_stays_unknown():
    assert parse_usage(_payload(), NOW)["reports"][0]["no_limits"] is True
    [window] = parse_usage(_payload({"window": "5h", "usedPercent": 5, "resetsAt": None}), NOW)["reports"][0]["windows"]
    assert window["resets_in_minutes"] is None


def test_an_unreadable_answer_is_unavailable_not_empty():
    with pytest.raises(UsageUnavailable):
        parse_usage(["not", "a", "report"], NOW)


@pytest.fixture
def pi_host_usage(monkeypatch, tmp_path):
    """pi-host answering `/usage` from a queue of responses, with the clocks pinned."""
    (tmp_path / "pi-host.token").write_text("host-token\n", encoding="utf-8")
    monkeypatch.setenv("PI_HOST_TOKEN_FILE", str(tmp_path / "pi-host.token"))
    monkeypatch.setenv("GIGACHAD_PI_HOST_URL", "http://127.0.0.1:8399")
    monkeypatch.setattr(usage, "_last_good", None)
    monkeypatch.setattr(usage.time, "time", lambda: NOW)
    answers: list[httpx.Response] = []
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return answers.pop(0)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(usage.httpx, "get", lambda url, **kwargs: httpx.Client(transport=transport).get(url, **kwargs))
    return answers, requests


def test_refresh_bypasses_pi_hosts_cache_with_the_bearer(pi_host_usage):
    answers, requests = pi_host_usage
    answers.append(httpx.Response(200, json=_payload()))

    usage.provider_usage(refresh=True)

    assert str(requests[0].url) == "http://127.0.0.1:8399/usage?refresh=1"
    assert requests[0].headers["Authorization"] == "Bearer host-token"


def test_a_failed_fetch_serves_the_last_report_aged(pi_host_usage, monkeypatch):
    answers, _ = pi_host_usage
    answers += [
        httpx.Response(200, json=_payload({"window": "5h", "usedPercent": 40, "resetsAt": _iso(120 * 60)})),
        httpx.Response(429, json={"error": {"message": "rate limited", "status": 429}}),
        httpx.Response(429, json={"error": {"message": "rate limited", "status": 429}}),
    ]
    clock = iter([0.0, 30 * 60.0, 2 * 3600.0])
    monkeypatch.setattr(usage.time, "monotonic", lambda: next(clock))

    usage.provider_usage()
    [report] = usage.provider_usage()["reports"]

    assert report["windows"] == [{"id": "5h", "percent": 40.0, "resets_in_minutes": 90}]
    with pytest.raises(UsageUnavailable, match="rate limited"):
        usage.provider_usage()


def test_pi_host_down_without_a_last_report_is_unavailable(pi_host_usage, monkeypatch):
    monkeypatch.setenv("PI_HOST_TOKEN_FILE", "/nonexistent/pi-host.token")
    with pytest.raises(UsageUnavailable, match="pi-host"):
        usage.provider_usage()
