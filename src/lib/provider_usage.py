"""Subscription rate-limit usage per provider account, read from `omp usage --json`."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from typing import Any

from lib.omp_cli import run_omp

_CLI_TIMEOUT_SECONDS = 30
_CACHE_SECONDS = 5 * 60
# A provider whose usage fetch fails drops out of `omp usage`; its last report stands in this long.
_STALE_REPORT_SECONDS = 60 * 60
_WINDOW_DURATIONS_MS = {"5h": 5 * 3_600_000, "7d": 7 * 86_400_000, "30d": 30 * 86_400_000}
# omp's own window ids, folded onto the three the UI shows.
_WINDOW_ALIASES = {"monthly": "30d", "daily": "5h"}

_lock = threading.RLock()
_cache: tuple[float, str] | None = None
_last_reports: dict[str, tuple[float, dict[str, Any]]] = {}


class UsageUnavailable(RuntimeError):
    """The omp CLI is missing, failed, or printed something unreadable."""


def _run_omp(*arguments: str) -> str:
    try:
        return run_omp(*arguments, timeout=_CLI_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError) as exc:
        raise UsageUnavailable(str(exc)) from exc


def _output(refresh: bool) -> str:
    """Raw CLI output, cached for a few minutes; *refresh* makes omp refetch from the providers first."""
    global _cache
    with _lock:
        if not refresh and _cache is not None and time.monotonic() - _cache[0] < _CACHE_SECONDS:
            return _cache[1]
        if refresh:
            _run_omp("usage", "invalidate")
        output = _run_omp("usage", "--json")
        _cache = (time.monotonic(), output)
        return output


def _window_id(limit: dict[str, Any]) -> str | None:
    scope, window = limit.get("scope") or {}, limit.get("window") or {}
    window_id = _WINDOW_ALIASES.get(scope.get("windowId"), scope.get("windowId"))
    if window_id in _WINDOW_DURATIONS_MS:
        return window_id
    duration = window.get("durationMs")
    if isinstance(duration, (int, float)):
        return next((name for name, expected in _WINDOW_DURATIONS_MS.items() if abs(duration - expected) <= 3_600_000), None)
    return None


def _summarize(report: dict[str, Any], now_ms: float) -> dict[str, Any]:
    """One account: its worst limit per window, or `no_limits` when omp reports none."""
    worst: dict[str, tuple[float, float | None]] = {}
    for limit in report.get("limits") or []:
        fraction = (limit.get("amount") or {}).get("usedFraction")
        window_id = _window_id(limit)
        if window_id is None or not isinstance(fraction, (int, float)):
            continue
        if window_id not in worst or fraction > worst[window_id][0]:
            worst[window_id] = (fraction, (limit.get("window") or {}).get("resetsAt"))
    metadata = report.get("metadata") or {}
    return {
        "provider": report.get("provider"),
        # Email only: the account id is an opaque UUID.
        "account": metadata.get("email") or None,
        "no_limits": not worst,
        "windows": [
            {
                "id": window_id,
                "percent": fraction * 100,
                "resets_in_minutes": max(0, round((resets_at - now_ms) / 60_000)) if isinstance(resets_at, (int, float)) else None,
            }
            for window_id, (fraction, resets_at) in sorted(worst.items(), key=lambda item: list(_WINDOW_DURATIONS_MS).index(item[0]))
        ],
    }


def parse_usage(output: str, now_ms: float) -> dict[str, Any]:
    try:
        payload = json.loads(output)
    except ValueError as exc:
        raise UsageUnavailable("omp usage returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise UsageUnavailable("omp usage returned an invalid response")
    reports = [_summarize(report, now_ms) for report in payload.get("reports") or [] if isinstance(report, dict) and report.get("provider")]
    # Accounts whose usage fetch failed this time, e.g. a rate-limited provider usage API.
    reports += [
        {"provider": account["provider"], "account": account.get("email") or None, "no_limits": False, "unavailable": True, "windows": []}
        for account in payload.get("accountsWithoutUsage") or []
        if isinstance(account, dict) and account.get("provider")
    ]
    return {"generated_at": payload.get("generatedAt"), "reports": reports}


def _aged_report(report: dict[str, Any], elapsed_minutes: int) -> dict[str, Any]:
    """*report* with every window's reset countdown reduced by the minutes since it was read."""
    windows = [
        {**window, "resets_in_minutes": max(0, window["resets_in_minutes"] - elapsed_minutes)} if window["resets_in_minutes"] is not None else window
        for window in report["windows"]
    ]
    return {**report, "windows": windows}


def _with_dropped_providers(usage: dict[str, Any], now: float) -> dict[str, Any]:
    """Replace providers this fetch could not read with their last report, so a failed fetch does not hide them."""
    fresh = [report for report in usage["reports"] if not report.get("unavailable")]
    present = {report["provider"] for report in fresh}
    for report in fresh:
        _last_reports[report["provider"]] = (now, report)
    reports = list(fresh)
    for provider, (seen_at, report) in list(_last_reports.items()):
        if provider in present:
            continue
        if now - seen_at > _STALE_REPORT_SECONDS:
            del _last_reports[provider]
            continue
        reports.append(_aged_report(report, round((now - seen_at) / 60)))
    known = {report["provider"] for report in reports}
    reports += [report for report in usage["reports"] if report.get("unavailable") and report["provider"] not in known]
    return {**usage, "reports": reports}


def provider_usage(refresh: bool = False) -> dict[str, Any]:
    """`{"generated_at", "reports": [{"provider", "account", "no_limits", "unavailable"?, "windows": [{"id", "percent", "resets_in_minutes"}]}]}`."""
    with _lock:
        return _with_dropped_providers(parse_usage(_output(refresh), time.time() * 1000), time.monotonic())
