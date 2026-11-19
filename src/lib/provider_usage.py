"""Subscription rate-limit usage of Pi's login, read from pi-host's `/usage`.

- Consumes `GET /usage` on pi-host: `{provider, limits: [{window, label?, usedPercent, resetsAt}], fetchedAt}`, which
  pi-host caches for 60 s (`?refresh=1` bypasses it), or a non-200 `{error: {message}}` when the provider refuses
- Provides `provider_usage(refresh)`, the route's shape: the worst limit per window, with the last good report
  standing in for an hour while the usage endpoint fails (a rate-limited usage API is common)
"""

from __future__ import annotations

from datetime import datetime
import threading
import time
from typing import Any

import httpx

from lib import pi_host

_REQUEST_TIMEOUT_SECONDS = 30.0
_STALE_REPORT_SECONDS = 60 * 60
_WINDOW_ORDER = ("5h", "7d")

# Spans the fetch and the `_last_good` update, so concurrent callers apply answers in fetch order.
_lock = threading.Lock()
_last_good: tuple[float, dict[str, Any]] | None = None


class UsageUnavailable(RuntimeError):
    """pi-host or the provider's usage endpoint did not answer, or answered something unreadable."""


def _error_message(response: httpx.Response) -> str:
    try:
        error = response.json().get("error") or {}
        message = error.get("message") if isinstance(error, dict) else error
    except (ValueError, AttributeError):
        message = None
    return f"usage {response.status_code}: {message or response.text[:200]}"


def _fetch(refresh: bool) -> object:
    """pi-host's `/usage` body. Raises `UsageUnavailable`."""
    try:
        response = httpx.get(
            f"{pi_host.base_url()}/usage",
            params={"refresh": 1} if refresh else None,
            headers={"Authorization": f"Bearer {pi_host.token()}"},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except (pi_host.PiHostUnavailable, httpx.HTTPError) as exc:
        raise UsageUnavailable(f"pi-host does not answer ({exc})") from exc
    if not response.is_success:
        raise UsageUnavailable(_error_message(response))
    try:
        return response.json()
    except ValueError as exc:
        raise UsageUnavailable("pi-host usage returned invalid JSON") from exc


def _minutes_until(resets_at: object, now: float) -> int | None:
    if not isinstance(resets_at, str):
        return None
    try:
        return max(0, round((datetime.fromisoformat(resets_at).timestamp() - now) / 60))
    except ValueError:
        return None


def parse_usage(payload: object, now: float) -> dict[str, Any]:
    """One report for the login: its worst limit per window (several 7d limits exist, one per model family)."""
    if not isinstance(payload, dict) or not payload.get("provider"):
        raise UsageUnavailable("pi-host usage returned an invalid response")
    worst: dict[str, tuple[float, object]] = {}
    for limit in payload.get("limits") or []:
        if not isinstance(limit, dict):
            continue
        window_id, percent = limit.get("window"), limit.get("usedPercent")
        if window_id not in _WINDOW_ORDER or not isinstance(percent, (int, float)):
            continue
        if window_id not in worst or percent > worst[window_id][0]:
            worst[window_id] = (percent, limit.get("resetsAt"))
    report = {
        "provider": payload["provider"],
        "account": None,
        "no_limits": not worst,
        "windows": [
            {"id": window_id, "percent": float(percent), "resets_in_minutes": _minutes_until(resets_at, now)}
            for window_id, (percent, resets_at) in sorted(worst.items(), key=lambda item: _WINDOW_ORDER.index(item[0]))
        ],
    }
    return {"generated_at": payload.get("fetchedAt"), "reports": [report]}


def _aged(usage: dict[str, Any], elapsed_minutes: int) -> dict[str, Any]:
    """*usage* with every window's reset countdown reduced by the minutes since it was read."""
    return {
        **usage,
        "reports": [
            {
                **report,
                "windows": [
                    window if window["resets_in_minutes"] is None
                    else {**window, "resets_in_minutes": max(0, window["resets_in_minutes"] - elapsed_minutes)}
                    for window in report["windows"]
                ],
            }
            for report in usage["reports"]
        ],
    }


def provider_usage(refresh: bool = False) -> dict[str, Any]:
    """Usage per account, the shape the route returns as is.

    `{"generated_at", "reports": [{"provider", "account", "no_limits", "windows": [window]}]}`,
    a window being `{"id", "percent", "resets_in_minutes"}`.
    """
    global _last_good
    with _lock:
        now = time.monotonic()
        try:
            usage = parse_usage(_fetch(refresh), time.time())
        except UsageUnavailable:
            if _last_good is None or now - _last_good[0] > _STALE_REPORT_SECONDS:
                raise
            return _aged(_last_good[1], round((now - _last_good[0]) / 60))
        _last_good = (now, usage)
        return usage
