"""Shared timestamp helpers."""

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()
