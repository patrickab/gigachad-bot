"""Retry stale LiteLLM capability rejections against the provider once."""

from __future__ import annotations

import re
from typing import Any

from litellm.exceptions import UnsupportedParamsError

_UNSUPPORTED_PARAMS_RE = re.compile(r"does not support parameters:\s*\[(.*?)\]")


def _parse_unsupported_params(message: str) -> set[str]:
    match = _UNSUPPORTED_PARAMS_RE.search(message)
    if not match:
        return set()
    return {p.strip().strip("'\"") for p in match.group(1).split(",") if p.strip()}


def api_query_resilient(client: Any, **kwargs: Any) -> Any:
    """Call ``api_query``, retrying once with parameters rejected by stale LiteLLM metadata."""
    try:
        return client.api_query(**kwargs)
    except UnsupportedParamsError as e:
        unsupported = _parse_unsupported_params(str(e))
        if not unsupported:
            raise
        already_allowed = set(kwargs.get("allowed_openai_params") or [])
        if unsupported <= already_allowed:
            raise  # already retried this exact set, provider genuinely rejects it
        retry_kwargs = {**kwargs, "allowed_openai_params": sorted(already_allowed | unsupported)}
        return client.api_query(**retry_kwargs)
