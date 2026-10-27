"""A single value cached for a fixed number of seconds."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class TtlCache[T]:
    """Holds the last computed value until it is *seconds* old. A raising compute stores nothing.

    Holds no lock: concurrent misses compute in parallel and the last one stored wins. Callers that
    must serialize the computation wrap it in their own lock.
    """

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._entry: tuple[float, T] | None = None

    def get(self, compute: Callable[[], T], *, refresh: bool = False) -> T:
        """The cached value, or a fresh one from *compute* when it is stale, missing or *refresh* is set."""
        entry = self._entry
        if not refresh and entry is not None and time.monotonic() - entry[0] < self._seconds:
            return entry[1]
        value = compute()
        self._entry = (time.monotonic(), value)
        return value

    def clear(self) -> None:
        self._entry = None
