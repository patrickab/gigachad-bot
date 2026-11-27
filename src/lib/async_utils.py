"""Bridges from blocking code to asyncio."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

_END = object()


async def aiter_in_thread[T](iterable: Iterable[T]) -> AsyncIterator[T]:
    """Iterate a blocking iterable, pulling each item in a worker thread."""
    it = iter(iterable)
    while (item := await asyncio.to_thread(next, it, _END)) is not _END:
        yield item


class KeyedLocks[K]:
    """One `asyncio.Lock` per key, forgotten once nobody holds or waits for it."""

    def __init__(self) -> None:
        self._locks: WeakValueDictionary[K, asyncio.Lock] = WeakValueDictionary()

    def __call__(self, key: K) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = self._locks[key] = asyncio.Lock()
        return lock
