"""Bridges from blocking code to asyncio."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

_END = object()


async def aiter_in_thread[T](iterable: Iterable[T]) -> AsyncIterator[T]:
    """Iterate a blocking iterable, pulling each item in a worker thread."""
    it = iter(iterable)
    while (item := await asyncio.to_thread(next, it, _END)) is not _END:
        yield item
