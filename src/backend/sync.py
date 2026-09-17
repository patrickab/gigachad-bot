"""Fan PostgreSQL change notifications out to per-user subscribers.

One dedicated connection holds `LISTEN gigachad_changes` for the whole process
(the trigger from migration 0002 feeds it). Every notification is metadata only
and is copied into a bounded queue per subscriber; a subscriber that stops
draining its queue is closed instead of buffering forever, and resumes from
`GET /api/sync/stream?since=<seq>`.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import json
import logging
import time
from typing import AsyncIterator
from uuid import UUID

import psycopg

CHANNEL = "gigachad_changes"

# Enough to absorb a bulk import; past this a subscriber is too far behind to catch up
# from memory and is told to reconnect with ?since=<seq> instead.
QUEUE_MAXSIZE = 256

_BACKOFF_SECONDS = (1, 2, 5, 10, 30)
# A connection that stayed up this long counts as healthy, so the next drop retries fast.
_HEALTHY_SECONDS = 30

logger = logging.getLogger(__name__)

# None is the end-of-stream sentinel handed to a subscriber that is closed.
_Queue = asyncio.Queue["dict | None"]


def _close_queue(queue: _Queue) -> None:
    """Free a slot if needed and hand the subscriber its end-of-stream sentinel."""
    with suppress(asyncio.QueueEmpty):
        queue.get_nowait()
    with suppress(asyncio.QueueFull):
        queue.put_nowait(None)


class ChangeBroker:
    """Owns the LISTEN connection and the per-user subscriber queues."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[_Queue]] = {}
        self._task: asyncio.Task[None] | None = None
        self._database_url: str | None = None

    async def start(self, database_url: str) -> None:
        """Begin listening; a second call while running is a no-op."""
        if self._task is not None:
            return
        self._database_url = database_url
        self._task = asyncio.create_task(self._listen_forever())

    async def stop(self) -> None:
        """Stop the listener and close every subscriber. Safe when never started."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        for queues in self._subscribers.values():
            for queue in queues:
                _close_queue(queue)
        self._subscribers.clear()

    @asynccontextmanager
    async def subscribe(self, user_id: UUID) -> AsyncIterator[AsyncIterator[dict]]:
        """Yield an iterator over this user's live changes for as long as the block runs."""
        queue: _Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        key = str(user_id)
        self._subscribers.setdefault(key, set()).add(queue)
        try:
            yield self._drain(queue)
        finally:
            queues = self._subscribers.get(key)
            if queues is not None:
                queues.discard(queue)
                if not queues:
                    del self._subscribers[key]

    def dispatch(self, payload: str) -> None:
        """Copy one notification payload into every matching subscriber queue."""
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Ignoring malformed change notification")
            return
        queues = self._subscribers.get(str(event.get("user_id")))
        if not queues:
            return
        for queue in list(queues):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Dropping change subscriber that fell %d events behind", QUEUE_MAXSIZE)
                queues.discard(queue)
                _close_queue(queue)

    async def _drain(self, queue: _Queue) -> AsyncIterator[dict]:
        while True:
            event = await queue.get()
            if event is None:
                return
            yield event

    async def _listen_forever(self) -> None:
        attempt = 0
        while True:
            started = time.monotonic()
            try:
                await self._listen_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Change listener failed; reconnecting")
            attempt = 0 if time.monotonic() - started >= _HEALTHY_SECONDS else min(attempt + 1, len(_BACKOFF_SECONDS) - 1)
            await asyncio.sleep(_BACKOFF_SECONDS[attempt])

    async def _listen_once(self) -> None:
        assert self._database_url is not None
        async with await psycopg.AsyncConnection.connect(self._database_url, autocommit=True) as connection:
            await connection.execute(f"LISTEN {CHANNEL}")
            async for notify in connection.notifies():
                self.dispatch(notify.payload)


_broker = ChangeBroker()


def get_change_broker() -> ChangeBroker:
    """Return the process-wide broker shared by every SSE subscriber."""
    return _broker
