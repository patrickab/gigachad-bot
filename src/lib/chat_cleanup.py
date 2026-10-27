"""Release what deleted chats leave behind once a store deletion committed: uploads, derived data, sandbox state."""

import asyncio
from collections.abc import Callable
import logging

from lib.references import References, Released
from lib.sandbox_service import SandboxService

log = logging.getLogger(__name__)


class ChatCleanup:
    """Runs a chat-deleting store call, then releases what it deleted and clears each deleted chat's sandbox state."""

    def __init__(self, references: References, sandbox: SandboxService) -> None:
        self._references = references
        self._sandbox = sandbox

    async def after[T](self, run: Callable[[Released], T]) -> T:
        """Run *run* in a thread with a `Released` it records its deletions in, then release them after it committed.

        A failing *run* releases nothing. Sandbox failures are logged, not raised, because the chats are already gone.
        """
        released = Released()
        result = await asyncio.to_thread(run, released)
        await asyncio.to_thread(self._references.release, released)
        for chat_id, _ in released.chats:
            try:
                await self._sandbox.delete_chat_state(chat_id=chat_id)
            except Exception:
                log.exception("sandbox cleanup failed for chat %s after history deletion", chat_id)
        return result
