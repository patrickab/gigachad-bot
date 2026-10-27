"""Purge the uploads and sandbox state of chats once a store deletion succeeded."""

import asyncio
from collections.abc import Callable
import logging

from lib.asset_store import AssetStore
from lib.attachments import ChatUploads
from lib.chat_store import CleanupUploads
from lib.sandbox_service import SandboxService

log = logging.getLogger(__name__)


class ChatCleanup:
    """Runs a chat-deleting store call, then clears each deleted chat's uploads and sandbox state."""

    def __init__(self, assets: AssetStore, sandbox: SandboxService) -> None:
        self._assets = assets
        self._sandbox = sandbox

    async def after[T](self, run: Callable[[CleanupUploads], T]) -> T:
        """Run *run* in a thread with a per-chat cleanup callback, then delete the sandbox state of every chat it reported.

        Sandbox failures are logged, not raised, because the chats are already gone.
        """
        deleted_chat_ids: list[str] = []

        def cleanup(chat_id: str, slug: str | None = None) -> None:
            ChatUploads(self._assets, chat_id, slug).delete_all()
            # The store's cleanup callback cannot await, so sandbox deletion waits for the IDs recorded here.
            deleted_chat_ids.append(chat_id)

        result = await asyncio.to_thread(run, cleanup)
        for chat_id in deleted_chat_ids:
            try:
                await self._sandbox.delete_chat_state(chat_id=chat_id)
            except Exception:
                log.exception("sandbox cleanup failed for chat %s after history deletion", chat_id)
        return result
