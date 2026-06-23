import asyncio
import base64
from contextlib import contextmanager
import json
import re
from typing import Any, Iterator

from llm_baseclient.client import LLMClient
from sse_starlette.sse import EventSourceResponse

from config import BASE_DIR, DIRECTORY_CHAT_HISTORIES
from lib.chat_store import ChatStore
from lib.memory_store import MemoryStore
from lib.obsidian_vault import ObsidianVault
from lib.project_store import ProjectStore
from lib.prompt_store import PromptStore

_client: LLMClient | None = None
_chat_store: ChatStore | None = None
_project_store: ProjectStore | None = None
_memory_store: MemoryStore | None = None
_obsidian_vault: ObsidianVault | None = None
_prompt_store: PromptStore | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def get_chat_store() -> ChatStore:
    global _chat_store
    if _chat_store is None:
        _chat_store = ChatStore(DIRECTORY_CHAT_HISTORIES)
    return _chat_store


def get_project_store() -> ProjectStore:
    global _project_store
    if _project_store is None:
        _project_store = ProjectStore(DIRECTORY_CHAT_HISTORIES, chat_store=get_chat_store())
    return _project_store


def get_memory_store() -> MemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store


def get_obsidian_vault() -> ObsidianVault:
    global _obsidian_vault
    if _obsidian_vault is None:
        _obsidian_vault = ObsidianVault()
    return _obsidian_vault


def get_prompt_store() -> PromptStore:
    global _prompt_store
    if _prompt_store is None:
        _prompt_store = PromptStore(BASE_DIR / "prompts")
    return _prompt_store


def shutdown_client() -> None:
    global _client
    if _client is not None:
        _client.kill_inference_engines()
        _client = None


@contextmanager
def request_client() -> Iterator[LLMClient]:
    c = get_client()
    c.reset_history()
    try:
        yield c
    finally:
        c.reset_history()


def decode_image(base64_data: str | None) -> bytes | None:
    if not base64_data:
        return None
    match = re.match(r"data:image/\w+;base64,(.+)", base64_data)
    if match:
        return base64.b64decode(match.group(1))
    return base64.b64decode(base64_data)


_SENTINEL = object()


def sse_event_stream(chunks: Iterator[str] | Iterator[str | dict]) -> EventSourceResponse:
    async def event_stream() -> Any:
        try:
            loop = asyncio.get_running_loop()
            it = iter(chunks)
            while True:
                chunk = await loop.run_in_executor(None, lambda: next(it, _SENTINEL))
                if chunk is _SENTINEL:
                    break
                if isinstance(chunk, dict):
                    yield {"event": "usage", "data": json.dumps(chunk)}
                else:
                    yield {"event": "token", "data": chunk}
            yield {"event": "done", "data": ""}
        except Exception as e:
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_stream())
