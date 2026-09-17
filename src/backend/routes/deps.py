import asyncio
import base64
from contextlib import contextmanager
import json
from typing import Any, Iterator

from fastapi import Depends
from llm_baseclient.client import LLMClient
from sse_starlette.sse import EventSourceResponse

from backend.identity import RequestIdentity, get_request_identity
from config import get_data_store, get_postgres_pool, seed_prompts
from lib.architecture_graph import ArchitectureGraphStore
from lib.asset_store import AssetStore
from lib.chat_store import ChatStore
from lib.data_store import DataStore
from lib.file_vault import FileVault, PostgresVaultRootRepository
from lib.image_paths import _DATA_URI_RE
from lib.memory_store import MemoryStore
from lib.model_provider_store import ModelProviderStore
from lib.project_store import ProjectStore
from lib.prompt_store import PromptStore
from lib.storage_namespace import MODEL

_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def _store(identity: RequestIdentity) -> DataStore:
    return get_data_store(identity.user_id, device_id=identity.device_id)


def get_document_store(identity: RequestIdentity = Depends(get_request_identity)) -> DataStore:
    return get_data_store(identity.user_id, device_id=identity.device_id)


def get_asset_store(identity: RequestIdentity = Depends(get_request_identity)) -> AssetStore:
    return AssetStore(get_postgres_pool(), identity.user_id, device_id=identity.device_id)


def get_chat_store(identity: RequestIdentity = Depends(get_request_identity)) -> ChatStore:
    return ChatStore(data_store=_store(identity))


def get_project_store(identity: RequestIdentity = Depends(get_request_identity)) -> ProjectStore:
    data_store = _store(identity)
    chats = ChatStore(data_store=data_store)
    return ProjectStore(chat_store=chats, data_store=data_store)


def get_architecture_graph_store(
    identity: RequestIdentity = Depends(get_request_identity),
) -> ArchitectureGraphStore:
    return ArchitectureGraphStore(data_store=_store(identity))


def get_memory_store(identity: RequestIdentity = Depends(get_request_identity)) -> MemoryStore:
    return MemoryStore(data_store=_store(identity))


def get_file_vault(identity: RequestIdentity = Depends(get_request_identity)) -> FileVault:
    repository = PostgresVaultRootRepository(get_postgres_pool(), identity.user_id, device_id=identity.device_id)
    return FileVault(repository=repository)


def get_prompt_store(identity: RequestIdentity = Depends(get_request_identity)) -> PromptStore:
    data_store = _store(identity)
    seed_prompts(data_store)
    return PromptStore(data_store=data_store)


def get_model_provider_store(identity: RequestIdentity = Depends(get_request_identity)) -> ModelProviderStore:
    return ModelProviderStore(_store(identity), prefix=MODEL)


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
    match = _DATA_URI_RE.match(base64_data)
    if match:
        return base64.b64decode(match.group(1))
    return base64.b64decode(base64_data)


_SENTINEL = object()


def sse_event_stream(chunks: Iterator[str] | Iterator[str | dict]) -> EventSourceResponse:
    async def event_stream() -> Any:
        try:
            it = iter(chunks)
            while True:
                chunk = await asyncio.to_thread(next, it, _SENTINEL)
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
