import asyncio
import base64
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
import hashlib
import json
import os
import re
from typing import Any

from fastapi import Depends
from llm_baseclient.client import LLMClient
from sse_starlette.sse import EventSourceResponse

from backend.identity import RequestIdentity, get_request_identity
from config import get_data_store, get_postgres_pool, seed_prompts
from lib.agent_sandbox_adapter import AgentSandboxRunnerAdapter, FakeSandboxRunner, SandboxRunner
from lib.architecture_graph import ArchitectureGraphStore
from lib.asset_store import AssetStore
from lib.chat_store import ChatStore
from lib.data_store import DataStore
from lib.file_vault import FileVault, PostgresVaultRootRepository
from lib.memory_store import MemoryStore
from lib.model_provider_store import ModelProviderStore
from lib.project_store import ProjectStore
from lib.prompt_store import PromptStore
from lib.sandbox_service import SandboxRuntimeState, SandboxService
from lib.storage_namespace import MODEL

_DATA_URI_RE = re.compile(r"data:image/\w+;base64,(.+)")

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


_sandbox_runner: SandboxRunner | None = None
# Keep runtime state beyond request scope until chat save.
_sandbox_runtime = SandboxRuntimeState()


def get_sandbox_runner() -> SandboxRunner:
    global _sandbox_runner
    if _sandbox_runner is None:
        use_fake = os.environ.get("GIGACHAD_SANDBOX_EXECUTION", "1").lower() in {"0", "false", "no"}
        use_fake = use_fake or os.environ.get("GIGACHAD_SANDBOX_FAKE", "").lower() in {"1", "true", "yes"}
        _sandbox_runner = FakeSandboxRunner() if use_fake else AgentSandboxRunnerAdapter()
    return _sandbox_runner


def get_sandbox_service(identity: RequestIdentity = Depends(get_request_identity)) -> SandboxService:
    data_store = _store(identity)
    assets = AssetStore(get_postgres_pool(), identity.user_id, device_id=identity.device_id)
    slot_prefix = hashlib.sha256(str(identity.user_id).encode("utf-8")).hexdigest()[:16]
    defaults = get_model_provider_store(identity).load_defaults()
    return SandboxService(
        data_store=data_store,
        asset_store=assets,
        runner=get_sandbox_runner(),
        runtime_state=_sandbox_runtime,
        model=defaults["omp_model"],
        profile="gigachad",
        slot_prefix=slot_prefix,
    )


def get_project_store(identity: RequestIdentity = Depends(get_request_identity)) -> ProjectStore:
    data_store = _store(identity)
    chats = ChatStore(data_store=data_store)
    return ProjectStore(chat_store=chats, data_store=data_store)


def get_architecture_graph_store(
    identity: RequestIdentity = Depends(get_request_identity),
) -> ArchitectureGraphStore:
    return ArchitectureGraphStore(data_store=_store(identity))


def get_memory_store(identity: RequestIdentity = Depends(get_request_identity)) -> MemoryStore:
    defaults = get_model_provider_store(identity).load_defaults()
    return MemoryStore(data_store=_store(identity), model=defaults["memory_model"])


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


def sse_tool_event_stream(events: AsyncIterator[tuple[str, Any]]) -> EventSourceResponse:
    """Forward a tool-loop event stream. `token` data stays raw text, everything else is JSON."""

    async def event_stream() -> Any:
        try:
            async for name, data in events:
                yield {"event": name, "data": data if isinstance(data, str) else json.dumps(data)}
            yield {"event": "done", "data": ""}
        except Exception as e:  # noqa: BLE001 - a failed turn must reach the UI as an error event
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_stream())
