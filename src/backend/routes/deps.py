import base64
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
import hashlib
import json
import re
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from llm_baseclient.client import LLMClient
from sse_starlette.sse import EventSourceResponse

from backend.canvas import CanvasMutationStore
from backend.identity import RequestIdentity, get_request_identity
from config import env_flag, get_postgres_pool, seed_prompts
from lib.agent_sandbox_adapter import AgentSandboxRunnerAdapter, FakeSandboxRunner, SandboxRunner
from lib.architecture_workspace import ArchitectureStore
from lib.asset_store import AssetStore
from lib.async_utils import aiter_in_thread
from lib.chat_store import ChatStore
from lib.data_store import DataStore
from lib.file_roots import FileRoots, PostgresFileRootRepository
from lib.llm_resilience import api_query_resilient
from lib.memory_store import MemoryStore
from lib.model_provider_store import ModelProviderStore
from lib.postgres_data_store import PostgresDataStore
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


def get_document_store(identity: RequestIdentity = Depends(get_request_identity)) -> DataStore:
    return PostgresDataStore(get_postgres_pool(), identity.user_id, device_id=identity.device_id)


def get_asset_store(identity: RequestIdentity = Depends(get_request_identity)) -> AssetStore:
    return AssetStore(get_postgres_pool(), identity.user_id, device_id=identity.device_id)


def get_chat_store(docs: DataStore = Depends(get_document_store)) -> ChatStore:
    return ChatStore(data_store=docs)


def get_canvas_store(identity: RequestIdentity = Depends(get_request_identity)) -> CanvasMutationStore:
    return CanvasMutationStore(get_postgres_pool(), identity.user_id, identity.device_id)


_sandbox_runner: SandboxRunner | None = None
# Keep runtime state beyond request scope until chat save.
_sandbox_runtime = SandboxRuntimeState()


def get_sandbox_runner() -> SandboxRunner:
    global _sandbox_runner
    if _sandbox_runner is None:
        # Fake runner when execution is disabled (EXECUTION=0) or explicitly requested (FAKE=1).
        use_fake = not env_flag("GIGACHAD_SANDBOX_EXECUTION", True) or env_flag("GIGACHAD_SANDBOX_FAKE", False)
        _sandbox_runner = FakeSandboxRunner() if use_fake else AgentSandboxRunnerAdapter()
    return _sandbox_runner


def get_model_provider_store(docs: DataStore = Depends(get_document_store)) -> ModelProviderStore:
    return ModelProviderStore(docs, prefix=MODEL)


def get_sandbox_service(
    identity: RequestIdentity = Depends(get_request_identity),
    docs: DataStore = Depends(get_document_store),
    assets: AssetStore = Depends(get_asset_store),
    models: ModelProviderStore = Depends(get_model_provider_store),
) -> SandboxService:
    # Per-user sandbox slot namespace without exposing the user id.
    slot_prefix = hashlib.sha256(str(identity.user_id).encode("utf-8")).hexdigest()[:16]
    return SandboxService(
        data_store=docs,
        asset_store=assets,
        runner=get_sandbox_runner(),
        runtime_state=_sandbox_runtime,
        model=models.load_defaults()["omp_model"],
        profile="gigachad",
        slot_prefix=slot_prefix,
    )


def get_project_store(docs: DataStore = Depends(get_document_store)) -> ProjectStore:
    return ProjectStore(chat_store=ChatStore(data_store=docs), data_store=docs)


def get_memory_store(
    docs: DataStore = Depends(get_document_store),
    models: ModelProviderStore = Depends(get_model_provider_store),
) -> MemoryStore:
    return MemoryStore(data_store=docs, model=models.load_defaults()["memory_model"])


def get_file_roots(identity: RequestIdentity = Depends(get_request_identity)) -> FileRoots:
    repository = PostgresFileRootRepository(get_postgres_pool(), identity.user_id, device_id=identity.device_id)
    return FileRoots(repository=repository, host_access=identity.has_host_access)


def get_architecture_store(
    slug: str,
    docs: DataStore = Depends(get_document_store),
    roots: FileRoots = Depends(get_file_roots),
) -> ArchitectureStore:
    """The project's architecture: its linked file mount's ``docs/architecture`` when present, else the database copy."""
    return ArchitectureStore.for_project(docs, slug, roots.roots())


def get_prompt_store(docs: DataStore = Depends(get_document_store)) -> PromptStore:
    seed_prompts(docs)
    return PromptStore(data_store=docs)


def shutdown_client() -> None:
    global _client
    if _client is not None:
        _client.kill_inference_engines()
        _client = None


@contextmanager
def request_client() -> Iterator[LLMClient]:
    """Yield the shared client with its history reset, since it is one process-wide instance."""
    c = get_client()
    c.reset_history()
    try:
        yield c
    finally:
        c.reset_history()


@contextmanager
def http_errors(*rules: tuple[type[Exception] | tuple[type[Exception], ...], int, str | None]) -> Iterator[None]:
    """Re-raise as `HTTPException` per the first matching `(exception, status, detail)` rule, a `None` detail uses `str(exc)`."""
    try:
        yield
    except Exception as exc:
        for kinds, status, detail in rules:
            if isinstance(exc, kinds):
                raise HTTPException(status_code=status, detail=str(exc) if detail is None else detail) from exc
        raise


def decode_image(base64_data: str | None) -> bytes | None:
    """Decode a raw or `data:image/...;base64,` string to bytes."""
    if not base64_data:
        return None
    match = _DATA_URI_RE.match(base64_data)
    if match:
        return base64.b64decode(match.group(1))
    return base64.b64decode(base64_data)


_HEARTBEAT_SECONDS = 15
_SSE_HEADERS = {
    # Some reverse proxies otherwise buffer a small SSE response until the model
    # finishes. Vision providers commonly have larger first-token gaps, exposing
    # that buffering as an apparently non-streaming reply.
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
}


def _sse(events: AsyncIterator[tuple[str, str]]) -> EventSourceResponse:
    """Forward `(event, data)` pairs, then `done`, or an `error` event when the stream fails."""

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        try:
            async for event, data in events:
                yield {"event": event, "data": data}
            yield {"event": "done", "data": ""}
        except Exception as e:  # noqa: BLE001 - a failed turn must reach the UI as an error event
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_stream(), headers=_SSE_HEADERS)


async def stream_completion(client: LLMClient, **query: Any) -> EventSourceResponse:
    """Stream a completion as SSE: string chunks become `token` events, dict chunks `usage` events."""
    chunks = await run_in_threadpool(api_query_resilient, client, stream=True, **query)

    async def events() -> AsyncIterator[tuple[str, str]]:
        async for chunk in aiter_in_thread(chunks):
            yield ("usage", json.dumps(chunk)) if isinstance(chunk, dict) else ("token", chunk)

    return _sse(events())


def sse_tool_event_stream(events: AsyncIterator[tuple[str, Any]]) -> EventSourceResponse:
    """Forward a tool-loop event stream. `token` data stays raw text, everything else is JSON."""

    async def tagged() -> AsyncIterator[tuple[str, str]]:
        async for name, data in events:
            yield name, data if isinstance(data, str) else json.dumps(data)

    return _sse(tagged())


def live_stream(events: AsyncIterator[dict[str, str]]) -> EventSourceResponse:
    """Serve long-lived SSE events, pinging idle connections so proxies keep them open."""
    return EventSourceResponse(events, ping=_HEARTBEAT_SECONDS)
