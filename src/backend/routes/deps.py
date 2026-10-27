from collections.abc import Iterator
from contextlib import contextmanager
import hashlib

from fastapi import Depends
from llm_baseclient.client import LLMClient

from backend.identity import RequestIdentity, get_request_identity
from config import env_flag
from lib.agent_sandbox_adapter import AgentSandboxRunnerAdapter, SandboxRunner
from lib.architecture_workspace import ArchitectureStore
from lib.asset_store import AssetStore
from lib.canvas_store import CanvasMutationStore
from lib.changes import UserScope
from lib.chat_cleanup import ChatCleanup
from lib.chat_store import ChatStore
from lib.data_store import DataStore
from lib.fake_sandbox_runner import FakeSandboxRunner
from lib.file_roots import FileRoots
from lib.file_roots_repository import PostgresFileRootRepository
from lib.memory_store import MemoryStore
from lib.model_provider_store import ModelProviderStore
from lib.postgres_data_store import PostgresDataStore
from lib.postgres_pool import get_postgres_pool
from lib.project_rename import PostgresProjectRenamer
from lib.project_store import ProjectStore
from lib.prompt_store import PromptStore, seed_prompts
from lib.sandbox_service import SandboxRuntimeState, SandboxService

_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def get_user_scope(identity: RequestIdentity = Depends(get_request_identity)) -> UserScope:
    return UserScope(get_postgres_pool(), identity.user_id, identity.device_id)


def get_document_store(scope: UserScope = Depends(get_user_scope)) -> DataStore:
    return PostgresDataStore(scope)


def get_asset_store(scope: UserScope = Depends(get_user_scope)) -> AssetStore:
    return AssetStore(scope)


def get_chat_store(docs: DataStore = Depends(get_document_store)) -> ChatStore:
    return ChatStore(data_store=docs)


def get_canvas_store(scope: UserScope = Depends(get_user_scope)) -> CanvasMutationStore:
    return CanvasMutationStore(scope)


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
    return ModelProviderStore(docs)


def get_model_defaults(models: ModelProviderStore = Depends(get_model_provider_store)) -> dict[str, str]:
    """The user's model defaults, read once per request (FastAPI caches a dependency within a request)."""
    return models.load_defaults()


def get_sandbox_service(
    identity: RequestIdentity = Depends(get_request_identity),
    docs: DataStore = Depends(get_document_store),
    assets: AssetStore = Depends(get_asset_store),
    defaults: dict[str, str] = Depends(get_model_defaults),
) -> SandboxService:
    # Per-user sandbox slot namespace without exposing the user id.
    slot_prefix = hashlib.sha256(str(identity.user_id).encode("utf-8")).hexdigest()[:16]
    return SandboxService(
        data_store=docs,
        asset_store=assets,
        runner=get_sandbox_runner(),
        runtime_state=_sandbox_runtime,
        model=defaults["omp_model"],
        profile="gigachad",
        slot_prefix=slot_prefix,
    )


def get_chat_cleanup(
    assets: AssetStore = Depends(get_asset_store),
    sandbox: SandboxService = Depends(get_sandbox_service),
) -> ChatCleanup:
    return ChatCleanup(assets, sandbox)


def get_project_store(
    scope: UserScope = Depends(get_user_scope),
    chats: ChatStore = Depends(get_chat_store),
) -> ProjectStore:
    return ProjectStore(chats, rename=PostgresProjectRenamer(scope).rename)


def get_memory_store(docs: DataStore = Depends(get_document_store)) -> MemoryStore:
    return MemoryStore(docs)


def get_file_roots(
    identity: RequestIdentity = Depends(get_request_identity),
    scope: UserScope = Depends(get_user_scope),
) -> FileRoots:
    return FileRoots(repository=PostgresFileRootRepository(scope), host_access=identity.has_host_access)


def get_architecture_store(
    slug: str,
    docs: DataStore = Depends(get_document_store),
    roots: FileRoots = Depends(get_file_roots),
) -> ArchitectureStore:
    """The project's architecture: its linked file mount's ``docs/architecture`` when present, else the database copy."""
    return ArchitectureStore.for_project(docs, slug, roots.roots())


def get_prompt_store(docs: DataStore = Depends(get_document_store)) -> PromptStore:
    seed_prompts(docs)
    return PromptStore(docs)


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
