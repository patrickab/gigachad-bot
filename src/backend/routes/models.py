import asyncio
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path
from llm_baseclient.config import discover_ollama_models
from pydantic import BaseModel

from backend.http_plumbing import http_errors
from lib import omp_source, reasoning_levels
from lib.model_provider_store import ModelProviderStore, providers_for_ui
from lib.prompt_store import PromptStore
from lib.provider_usage import UsageUnavailable, provider_usage

from .deps import get_model_provider_store, get_prompt_store

router = APIRouter(prefix="/api", tags=["models"])

# Restrict slugs to a flat filename. Blocks path traversal (../, %2F) at the boundary.
Slug = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]+$")]


# Store calls block on Postgres, so they run in a worker thread (or the handler is sync, run in FastAPI's threadpool):
# a blocking call on the event loop would stall every concurrent request, SSE streams included.
async def _models_payload(
    store: ModelProviderStore,
    *,
    providers: dict | None = None,
    defaults: dict | None = None,
    tab_order: list[str] | None = None,
) -> dict:
    """The models catalog, where a part the caller just saved replaces re-reading it from the store."""

    def stored() -> tuple[dict, dict, list[str]]:
        return (
            store.load() if providers is None else providers,
            store.load_defaults() if defaults is None else defaults,
            store.load_tab_order() if tab_order is None else tab_order,
        )

    # `ollama list` is a subprocess, so it runs alongside the store reads instead of after them.
    ollama, (all_providers, all_defaults, all_tab_order) = await asyncio.gather(
        asyncio.to_thread(discover_ollama_models), asyncio.to_thread(stored)
    )
    visible = providers_for_ui(all_providers)
    return {
        "ollama": ollama,
        "providers": [{"label": label, **provider} for label, provider in visible.items()],
        "defaults": all_defaults,
        "tab_order": all_tab_order,
    }


class OrderBody(BaseModel):
    order: list[str]


@router.get("/models")
async def get_models(store: ModelProviderStore = Depends(get_model_provider_store)) -> dict:
    return await _models_payload(store)


@router.get("/provider-usage")
async def get_provider_usage(refresh: bool = False) -> dict:
    """Subscription rate-limit usage per OMP login; `refresh` refetches instead of using the 5 minute cache."""
    with http_errors((UsageUnavailable, 502, None)):
        return await asyncio.to_thread(provider_usage, refresh)


@router.get("/models/reasoning-levels")
async def get_reasoning_levels(model: str, background_tasks: BackgroundTasks) -> dict:
    """Ordered `reasoning_effort` levels `model` accepts, empty when it can't reason.

    - Stored levels answer instantly, a background lookup revalidates them
    - Unknown models resolve live once, failed lookups (e.g. Ollama down) get no levels
    """
    stored = await asyncio.to_thread(reasoning_levels.load_stored, model)
    if stored is not None:
        background_tasks.add_task(reasoning_levels.refresh, model)
        return {"levels": stored}
    return {"levels": await reasoning_levels.refresh(model) or []}


class ProviderDefinition(BaseModel):
    litellm_id: str
    models: list[str]
    # Marks where the models came from, e.g. "omp". Absent for hand-entered
    # providers. The store drops the key rather than persisting a null.
    source: str | None = None


class ProviderCatalog(BaseModel):
    providers: dict[str, ProviderDefinition]


class ModelDefaults(BaseModel):
    default_model: str
    small_model: str
    vision_model: str
    memory_model: str
    omp_model: str


@router.put("/models/providers")
async def save_model_providers(
    body: ProviderCatalog, store: ModelProviderStore = Depends(get_model_provider_store)
) -> dict:
    with http_errors((ValueError, 422, None)):
        providers = await asyncio.to_thread(store.save, {label: provider.model_dump() for label, provider in body.providers.items()})
    return await _models_payload(store, providers=providers)


@router.put("/models/defaults")
async def save_model_defaults(body: ModelDefaults, store: ModelProviderStore = Depends(get_model_provider_store)) -> dict:
    with http_errors((ValueError, 422, None)):
        defaults = await asyncio.to_thread(store.save_defaults, body.model_dump())
    return await _models_payload(store, defaults=defaults)


@router.put("/models/tab-order")
async def save_model_tab_order(body: OrderBody, store: ModelProviderStore = Depends(get_model_provider_store)) -> dict:
    with http_errors((ValueError, 422, None)):
        tab_order = await asyncio.to_thread(store.save_tab_order, body.order)
    return await _models_payload(store, tab_order=tab_order)


@router.get("/models/omp")
async def get_omp_models(refresh: bool = False) -> dict:
    """Providers OMP is logged into, and the models each one can serve.

    Reachability is part of the payload, never an HTTP error: OMP can be
    installed while its gateway is stopped, and the settings panel renders
    that difference instead of failing to load.
    """
    return await asyncio.to_thread(omp_source.catalog, refresh=refresh)


@router.get("/prompts")
def get_prompts(store: PromptStore = Depends(get_prompt_store)) -> dict[str, dict[str, str]]:
    return {"prompts": store.prompt_map()}


@router.get("/prompts/list")
def list_prompts(store: PromptStore = Depends(get_prompt_store)) -> list[dict]:
    return store.list_prompts()


@router.get("/prompts/blocks")
def list_blocks(store: PromptStore = Depends(get_prompt_store)) -> dict[str, str]:
    return store.blocks()


@router.get("/prompts/{slug}")
def get_prompt(slug: Slug, store: PromptStore = Depends(get_prompt_store)) -> dict[str, str]:
    raw = store.get_raw(slug)
    if raw is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"slug": slug, "content": raw}


@router.put("/prompt-order")
def set_prompt_order(body: OrderBody, store: PromptStore = Depends(get_prompt_store)) -> dict[str, bool]:
    store.set_order(body.order)
    return {"ok": True}


class PromptSave(BaseModel):
    content: str


@router.put("/prompts/{slug}")
def save_prompt(slug: Slug, body: PromptSave, store: PromptStore = Depends(get_prompt_store)) -> dict[str, str]:
    name = store.save(slug, body.content)
    return {"slug": slug, "name": name}


@router.delete("/prompts/{slug}")
def delete_prompt(slug: Slug, store: PromptStore = Depends(get_prompt_store)) -> dict[str, bool]:
    ok = store.delete(slug)
    if not ok:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"deleted": True}
