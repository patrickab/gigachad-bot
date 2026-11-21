import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel

from backend.http_plumbing import http_errors
from lib import ollama_models, pi_models, reasoning_levels
from lib.model_provider_store import ModelProviderStore, providers_for_ui
from lib.prompt_store import PromptStore
from lib.provider_usage import UsageUnavailable, provider_usage

from .deps import get_model_provider_store, get_prompt_store

router = APIRouter(prefix="/api", tags=["models"])

Models = Annotated[ModelProviderStore, Depends(get_model_provider_store)]
Prompts = Annotated[PromptStore, Depends(get_prompt_store)]

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

    def ollama() -> tuple[list[str], dict[str, list[str]] | None]:
        """Installed Ollama chat models as selectors, then the gateway's models."""
        return [f"ollama/{tag}" for tag in ollama_models.installed_chat_models()], pi_models.served_ladders()

    # Ollama's API runs alongside the store reads instead of after them.
    (installed, served), (all_providers, all_defaults, all_tab_order) = await asyncio.gather(
        asyncio.to_thread(ollama), asyncio.to_thread(stored)
    )

    visible = providers_for_ui(all_providers)
    catalog = [f"{provider['provider_id']}/{model}" for provider in visible.values() for model in provider["models"]]
    selectors = installed + catalog
    return {
        "ollama": installed,
        "providers": [{"label": label, **provider} for label, provider in visible.items()],
        # Catalog models the gateway does not list (D16), empty while the gateway is unreachable and nothing is known.
        "unavailable": [] if served is None else [s for s in selectors if pi_models.gateway_model_id(s) not in served],
        "defaults": all_defaults,
        "tab_order": all_tab_order,
    }


class OrderBody(BaseModel):
    order: list[str]


@router.get("/models")
async def get_models(store: Models) -> dict:
    return await _models_payload(store)


@router.get("/provider-usage")
async def get_provider_usage(refresh: bool = False) -> dict:
    """Subscription rate-limit usage of Pi's login; `refresh` bypasses pi-host's 60 s cache."""
    with http_errors((UsageUnavailable, 502, None)):
        return await asyncio.to_thread(provider_usage, refresh)


@router.get("/models/reasoning-levels")
async def get_reasoning_levels(model: str) -> dict:
    """Ordered `reasoning_effort` levels `model` accepts per the gateway, empty when it can't reason or isn't listed."""
    return {"levels": await asyncio.to_thread(reasoning_levels.levels, model)}


class ProviderDefinition(BaseModel):
    provider_id: str
    models: list[str]
    # Marks where the models came from (`pi_models.SOURCE`, the gateway row). Absent for hand-entered
    # providers. The store drops the key rather than persisting a null.
    source: str | None = None


class ProviderCatalog(BaseModel):
    providers: dict[str, ProviderDefinition]


class ModelDefaults(BaseModel):
    default_model: str
    small_model: str
    vision_model: str
    memory_model: str
    sandbox_model: str


@router.put("/models/providers")
async def save_model_providers(
    body: ProviderCatalog, store: Models
) -> dict:
    with http_errors((ValueError, 422, None)):
        providers = await asyncio.to_thread(store.save, {label: provider.model_dump() for label, provider in body.providers.items()})
    return await _models_payload(store, providers=providers)


@router.put("/models/defaults")
async def save_model_defaults(body: ModelDefaults, store: Models) -> dict:
    with http_errors((ValueError, 422, None)):
        defaults = await asyncio.to_thread(store.save_defaults, body.model_dump())
    return await _models_payload(store, defaults=defaults)


@router.put("/models/tab-order")
async def save_model_tab_order(body: OrderBody, store: Models) -> dict:
    with http_errors((ValueError, 422, None)):
        tab_order = await asyncio.to_thread(store.save_tab_order, body.order)
    return await _models_payload(store, tab_order=tab_order)


@router.get("/models/pi")
async def get_pi_models(refresh: bool = False) -> dict:
    """Providers pi-host's gateway serves, and their models.

    Reachability is part of the payload, never an HTTP error: pi-host can be
    stopped, and the settings panel renders that instead of failing to load.
    """
    return await asyncio.to_thread(pi_models.catalog, refresh=refresh)


@router.get("/prompts")
def get_prompts(store: Prompts) -> dict[str, dict[str, str]]:
    return {"prompts": store.prompt_map()}


@router.get("/prompts/list")
def list_prompts(store: Prompts) -> list[dict]:
    return store.list_prompts()


@router.get("/prompts/blocks")
def list_blocks(store: Prompts) -> dict[str, str]:
    return store.blocks()


@router.get("/prompts/{slug}")
def get_prompt(slug: Slug, store: Prompts) -> dict[str, str]:
    raw = store.get_raw(slug)
    if raw is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"slug": slug, "content": raw}


@router.put("/prompt-order")
def set_prompt_order(body: OrderBody, store: Prompts) -> dict[str, bool]:
    store.set_order(body.order)
    return {"ok": True}


class PromptSave(BaseModel):
    content: str


@router.put("/prompts/{slug}")
def save_prompt(slug: Slug, body: PromptSave, store: Prompts) -> dict[str, str]:
    name = store.save(slug, body.content)
    return {"slug": slug, "name": name}


@router.delete("/prompts/{slug}")
def delete_prompt(slug: Slug, store: Prompts) -> dict[str, bool]:
    ok = store.delete(slug)
    if not ok:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"deleted": True}
