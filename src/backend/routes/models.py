import asyncio
from typing import Annotated

import httpx
import litellm
from fastapi import APIRouter, Depends, HTTPException, Path
from llm_baseclient.config import discover_ollama_models
from pydantic import BaseModel

from config import OLLAMA_BASE_URL
from lib import omp_source
from lib.model_provider_store import ModelProviderStore, providers_for_ui
from lib.prompt_store import PromptStore

from .deps import get_model_provider_store, get_prompt_store

router = APIRouter(prefix="/api", tags=["models"])

# Restrict slugs to a flat filename — blocks path traversal (../, %2F) at the boundary.
Slug = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]+$")]


@router.get("/models")
async def get_models(store: ModelProviderStore = Depends(get_model_provider_store)) -> dict:
    providers = providers_for_ui(store.load())
    return {
        "ollama": discover_ollama_models(),
        "providers": [{"label": label, **provider} for label, provider in providers.items()],
        "defaults": store.load_defaults(),
        "tab_order": store.load_tab_order(),
    }


async def _ollama_supports_reasoning(model: str) -> bool:
    """Ask the Ollama server itself: LiteLLM's static registry doesn't know
    locally pulled or Ollama Cloud model tags (`gemma4:31b-cloud`), but
    `/api/show` reports each model's real capabilities, `thinking` among
    them, for both local and cloud-proxied models.
    """
    try:
        async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=5.0) as client:
            resp = await client.post("/api/show", json={"model": model})
            resp.raise_for_status()
            return "thinking" in (resp.json().get("capabilities") or [])
    except (httpx.HTTPError, ValueError):
        return False


@router.get("/models/reasoning-support")
async def get_reasoning_support(model: str) -> dict:
    """Whether `model` accepts `reasoning_effort`.

    Ollama-served models (`ollama/<tag>`, including Ollama Cloud tags) are
    checked against the live Ollama server, since those model tags aren't in
    LiteLLM's static registry. Every other model uses LiteLLM's metadata.
    Never raises: unknown or unreachable models fall back to `False`.
    """
    if model.startswith("ollama/"):
        return {"supports_reasoning": await _ollama_supports_reasoning(model.removeprefix("ollama/"))}
    return {"supports_reasoning": litellm.supports_reasoning(model=model)}


class ProviderDefinition(BaseModel):
    litellm_id: str
    models: list[str]
    # Marks where the models came from, e.g. "omp". Absent for hand-entered
    # providers; the store drops the key rather than persisting a null.
    source: str | None = None


class ProviderCatalog(BaseModel):
    providers: dict[str, ProviderDefinition]


class ModelDefaults(BaseModel):
    default_model: str
    small_model: str
    vision_model: str
    memory_model: str


@router.put("/models/providers")
async def save_model_providers(
    body: ProviderCatalog, store: ModelProviderStore = Depends(get_model_provider_store)
) -> dict:
    try:
        providers = store.save({label: provider.model_dump() for label, provider in body.providers.items()})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    visible = providers_for_ui(providers)
    return {"ollama": discover_ollama_models(), "providers": [{"label": label, **provider} for label, provider in visible.items()], "defaults": store.load_defaults(), "tab_order": store.load_tab_order()}


@router.put("/models/defaults")
async def save_model_defaults(body: ModelDefaults, store: ModelProviderStore = Depends(get_model_provider_store)) -> dict:
    try:
        defaults = store.save_defaults(body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    providers = providers_for_ui(store.load())
    return {"ollama": discover_ollama_models(), "providers": [{"label": label, **provider} for label, provider in providers.items()], "defaults": defaults, "tab_order": store.load_tab_order()}


class TabOrder(BaseModel):
    order: list[str]


@router.put("/models/tab-order")
async def save_model_tab_order(body: TabOrder, store: ModelProviderStore = Depends(get_model_provider_store)) -> dict:
    try:
        tab_order = store.save_tab_order(body.order)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    providers = providers_for_ui(store.load())
    return {"ollama": discover_ollama_models(), "providers": [{"label": label, **provider} for label, provider in providers.items()], "defaults": store.load_defaults(), "tab_order": tab_order}


@router.get("/models/omp")
async def get_omp_models(refresh: bool = False) -> dict:
    """Providers OMP is logged into, and the models each one can serve.

    Reachability is part of the payload, never an HTTP error: OMP can be
    installed while its gateway is stopped, and the settings panel renders
    that difference instead of failing to load.
    """
    return await asyncio.to_thread(omp_source.catalog, refresh=refresh)


@router.get("/prompts")
async def get_prompts(store: PromptStore = Depends(get_prompt_store)) -> dict[str, dict[str, str]]:
    return {"prompts": store.prompt_map()}


@router.get("/prompts/list")
async def list_prompts(store: PromptStore = Depends(get_prompt_store)) -> list[dict]:
    return store.list_prompts()


@router.get("/prompts/blocks")
async def list_blocks(store: PromptStore = Depends(get_prompt_store)) -> dict[str, str]:
    return store.blocks()


@router.get("/prompts/{slug}")
async def get_prompt(slug: Slug, store: PromptStore = Depends(get_prompt_store)) -> dict[str, str]:
    raw = store.get_raw(slug)
    if raw is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"slug": slug, "content": raw}


class PromptOrder(BaseModel):
    order: list[str]


@router.put("/prompt-order")
async def set_prompt_order(body: PromptOrder, store: PromptStore = Depends(get_prompt_store)) -> dict[str, bool]:
    store.set_order(body.order)
    return {"ok": True}


class PromptSave(BaseModel):
    content: str


@router.put("/prompts/{slug}")
async def save_prompt(slug: Slug, body: PromptSave, store: PromptStore = Depends(get_prompt_store)) -> dict[str, str]:
    name = store.save(slug, body.content)
    return {"slug": slug, "name": name}


@router.delete("/prompts/{slug}")
async def delete_prompt(slug: Slug, store: PromptStore = Depends(get_prompt_store)) -> dict[str, bool]:
    ok = store.delete(slug)
    if not ok:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"deleted": True}
