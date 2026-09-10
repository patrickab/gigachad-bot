from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path
from llm_baseclient.config import discover_ollama_models
from pydantic import BaseModel

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
    }


class ProviderDefinition(BaseModel):
    litellm_id: str
    models: list[str]


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
    return {"ollama": discover_ollama_models(), "providers": [{"label": label, **provider} for label, provider in visible.items()], "defaults": store.load_defaults()}


@router.put("/models/defaults")
async def save_model_defaults(body: ModelDefaults, store: ModelProviderStore = Depends(get_model_provider_store)) -> dict:
    try:
        defaults = store.save_defaults(body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    providers = providers_for_ui(store.load())
    return {"ollama": discover_ollama_models(), "providers": [{"label": label, **provider} for label, provider in providers.items()], "defaults": defaults}


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
