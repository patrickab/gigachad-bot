from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path
from llm_baseclient.config import (
    AVAILABLE_MODELS,
    MODELS_DEEPSEEK,
    MODELS_GEMINI,
    MODELS_OLLAMA,
)
from pydantic import BaseModel

from lib.prompt_store import PromptStore

from .deps import get_prompt_store

router = APIRouter(prefix="/api", tags=["models"])

# Restrict slugs to a flat filename — blocks path traversal (../, %2F) at the boundary.
Slug = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]+$")]


@router.get("/models")
async def get_models() -> dict:
    return {
        "all": AVAILABLE_MODELS,
        "ollama": MODELS_OLLAMA,
        "gemini": MODELS_GEMINI,
        "deepseek": MODELS_DEEPSEEK,
    }


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
