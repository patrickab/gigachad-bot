import asyncio
import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.http_plumbing import sse_tool_event_stream, stream_completion
from backend.identity import RequestIdentity, get_request_identity
from lib import completions, reasoning_levels
from lib.asset_store import AssetStore
from lib.image_paths import resolve_chat_image_paths, resolve_sandbox_prompt_images
from lib.memory_context import memory_prompt, memory_root, project_memory_store
from lib.project_store import ProjectStore
from lib.sandbox_service import SandboxService
from lib.tool_context import ToolContext, ToolOptions
from lib.tools import stream_chat_with_tools

from .deps import (
    get_asset_store,
    get_model_defaults,
    get_project_store,
    get_sandbox_service,
)

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    model: str
    chat_id: str
    user_msg: str
    system_prompt: str = ""
    reasoning_effort: str | None = None
    img_paths: list[str] = []
    downscale_images: bool = True
    messages: list[dict[str, str]] = []
    project_slug: str | None = None
    """Tool names the model may call this turn. Empty means a plain, tool-free completion."""
    tools: list[str] = []
    tool_options: ToolOptions = ToolOptions()


def _resolve_images(req: ChatRequest, assets: AssetStore) -> list | None:
    paths = resolve_chat_image_paths(
        req.chat_id,
        req.project_slug,
        req.img_paths,
        req.downscale_images,
        assets=assets,
    )
    if not paths:
        return None
    return paths if len(paths) > 1 else paths[0]


def _project(project_store: ProjectStore, slug: str | None) -> dict | None:
    """The chat's project, None outside projects or when it no longer exists."""
    if not slug:
        return None
    try:
        return project_store.get_project(slug)
    except FileNotFoundError:
        return None


def _memory_block(root: Path, project_store: ProjectStore, slug: str | None) -> str:
    """The memory block for the system prompt, empty when memory fails: the chat never fails because of memory."""
    try:
        return memory_prompt(root, project_memory_store(_project(project_store, slug), slug))
    except Exception:  # noqa: BLE001 - any memory failure (corrupt store, unusable root) only drops the block
        logger.exception("Memory block failed, the chat continues without it")
        return ""


@router.post("/chat")
async def chat(
    req: ChatRequest,
    assets: Annotated[AssetStore, Depends(get_asset_store)],
    sandbox_service: Annotated[SandboxService, Depends(get_sandbox_service)],
    model_defaults: Annotated[dict[str, str], Depends(get_model_defaults)],
    identity: Annotated[RequestIdentity, Depends(get_request_identity)],
    project_store: Annotated[ProjectStore, Depends(get_project_store)],
) -> EventSourceResponse:
    """Stream a completion as SSE, running the tool loop when `req.tools` is non-empty."""
    kwargs = reasoning_levels.effort_kwargs(req.reasoning_effort)
    root = memory_root(identity.user_id, identity.has_host_access)

    def prepare() -> tuple[list | bytes | str | None, str, tuple]:
        img = _resolve_images(req, assets)
        system_prompt = req.system_prompt
        block = _memory_block(root, project_store, req.project_slug) if "memory" in req.tools else ""
        if block:
            system_prompt = f"{system_prompt.rstrip()}\n\n{block}" if system_prompt.strip() else block
        prompt_images = (
            resolve_sandbox_prompt_images(req.chat_id, req.project_slug, req.img_paths, assets)
            if {"workspace_agent", "sandbox_plot"}.intersection(req.tools)
            else ()
        )
        return img, system_prompt, prompt_images

    # Upload reads, downscaling and the memory files block, so they run off the event loop.
    img, system_prompt, prompt_images = await asyncio.to_thread(prepare)
    if req.tools:
        return sse_tool_event_stream(
            stream_chat_with_tools(
                ToolContext(
                    model=req.model,
                    opts=req.tool_options,
                    chat_id=req.chat_id,
                    sandbox_service=sandbox_service,
                    prompt_images=prompt_images,
                    small_model=model_defaults["small_model"],
                    history=tuple(req.messages),
                    user_msg=req.user_msg,
                    img=img,
                    vision_model=model_defaults["vision_model"],
                    project_slug=req.project_slug,
                    project_store=project_store,
                    memory_root=root,
                ),
                system_prompt=system_prompt,
                enabled=req.tools,
                host_access=identity.has_host_access,
                **kwargs,
            )
        )
    # Encoding undownscaled uploads as data URIs blocks too.
    messages = await asyncio.to_thread(completions.build_messages, req.user_msg, req.messages, system_prompt, img)
    return stream_completion(req.model, messages, **kwargs)
