from typing import Annotated

from fastapi import APIRouter, Depends
from llm_baseclient.client import LLMClient
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.http_plumbing import sse_tool_event_stream, stream_completion
from lib import reasoning_levels
from lib.asset_store import AssetStore
from lib.image_paths import resolve_chat_image_paths, resolve_sandbox_prompt_images
from lib.memory_context import augment_system_prompt
from lib.memory_store import MemoryStore
from lib.sandbox_service import SandboxService
from lib.tool_context import ToolOptions
from lib.tools import stream_chat_with_tools

from .deps import (
    get_asset_store,
    get_memory_store,
    get_model_defaults,
    get_sandbox_service,
    request_client,
)

router = APIRouter(prefix="/api", tags=["chat"])

MemoryStoreDep = Annotated[MemoryStore, Depends(get_memory_store)]


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


def _resolve_images(c: LLMClient, req: ChatRequest, assets: AssetStore) -> list | None:
    paths = resolve_chat_image_paths(
        c,
        req.chat_id,
        req.project_slug,
        req.img_paths,
        req.downscale_images,
        assets=assets,
    )
    if not paths:
        return None
    return paths if len(paths) > 1 else paths[0]


@router.post("/chat")
async def chat(
    req: ChatRequest,
    memory_store: MemoryStoreDep,
    assets: Annotated[AssetStore, Depends(get_asset_store)],
    sandbox_service: Annotated[SandboxService, Depends(get_sandbox_service)],
    model_defaults: Annotated[dict[str, str], Depends(get_model_defaults)],
) -> EventSourceResponse:
    """Stream a completion as SSE, running the tool loop when `req.tools` is non-empty."""
    with request_client() as c:
        kwargs = reasoning_levels.effort_kwargs(req.reasoning_effort, req.model)
        img = _resolve_images(c, req, assets)
        system_prompt = augment_system_prompt(memory_store, req.system_prompt, req.project_slug)
        prompt_images = (
            resolve_sandbox_prompt_images(req.chat_id, req.project_slug, req.img_paths, assets)
            if {"workspace_agent", "sandbox_plot"}.intersection(req.tools)
            else ()
        )
        if req.tools:
            # The tool loop owns its own model calls, so it needs the client to outlive this
            # `with` block; `request_client` only guards conversation state, which the loop
            # never touches (it passes an explicit message list on every call).
            return sse_tool_event_stream(
                stream_chat_with_tools(
                    client=c,
                    model=req.model,
                    user_msg=req.user_msg,
                    history=req.messages,
                    system_prompt=system_prompt,
                    img=img,
                    enabled=req.tools,
                    opts=req.tool_options,
                    chat_id=req.chat_id,
                    sandbox_service=sandbox_service,
                    prompt_images=prompt_images,
                    small_model=model_defaults["small_model"],
                    vision_model=model_defaults["vision_model"],
                    **kwargs,
                )
            )
        return await stream_completion(
            c,
            model=req.model,
            user_msg=req.user_msg,
            user_msg_history=req.messages,
            system_prompt=system_prompt,
            img=img,
            return_usage=True,
            **kwargs,
        )

