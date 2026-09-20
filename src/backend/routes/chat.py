from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
import litellm
from llm_baseclient.client import LLMClient
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from lib.asset_store import AssetStore
from lib.image_paths import resolve_chat_image_paths, resolve_sandbox_prompt_images
from lib.llm_resilience import api_query_resilient
from lib.memory_store import MemoryStore
from lib.model_provider_store import ModelProviderStore
from lib.sandbox_service import SandboxService
from lib.tools import ToolOptions, stream_chat_with_tools

from .deps import (
    get_asset_store,
    get_memory_store,
    get_model_provider_store,
    get_sandbox_service,
    request_client,
    sse_event_stream,
    sse_tool_event_stream,
)

router = APIRouter(prefix="/api", tags=["chat"])

MemoryStoreDep = Annotated[MemoryStore, Depends(get_memory_store)]


class ChatRequest(BaseModel):
    model: str
    chat_id: str
    user_msg: str
    system_prompt: str = ""
    temperature: float = Field(default=0.2, ge=0, le=2)
    reasoning_effort: str | None = None
    img_paths: list[str] = []
    downscale_images: bool = True
    messages: list[dict[str, str]] = []
    project_slug: str | None = None
    """Tool names the model may call this turn. Empty means a plain, tool-free completion."""
    tools: list[str] = []
    tool_options: ToolOptions = ToolOptions()


def _build_kwargs(req: ChatRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"temperature": req.temperature}
    if req.reasoning_effort and req.reasoning_effort != "none" and _supports_reasoning_effort(req.model):
        kwargs["reasoning_effort"] = req.reasoning_effort
    return kwargs


def _supports_reasoning_effort(model: str) -> bool:
    """Not every provider accepts `reasoning_effort` (e.g. nvidia_nim rejects the
    param outright, regardless of value, and litellm raises UnsupportedParamsError).
    Ask litellm's own capability table instead of hardcoding a provider allowlist,
    so newly supported providers/models pick this up automatically. Fail closed on
    lookup failure (unrecognized/malformed model string): dropping the param at
    worst silently ignores a chosen reasoning level, whereas forwarding it blind
    risks the exact crash this function exists to prevent.
    """
    try:
        return "reasoning_effort" in (litellm.get_supported_openai_params(model=model) or [])
    except Exception:
        return False


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
    models: Annotated[ModelProviderStore, Depends(get_model_provider_store)],
) -> EventSourceResponse:
    with request_client() as c:
        model_defaults = models.load_defaults()
        kwargs = _build_kwargs(req)
        img = _resolve_images(c, req, assets)
        system_prompt = memory_store.augment_system_prompt(req.system_prompt, req.project_slug)
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
                    **kwargs,
                )
            )
        chunks = await run_in_threadpool(
            api_query_resilient,
            c,
            model=req.model,
            user_msg=req.user_msg,
            user_msg_history=req.messages,
            system_prompt=system_prompt,
            img=img,
            stream=True,
            return_usage=True,
            **kwargs,
        )
        return sse_event_stream(chunks)

