from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from lib.memory_store import MemoryStore

from .deps import decode_image, get_memory_store, request_client, sse_event_stream

router = APIRouter(prefix="/api", tags=["chat"])

MemoryStoreDep = Annotated[MemoryStore, Depends(get_memory_store)]


class ChatRequest(BaseModel):
    model: str
    user_msg: str
    system_prompt: str = ""
    temperature: float = Field(default=0.2, ge=0, le=2)
    reasoning_effort: str | None = None
    img_base64: str | None = None
    downscale_images: bool = True
    messages: list[dict[str, str]] = []
    project_slug: str | None = None


def _build_kwargs(req: ChatRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"temperature": req.temperature}
    if req.reasoning_effort and req.reasoning_effort != "none":
        kwargs["reasoning_effort"] = req.reasoning_effort
    return kwargs


@router.post("/chat")
async def chat(req: ChatRequest, memory_store: MemoryStoreDep) -> EventSourceResponse:
    with request_client() as c:
        kwargs = _build_kwargs(req)
        img = decode_image(req.img_base64)
        system_prompt = memory_store.augment_system_prompt(req.system_prompt, req.project_slug)
        chunks = c.api_query(
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


@router.post("/chat/nonstream")
async def chat_nonstream(req: ChatRequest, memory_store: MemoryStoreDep) -> dict[str, Any]:
    with request_client() as c:
        kwargs = _build_kwargs(req)
        img = decode_image(req.img_base64)
        system_prompt = memory_store.augment_system_prompt(req.system_prompt, req.project_slug)
        response = c.api_query(
            model=req.model,
            user_msg=req.user_msg,
            user_msg_history=req.messages,
            system_prompt=system_prompt,
            img=img,
            stream=False,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        return {"content": content}
