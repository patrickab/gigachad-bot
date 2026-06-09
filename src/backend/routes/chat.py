from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .deps import decode_image, request_client, sse_event_stream

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    model: str
    user_msg: str
    system_prompt: str = ""
    temperature: float = Field(default=0.2, ge=0, le=2)
    reasoning_effort: str | None = None
    img_base64: str | None = None
    downscale_images: bool = True
    messages: list[dict[str, str]] = []


def _build_kwargs(req: ChatRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"temperature": req.temperature}
    if req.reasoning_effort and req.reasoning_effort != "none":
        kwargs["reasoning_effort"] = req.reasoning_effort
    return kwargs


@router.post("/chat")
async def chat(req: ChatRequest) -> EventSourceResponse:
    with request_client() as c:
        kwargs = _build_kwargs(req)
        img = decode_image(req.img_base64)
        chunks = c.api_query(
            model=req.model,
            user_msg=req.user_msg,
            user_msg_history=req.messages,
            system_prompt=req.system_prompt,
            img=img,
            stream=True,
            return_usage=True,
            **kwargs,
        )
        return sse_event_stream(chunks)


@router.post("/chat/nonstream")
async def chat_nonstream(req: ChatRequest) -> dict[str, Any]:
    with request_client() as c:
        kwargs = _build_kwargs(req)
        img = decode_image(req.img_base64)
        response = c.api_query(
            model=req.model,
            user_msg=req.user_msg,
            user_msg_history=req.messages,
            system_prompt=req.system_prompt,
            img=img,
            stream=False,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        return {"content": content}
