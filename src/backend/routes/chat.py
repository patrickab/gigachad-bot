from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .deps import decode_image, get_client

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    model: str
    user_msg: str
    system_prompt: str = ""
    temperature: float = 0.2
    top_p: float = 0.95
    reasoning_effort: str | None = None
    img_base64: str | None = None
    downscale_images: bool = True
    messages: list[dict[str, Any]] = []


@router.post("/chat")
async def chat(req: ChatRequest) -> EventSourceResponse:
    c = get_client()
    kwargs: dict[str, Any] = {"temperature": req.temperature, "top_p": req.top_p}
    if req.reasoning_effort and req.reasoning_effort != "none":
        kwargs["reasoning_effort"] = req.reasoning_effort

    img = decode_image(req.img_base64)

    async def event_stream() -> Any:
        try:
            c.reset_history()
            for chunk in c.api_query(
                model=req.model,
                user_msg=req.user_msg,
                user_msg_history=[],
                system_prompt=req.system_prompt,
                img=img,
                stream=True,
                **kwargs,
            ):
                yield {"event": "token", "data": chunk}
            yield {"event": "done", "data": ""}
        except Exception as e:
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_stream())


@router.post("/chat/nonstream")
async def chat_nonstream(req: ChatRequest) -> dict[str, Any]:
    c = get_client()
    kwargs: dict[str, Any] = {"temperature": req.temperature, "top_p": req.top_p}
    if req.reasoning_effort and req.reasoning_effort != "none":
        kwargs["reasoning_effort"] = req.reasoning_effort

    img = decode_image(req.img_base64)

    c.reset_history()
    response = c.api_query(
        model=req.model,
        user_msg=req.user_msg,
        user_msg_history=[],
        system_prompt=req.system_prompt,
        img=img,
        stream=False,
        **kwargs,
    )
    content = response.choices[0].message.content or ""
    return {"content": content}