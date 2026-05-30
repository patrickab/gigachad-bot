import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import MORPHIC_URL

router = APIRouter(prefix="/api", tags=["morphic"])


class MorphicSearchRequest(BaseModel):
    query: str
    search_depth: str = "adaptive"
    model: str = ""


@router.post("/morphic-search")
async def morphic_search(req: MorphicSearchRequest) -> StreamingResponse:
    import httpx

    chat_id = str(uuid.uuid4())

    morphic_payload = {
        "message": {"role": "user", "parts": [{"type": "text", "text": req.query}]},
        "chatId": chat_id,
        "trigger": "submit-message",
        "isNewChat": True,
    }

    cookies: dict[str, str] = {"searchMode": req.search_depth}
    if req.model:
        provider_id = "ollama"
        model_id = req.model
        if model_id.startswith("ollama/"):
            model_id = model_id[len("ollama/"):]
        elif "/" in model_id:
            provider_id, model_id = model_id.split("/", 1)
        cookies["selectedModel"] = f"{provider_id}:{model_id}"

    async def event_stream():
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client, client.stream(
            "POST",
            f"{MORPHIC_URL.rstrip('/')}/api/chat",
            json=morphic_payload,
            headers={
                "Content-Type": "application/json",
                "Cookie": "; ".join(f"{k}={v}" for k, v in cookies.items()),
            },
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                yield f"event: error\ndata: {error_body.decode(errors='replace')}\n\n"
                return
            async for line in response.aiter_lines():
                if line.strip():
                    yield line + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )