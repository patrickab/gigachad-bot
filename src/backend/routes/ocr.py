from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from lib.llm_resilience import api_query_resilient
from lib.model_provider_store import ModelProviderStore
from lib.prompts.internal import SYS_OCR_TEXT_EXTRACTION

from .deps import decode_image, get_model_provider_store, request_client, sse_event_stream

router = APIRouter(prefix="/api", tags=["ocr"])


class OCRRequest(BaseModel):
    img_base64: str
    model: str = ""


@router.post("/ocr")
async def ocr(req: OCRRequest, store: ModelProviderStore = Depends(get_model_provider_store)) -> EventSourceResponse:
    with request_client() as c:
        model = req.model or store.load_defaults()["vision_model"]
        img = decode_image(req.img_base64)
        chunks = await run_in_threadpool(
            api_query_resilient,
            c,
            model=model,
            user_msg="Extract all text and LaTeX from this image.",
            system_prompt=SYS_OCR_TEXT_EXTRACTION,
            img=img,
            temperature=0.1,
            stream=True,
        )
        return sse_event_stream(chunks)
