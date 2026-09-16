from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config import get_model_defaults
from lib.prompts.internal import SYS_OCR_TEXT_EXTRACTION

from lib.llm_resilience import api_query_resilient

from .deps import decode_image, request_client, sse_event_stream

router = APIRouter(prefix="/api", tags=["ocr"])


class OCRRequest(BaseModel):
    img_base64: str
    model: str = ""


@router.post("/ocr")
async def ocr(req: OCRRequest) -> EventSourceResponse:
    with request_client() as c:
        model = req.model or get_model_defaults()["vision_model"]
        img = decode_image(req.img_base64)
        chunks = api_query_resilient(
            c,
            model=model,
            user_msg="Extract all text and LaTeX from this image.",
            system_prompt=SYS_OCR_TEXT_EXTRACTION,
            img=img,
            temperature=0.1,
            stream=True,
        )
        return sse_event_stream(chunks)
