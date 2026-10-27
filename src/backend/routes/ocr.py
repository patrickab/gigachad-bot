from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.http_plumbing import decode_image, stream_completion
from lib.model_provider_store import ModelProviderStore
from lib.prompts.non_user_prompts import SYS_OCR_TEXT_EXTRACTION

from .deps import get_model_provider_store, request_client

router = APIRouter(prefix="/api", tags=["ocr"])


class OCRRequest(BaseModel):
    img_base64: str
    model: str = ""


@router.post("/ocr")
async def ocr(req: OCRRequest, store: ModelProviderStore = Depends(get_model_provider_store)) -> EventSourceResponse:
    """Stream OCR text and LaTeX from an image as SSE, defaulting to the configured vision model."""
    with request_client() as c:
        return await stream_completion(
            c,
            model=req.model or store.load_defaults()["vision_model"],
            user_msg="Extract all text and LaTeX from this image.",
            system_prompt=SYS_OCR_TEXT_EXTRACTION,
            img=decode_image(req.img_base64),
        )
