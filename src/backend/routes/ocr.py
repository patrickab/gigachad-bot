from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from lib.prompts import SYS_OCR_TEXT_EXTRACTION
from llm_config import DEFAULT_VISION_MODEL

from .deps import decode_image, request_client, sse_event_stream

router = APIRouter(prefix="/api", tags=["ocr"])


class OCRRequest(BaseModel):
    img_base64: str
    model: str = ""


class DownscaleRequest(BaseModel):
    img_base64: str
    max_tokens: int = 2048


@router.post("/ocr")
async def ocr(req: OCRRequest) -> EventSourceResponse:
    with request_client() as c:
        model = req.model or DEFAULT_VISION_MODEL
        img = decode_image(req.img_base64)
        chunks = c.api_query(
            model=model,
            user_msg="Extract all text and LaTeX from this image.",
            system_prompt=SYS_OCR_TEXT_EXTRACTION,
            img=img,
            temperature=0.1,
            stream=True,
        )
        return sse_event_stream(chunks)


@router.post("/downscale-image")
def downscale_image(req: DownscaleRequest) -> dict[str, str]:
    with request_client() as c:
        try:
            result = c.downscale_img(img=req.img_base64, max_tokens=req.max_tokens)
            return {"img_base64": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))