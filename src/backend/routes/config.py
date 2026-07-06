from fastapi import APIRouter

from config import (
    DEFAULT_DOWNSCALE_IMAGES,
    DEFAULT_TEMPERATURE,
    MEMORY_MODEL,
    SMALL_MODEL,
    VISION_MODEL,
)

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/config")
async def get_config() -> dict:
    return {
        "default_model": SMALL_MODEL,
        "vision_model": VISION_MODEL,
        "memory_model": MEMORY_MODEL,
        "small_model": SMALL_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "downscale_images": DEFAULT_DOWNSCALE_IMAGES,
    }
