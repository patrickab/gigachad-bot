from fastapi import APIRouter

from config import (
    DEFAULT_DOWNSCALE_IMAGES,
    DEFAULT_TEMPERATURE,
    get_model_defaults,
)

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/config")
async def get_config() -> dict:
    models = get_model_defaults()
    return {
        **models,
        "temperature": DEFAULT_TEMPERATURE,
        "downscale_images": DEFAULT_DOWNSCALE_IMAGES,
    }
