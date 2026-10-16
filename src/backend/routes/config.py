from fastapi import APIRouter, Depends

from config import DEFAULT_DOWNSCALE_IMAGES
from lib.model_provider_store import ModelProviderStore

from .deps import get_model_provider_store

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/config")
async def get_config(store: ModelProviderStore = Depends(get_model_provider_store)) -> dict:
    return {
        **store.load_defaults(),
        "downscale_images": DEFAULT_DOWNSCALE_IMAGES,
    }
