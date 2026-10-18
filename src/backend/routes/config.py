from fastapi import APIRouter, Depends

from backend.identity import RequestIdentity, get_request_identity
from config import DEFAULT_DOWNSCALE_IMAGES
from lib import omp_web
from lib.model_provider_store import ModelProviderStore
from lib.neovim_session import is_neovim_available

from .deps import get_model_provider_store

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/config")
async def get_config(
    store: ModelProviderStore = Depends(get_model_provider_store),
    identity: RequestIdentity = Depends(get_request_identity),
) -> dict:
    return {
        **store.load_defaults(),
        "downscale_images": DEFAULT_DOWNSCALE_IMAGES,
        "neovim_available": is_neovim_available(identity.login),
        "omp_web_url": omp_web.omp_web_url(),
    }
