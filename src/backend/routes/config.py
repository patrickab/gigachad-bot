from fastapi import APIRouter, Depends

from backend.identity import RequestIdentity, get_request_identity
from config import DEFAULT_DOWNSCALE_IMAGES
from lib import omp_web
from lib.model_provider_store import ModelProviderStore
from lib.neovim_session import is_neovim_installed

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
        "host_access": identity.has_host_access,
        "neovim_available": identity.has_host_access and is_neovim_installed(),
        "omp_web_url": omp_web.omp_web_frame_url() if identity.has_host_access else None,
    }
