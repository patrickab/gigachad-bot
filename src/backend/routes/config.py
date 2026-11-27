from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.identity import RequestIdentity, get_request_identity
from lib import learnharness
from lib.model_provider_store import ModelProviderStore
from lib.neovim_session import is_neovim_installed
from lib.user_preferences import UserPreferences

from .deps import get_model_provider_store, get_user_preferences

router = APIRouter(prefix="/api", tags=["config"])


# Sync handler: FastAPI runs it in its threadpool, so the store read never stalls the event loop.
@router.get("/config")
def get_config(
    store: ModelProviderStore = Depends(get_model_provider_store),
    identity: RequestIdentity = Depends(get_request_identity),
    preferences: UserPreferences = Depends(get_user_preferences),
) -> dict:
    return {
        **store.load_defaults(),
        "animated_background": preferences.animated_background(),
        "host_access": identity.has_host_access,
        "neovim_available": identity.has_host_access and is_neovim_installed(),
        "learnharness_url": learnharness.learnharness_frame_url() if identity.has_host_access else None,
    }


class AnimatedBackgroundRequest(BaseModel):
    enabled: bool


@router.put("/config/animated-background", status_code=204)
def set_animated_background(body: AnimatedBackgroundRequest, preferences: UserPreferences = Depends(get_user_preferences)) -> None:
    preferences.set_animated_background(body.enabled)
