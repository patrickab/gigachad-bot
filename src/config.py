import os
from pathlib import Path

# All mutable application data belongs to the backend's storage root.
# ponytail: one shared workspace for now; user-specific roots can come later.
REMOTE_ROOT = Path(os.environ.get("GIGACHAD_BASE_DIR", "~/Nextcloud/linux")).expanduser()
DOCUMENTS = REMOTE_ROOT / "Documents"

# Browser origins allowed to call the API. Defaults to the pre-desktop wildcard; the Tauri
# shell narrows it to its own webview origins, and server deployments can do the same.
CORS_ORIGINS = [origin.strip() for origin in os.environ.get("GIGACHAD_CORS_ORIGINS", "*").split(",") if origin.strip()]


def is_production() -> bool:
    """Whether this process runs as the production deployment."""
    return os.environ.get("GIGACHAD_ENV") == "production"


def is_development() -> bool:
    """Whether this process runs as a local development checkout."""
    return os.environ.get("GIGACHAD_ENV") == "development"


def env_flag(name: str, default: bool) -> bool:
    """Read a boolean env var: 1/true/yes and 0/false/no win in any case, anything else keeps *default*."""
    value = os.environ.get(name, "").lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    return default


def is_cors_unsafe_for_production() -> bool:
    """Whether production would let any web page call the API: identity comes from the tailnet, not a cookie."""
    return is_production() and (not CORS_ORIGINS or "*" in CORS_ORIGINS)


# --- Model configuration ---
# The user-editable defaults in `model-defaults.yaml` are the single source of truth for every
# model the app uses. Nothing here may name a production model.
# Provider-prefixed models route through LiteLLM; unprefixed models are treated as Ollama by callers.
MODEL_DEFAULT_KEYS = ("default_model", "small_model", "vision_model", "memory_model", "omp_model")
# Every test runs against this model, and it seeds a store that has no defaults yet.
TEST_MODEL = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_DOWNSCALE_IMAGES = True

# --- MinerU PDF parsing config ---
# The only application artifact still mirrored to Nextcloud: PostgreSQL is
# authoritative, this tree exists only so MinerU can parse a real file.
DIRECTORY_OUTPUT_MINERU = DOCUMENTS / "Mineru"

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# External MinerU OCR server (mineru.cli.fast_api). When unset, the backend
# spawns one per parse from its own environment. The frozen desktop sidecar
# excludes the ML stack, so there it must be set.
MINERU_SERVER_URL = os.environ.get("MINERU_SERVER_URL")
