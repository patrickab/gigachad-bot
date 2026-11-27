import os
from pathlib import Path

# All mutable application data belongs to the backend's storage root.
# ponytail: one shared workspace for now; user-specific roots can come later.
REMOTE_ROOT = Path(os.environ.get("GIGACHAD_BASE_DIR", "~/Nextcloud/linux")).expanduser()
DOCUMENTS = REMOTE_ROOT / "Documents"
# File-first workbench documents (Markdown, LaTeX) and the Pi session directory of each kind, `lib.workbench`.
WORKBENCH = REMOTE_ROOT / "workbench"
# Shipped prompt and template sources (`templates/`).
TEMPLATES = Path(__file__).resolve().parents[1] / "templates"

# Browser origins allowed to call the API. Defaults to the wildcard; server deployments narrow it.
CORS_ORIGINS = [origin.strip() for origin in os.environ.get("GIGACHAD_CORS_ORIGINS", "*").split(",") if origin.strip()]


def is_production() -> bool:
    """Whether this process runs as the production deployment."""
    return os.environ.get("GIGACHAD_ENV") == "production"


def is_development() -> bool:
    """Whether this process runs as a local development checkout."""
    return os.environ.get("GIGACHAD_ENV") == "development"


def is_cors_unsafe_for_production() -> bool:
    """Whether production would let any web page call the API: identity comes from the tailnet, not a cookie."""
    return is_production() and (not CORS_ORIGINS or "*" in CORS_ORIGINS)


# --- Model configuration ---
# The user-editable defaults in `model-defaults.yaml` are the single source of truth for every
# model the app uses. Nothing here may name a production model.
# A model is a stored selector `<provider>/<model>`, answered by pi-host's gateway (`lib/completions.py`).
MODEL_DEFAULT_KEYS = ("default_model", "small_model", "vision_model", "sandbox_model")
# Every test runs against this model, and it seeds a store that has no defaults yet.
TEST_MODEL = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# External MinerU OCR server (mineru.cli.fast_api). When unset, the backend
# spawns one per parse from its own environment.
MINERU_SERVER_URL = os.environ.get("MINERU_SERVER_URL")
