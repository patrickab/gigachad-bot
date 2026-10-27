import os
from pathlib import Path
import sys

from psycopg_pool import ConnectionPool

from lib.data_store import DataStore
from lib.db_schema import database_url
from lib.storage_namespace import PROMPT

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

_postgres_pool: ConnectionPool | None = None


def get_postgres_pool() -> ConnectionPool:
    """Return the process-wide connection pool for Postgres storage."""
    global _postgres_pool
    if _postgres_pool is None:
        _postgres_pool = ConnectionPool(
            database_url(),
            min_size=int(os.environ.get("GIGACHAD_PG_POOL_MIN_SIZE", "1")),
            max_size=int(os.environ.get("GIGACHAD_PG_POOL_MAX_SIZE", "10")),
        )
    return _postgres_pool


def close_postgres_pool() -> None:
    """Release database connections during backend shutdown."""
    global _postgres_pool
    if _postgres_pool is not None:
        _postgres_pool.close()
        _postgres_pool = None


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# External MinerU OCR server (mineru.cli.fast_api). When unset, the backend
# spawns one per parse from its own environment. The frozen desktop sidecar
# excludes the ML stack, so there it must be set.
MINERU_SERVER_URL = os.environ.get("MINERU_SERVER_URL")


def seed_prompts(store: DataStore, *, prefix: str = PROMPT) -> None:
    """Initialize editable prompts once from the shipped defaults.

    An existing collection is authoritative, including deleted prompts. Defaults
    remain source assets; all subsequent editor writes go to persistent storage.
    """
    source = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)) / "prompts"
    if not source.is_dir() or store.exists(prefix):
        return
    store.mkdir(prefix)
    for path in source.rglob("*"):
        relative = path.relative_to(source).as_posix()
        key = f"{prefix}/{relative}"
        if path.is_dir():
            store.mkdir(key)
        else:
            store.write_bytes(key, path.read_bytes())
