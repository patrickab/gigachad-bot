import os
from pathlib import Path
import sys
from uuid import UUID

from psycopg_pool import ConnectionPool

from lib.data_store import DataStore
from lib.storage_namespace import PROMPT

# All mutable application data belongs to the backend's storage root.
# ponytail: one shared workspace for now; user-specific roots can come later.
REMOTE_ROOT = Path(os.environ.get("GIGACHAD_BASE_DIR", "~/Nextcloud/linux")).expanduser()
DOCUMENTS = REMOTE_ROOT / "Documents"

# --- Model configuration ---
# The user-editable defaults in `model-defaults.yaml` are the single source of truth for every
# model the app uses. Nothing here may name a production model.
# Provider-prefixed models route through LiteLLM; unprefixed models are treated as Ollama by callers.
MODEL_DEFAULT_KEYS = ("default_model", "small_model", "vision_model", "memory_model", "omp_model")
# Every test runs against this model, and it seeds a store that has no defaults yet.
TEST_MODEL = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_DOWNSCALE_IMAGES = True

# --- MinerU PDF parsing config ---
# The only application artifacts still mirrored to Nextcloud: PostgreSQL is
# authoritative for both, this tree exists only so MinerU can parse a real file.
DIRECTORY_OUTPUT_MINERU = DOCUMENTS / "Mineru"
DIRECTORY_OUTPUT_PDF = DOCUMENTS / "PDFs"

_postgres_pool: ConnectionPool | None = None


def get_postgres_pool() -> ConnectionPool:
    """Return the process-wide connection pool for Postgres storage."""
    global _postgres_pool
    if _postgres_pool is None:
        try:
            database_url = os.environ["GIGACHAD_DATABASE_URL"]
        except KeyError as exc:
            raise RuntimeError("GIGACHAD_DATABASE_URL is required") from exc
        _postgres_pool = ConnectionPool(
            database_url,
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


def get_data_store(user_id: UUID, *, device_id: UUID | None = None) -> DataStore:
    """Return an immutable, user-scoped Postgres store."""
    from lib.postgres_data_store import PostgresDataStore

    return PostgresDataStore(get_postgres_pool(), user_id, device_id=device_id)


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# External MinerU OCR server (mineru.cli.fast_api). When unset, the backend
# spawns one per parse from its own environment — impossible in the frozen
# desktop sidecar, which excludes the ML stack and requires this to be set.
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
