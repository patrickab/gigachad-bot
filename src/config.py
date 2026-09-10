import os
from pathlib import Path
import shutil
import sys

from lib.data_store import DataStore, LocalDataStore

# All mutable application data belongs to the backend's storage root.
# ponytail: one shared workspace for now; user-specific roots can come later.
REMOTE_ROOT = Path("~/Nextcloud/linux").expanduser()
DOCUMENTS = REMOTE_ROOT / "Documents"
DIRECTORY_CLOUD = REMOTE_ROOT  # legacy alias, only src/legacy_streamlit still imports it

# File-vault roots are NOT configured here. They live in
# chat_histories/file-vault-roots.json (the single source of truth), managed at
# runtime by FileVault — see src/lib/file_vault.py.
DIRECTORY_CHAT_HISTORIES = DOCUMENTS / "chat_history"
DIRECTORY_PROMPTS = DOCUMENTS / "Prompts"

# Uploads directory for non-project chats. Project-scoped uploads live under
# DIRECTORY_CHAT_HISTORIES / <slug> / "_uploads". Leading underscore keeps the
# folder visually distinct from chat JSON files when browsing with `ls`.
DIRECTORY_CHAT_UPLOADS = DIRECTORY_CHAT_HISTORIES / "_uploads"

# Non-project canvases/notes. Leading underscore mirrors DIRECTORY_CHAT_UPLOADS
# so it stays visually distinct from chat JSON files when browsing with `ls`.
DIRECTORY_NOTES = DIRECTORY_CHAT_HISTORIES / "_notes"

# --- Application-wide small/fast model defaults ---
# Used for lightweight tasks like query expansion, where speed matters more than raw capability.
# Provider-prefixed models route through LiteLLM; unprefixed models are treated as Ollama by callers.
SMALL_MODEL = "ollama/gemma4:31b-cloud"
MEMORY_MODEL = "gemini/gemini-3.1-flash-lite"
VISION_MODEL = "ollama/gemma4:31b-cloud"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_DOWNSCALE_IMAGES = True

# --- MinerU PDF parsing config ---
DIRECTORY_OUTPUT_MINERU = DOCUMENTS / "Mineru"
DIRECTORY_OUTPUT_PDF = DOCUMENTS / "PDFs"

# Cloud collection of user-created documents, mirrored on save (filename = identity,
# overwritten on conflict). Per-chat _uploads copies are independent of these.
DIRECTORY_OUTPUT_MARKDOWN = DOCUMENTS / "Markdown"
DIRECTORY_OUTPUT_LATEX = DOCUMENTS / "LaTeX"
DIRECTORY_OUTPUT_DRAWINGS = DOCUMENTS / "Drawings"
# Canonical, live Architecture Graph documents. Project/canvas/chat features
# reference files here rather than copying graph state into their own stores.
DIRECTORY_OUTPUT_ARCHITECTURE_GRAPHS = DOCUMENTS / "Architecture_Graphs"

_data_store: DataStore | None = None


def get_data_store() -> DataStore:
    """Use authoritative local storage; Nextcloud sync runs independently."""
    global _data_store
    if _data_store is None:
        # ponytail: always use local storage; retain optional WebDAV selection below.
        _data_store = LocalDataStore(DOCUMENTS)
        # from lib.data_store import WebDavDataStore
        # if WebDavDataStore.configured():
        #     _data_store = WebDavDataStore.from_environment(fallback=_data_store)
    return _data_store


def get_model_defaults() -> dict[str, str]:
    """Return the editable model defaults, falling back to this module's constants."""
    from lib.model_provider_store import ModelProviderStore

    return ModelProviderStore(get_data_store()).load_defaults()


# Vane (Perplexica) web-search sidecar. Single container, SearXNG bundled internally.
VANE_URL = os.environ.get("VANE_URL", "http://localhost:3001")
# Embedding model Vane uses to rerank sources. Must also be configured in Vane's
# provider settings (bge-m3 served by local Ollama).
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "ollama/bge-m3:latest")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# Standalone SearXNG (searxng-settings.yml) — retriever for Deep Research (gpt-researcher).
# Separate from Vane's internal SearXNG, which is not exposed on a host port.
SEARX_URL = os.environ.get("SEARX_URL", "http://localhost:8888")

# External MinerU OCR server (mineru.cli.fast_api). When unset, the backend
# spawns one per parse from its own environment — impossible in the frozen
# desktop sidecar, which excludes the ML stack and requires this to be set.
MINERU_SERVER_URL = os.environ.get("MINERU_SERVER_URL")


def uploads_dir_for(slug: str | None) -> Path:
    """Resolve the uploads directory for a given project slug (or None for non-project)."""
    if slug:
        return DIRECTORY_CHAT_HISTORIES / slug / "_uploads"
    return DIRECTORY_CHAT_UPLOADS


def chat_upload_dir(chat_id: str, slug: str | None = None) -> Path:
    """Resolve the per-chat upload directory."""
    return uploads_dir_for(slug) / chat_id


def ensure_directories() -> None:
    """Create all config-defined directories that the application needs at startup."""
    _dirs = [
        DIRECTORY_CHAT_HISTORIES,
        DIRECTORY_OUTPUT_MINERU,
        DIRECTORY_OUTPUT_MINERU / "images",
        DIRECTORY_OUTPUT_PDF,
        DIRECTORY_OUTPUT_MARKDOWN,
        DIRECTORY_OUTPUT_LATEX,
        DIRECTORY_OUTPUT_DRAWINGS,
        DIRECTORY_OUTPUT_ARCHITECTURE_GRAPHS,
        DIRECTORY_OUTPUT_ARCHITECTURE_GRAPHS / ".drafts",
        DIRECTORY_CHAT_UPLOADS,
        DIRECTORY_NOTES,
        DIRECTORY_CHAT_HISTORIES / "memory",
        DIRECTORY_CHAT_HISTORIES / "memory" / "pending",
        # DIRECTORY_PROMPTS is deliberately absent: seed_prompts() copies into it and
        # skips a directory that already exists.
    ]
    store = get_data_store()
    for directory in _dirs:
        store.mkdir(directory.relative_to(DOCUMENTS).as_posix())
    seed_prompts(store)


def seed_prompts(store: DataStore | None = None) -> None:
    """Initialize editable prompts once from the shipped defaults.

    An existing directory is authoritative, including deleted prompts. Defaults
    remain source assets; all subsequent editor writes go to remote storage.
    """
    source = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)) / "prompts"
    if not source.is_dir():
        return
    if store is not None:
        # The presence of the directory is authoritative, including an empty
        # one where a user intentionally deleted every prompt.
        if store.exists("Prompts"):
            return
        store.mkdir("Prompts")
        for path in source.rglob("*"):
            relative = path.relative_to(source).as_posix()
            key = f"Prompts/{relative}"
            if path.is_dir():
                store.mkdir(key)
            else:
                store.write_bytes(key, path.read_bytes())
        return
    if DIRECTORY_PROMPTS.exists():
        return
    # Stage then rename: a copy interrupted half-way must not leave a directory
    # that exists (so seeding never runs again) but is missing prompts.
    staged = DIRECTORY_PROMPTS.with_name(DIRECTORY_PROMPTS.name + ".seeding")
    shutil.rmtree(staged, ignore_errors=True)
    shutil.copytree(source, staged)
    staged.rename(DIRECTORY_PROMPTS)
