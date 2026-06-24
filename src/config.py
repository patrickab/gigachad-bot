import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(".")

# Fileserver data location
SERVER_STATIC_DIR = BASE_DIR / "src" / "static"

DIRECTORY_CLOUD = Path("~/Nextcloud/linux").expanduser()

# Obsidian vault roots are NOT configured here. They live in
# chat_histories/obsidian-roots.json (the single source of truth), managed at
# runtime by ObsidianVault — see src/lib/obsidian_vault.py.
DIRECTORY_CHAT_HISTORIES = BASE_DIR / "chat_histories"

# Uploads directory for non-project chats. Project-scoped uploads live under
# DIRECTORY_CHAT_HISTORIES / <slug> / "_uploads". Leading underscore keeps the
# folder visually distinct from chat JSON files when browsing with `ls`.
DIRECTORY_CHAT_UPLOADS = DIRECTORY_CHAT_HISTORIES / "_uploads"

# --- Application-wide small/fast model defaults ---
# Used for lightweight tasks like query expansion, where speed matters more than raw capability.
# Provider-prefixed models route through LiteLLM; unprefixed models are treated as Ollama by callers.
SMALL_MODEL = "ollama/gemma4:31b-cloud"
MEMORY_MODEL = "gemini/gemini-3.1-flash-lite"
VISION_MODEL = "ollama/qwen3-vl:235b-instruct-cloud"

# --- MinerU PDF parsing config ---
DIRECTORY_OUTPUT_MINERU = DIRECTORY_CLOUD / "Documents" / "Mineru"
DIRECTORY_OUTPUT_PDF = DIRECTORY_CLOUD / "Documents" / "PDFs"

# Cloud collection of user-created documents, mirrored on save (filename = identity,
# overwritten on conflict). Per-chat _uploads copies are independent of these.
DIRECTORY_OUTPUT_MARKDOWN = DIRECTORY_CLOUD / "Documents" / "Markdown"
DIRECTORY_OUTPUT_LATEX = DIRECTORY_CLOUD / "Documents" / "LaTeX"
DIRECTORY_OUTPUT_DRAWINGS = DIRECTORY_CLOUD / "Documents" / "Drawings"

# Vane (Perplexica) web-search sidecar. Single container, SearXNG bundled internally.
VANE_URL = os.environ.get("VANE_URL", "http://localhost:3001")
# Embedding model Vane uses to rerank sources. Must also be configured in Vane's
# provider settings (bge-m3 served by local Ollama).
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "ollama/bge-m3:latest")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# Standalone SearXNG (searxng-settings.yml) — retriever for Deep Research (gpt-researcher).
# Separate from Vane's internal SearXNG, which is not exposed on a host port.
SEARX_URL = os.environ.get("SEARX_URL", "http://localhost:8888")


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
        SERVER_STATIC_DIR,
        DIRECTORY_CHAT_HISTORIES,
        DIRECTORY_OUTPUT_MINERU,
        DIRECTORY_OUTPUT_MINERU / "images",
        DIRECTORY_OUTPUT_PDF,
        DIRECTORY_OUTPUT_MARKDOWN,
        DIRECTORY_OUTPUT_LATEX,
        DIRECTORY_OUTPUT_DRAWINGS,
        DIRECTORY_CHAT_UPLOADS,
        DIRECTORY_CHAT_HISTORIES / "memory",
        DIRECTORY_CHAT_HISTORIES / "memory" / "pending",
    ]
    for d in _dirs:
        d.mkdir(parents=True, exist_ok=True)
