import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(".")

# Fileserver data location
SERVER_STATIC_DIR = BASE_DIR / "src" / "static"

DIRECTORY_CLOUD = Path("~/Nextcloud/linux").expanduser()

# Set to empty string if you dont use Obsidian
# Note: You can sync obsidian with Nextcloud, Dropbox, etc. to enable cloud integration of your notes.
DIRECTORY_OBSIDIAN_VAULT = DIRECTORY_CLOUD / "obsidian"
DIRECTORY_CHAT_HISTORIES = BASE_DIR / "chat_histories"

# --- Application-wide small/fast model defaults ---
# Used for lightweight tasks like query expansion, where speed matters more than raw capability.
# Should point to a small local model or a fast cloud model.
SMALL_MODEL = "ollama/gemma4:31b-cloud"

# --- MinerU PDF parsing config ---
DIRECTORY_OUTPUT_MINERU = DIRECTORY_CLOUD / "Documents" / "Mineru"
DIRECTORY_OUTPUT_PDF = DIRECTORY_CLOUD / "Documents" / "PDFs"

DIRECTORY_CHAT_UPLOADS = DIRECTORY_CLOUD / "Documents" / "ChatUploads"
MORPHIC_URL = os.environ.get("MORPHIC_URL", "http://localhost:3001")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
SEARX_URL = os.environ.get("SEARX_URL", "http://localhost:8888")


def ensure_directories() -> None:
    """Create all config-defined directories that the application needs at startup."""
    _dirs = [
        SERVER_STATIC_DIR,
        DIRECTORY_CHAT_HISTORIES,
        DIRECTORY_OUTPUT_MINERU,
        DIRECTORY_OUTPUT_MINERU / "images",
        DIRECTORY_OUTPUT_PDF,
        DIRECTORY_CHAT_UPLOADS,
    ]
    for d in _dirs:
        d.mkdir(parents=True, exist_ok=True)
