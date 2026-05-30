import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(".")

# Fileserver data location
SERVER_STATIC_DIR = BASE_DIR / "src" / "static"

# Set to empty string if you dont use Obsidian
# Note: You can sync obsidian with Nextcloud, Dropbox, etc. to enable cloud integration of your notes.
DIRECTORY_OBSIDIAN_VAULT = Path("/home/noob/Nextcloud/obsidian")
DIRECTORY_CHAT_HISTORIES = BASE_DIR / "chat_histories"

# --- Application-wide small/fast model defaults ---
# Used for lightweight tasks like query expansion, where speed matters more than raw capability.
# Should point to a small local model or a fast cloud model.
SMALL_MODEL = "ollama/gemma4:31b-cloud"

MORPHIC_URL = os.environ.get("MORPHIC_URL", "http://localhost:3001")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
