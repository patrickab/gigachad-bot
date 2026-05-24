from pathlib import Path

# Base directory of the project
BASE_DIR = Path(".")

# Fileserver data location
SERVER_STATIC_DIR = BASE_DIR / "src" / "static"

# Set to empty string if you dont use Obsidian
# Note: You can sync obsidian with Nextcloud, Dropbox, etc. to enable cloud integration of your notes.
DIRECTORY_OBSIDIAN_VAULT = Path("/home/noob/Nextcloud/obsidian")
DIRECTORY_CHAT_HISTORIES = BASE_DIR / "chat_histories"
