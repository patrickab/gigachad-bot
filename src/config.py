import subprocess

# Set to empty string if you dont use Obsidian
# Note: You can sync obsidian with Nextcloud, Dropbox, etc. to enable cloud integration of your notes.
OBSIDIAN_VAULT = "/home/noob/Nextcloud/obsidian"
CHAT_HISTORY_FOLDER = "./chat_histories"

# Adjust to your preferred models
MACROTASK_MODEL = "gemini-2.5-pro"
MICROTASK_MODEL = "gemini-2.5-flash"
NANOTASK_MODEL = "gemini-2.5-flash"

MODELS_GEMINI = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]
MODELS_OPENAI = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-4o",
]

MODELS_OLLAMA = []
try:  # if ollama is available, add ollama models
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
    MODELS_OLLAMA += [line.split()[0] for line in result.stdout.strip().splitlines()[1:]]
except (FileNotFoundError, subprocess.CalledProcessError):
    pass  # ollama not available or command failed
