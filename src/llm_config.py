import subprocess

# Adjust to your preferred models
MACROTASK_MODEL = "gemini-2.5-pro"
MICROTASK_MODEL = "gemini-2.5-flash"
NANOTASK_MODEL = "gemini-2.5-flash"

# Specify local ollama model to use additional features like captioning of user prompts.
# Not strictly necessary, enhances user experience slightly. Will be ignored if ollama is not installed.
# Granite 4: Minimal hardware, very fast - can be run on any smartphone - competent for easy tasks.
LOCAL_NANOTASK_MODEL = "granite4:1b"

MODELS_GEMINI = [
    "gemini/gemini-2.5-flash-lite",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-pro",
]

MODELS_OPENAI = [
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-4o",
]

MODELS_OLLAMA = []
try:  # if ollama is available, add ollama models
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
    MODELS_OLLAMA += [f"ollama/{line.split()[0]}" for line in result.stdout.strip().splitlines()[1:]]
except (FileNotFoundError, subprocess.CalledProcessError):
    pass  # ollama not available or command failed

# Expects API-Keys in environment variables & Huggingface tokens for tokenizer
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
RAG_K_DOCS = 5

### OCR related config
MODELS_OCR_OLLAMA = ["qwen3-vl:2b", "qwen3-vl:8b"]