import os
import subprocess
from pathlib import Path

# --- Constants ---
HOME = Path.home()
DIRECTORY_TABBY = HOME / "tabbyAPI"
HUGGINGFACE_DIR = HOME / ".cache" / "huggingface" / "hub"

# --- Model Defaults ---
NANOTASK_MODEL = "ollama/devstral-2:123b-cloud"
DEFAULT_VISION_MODEL = "ollama/qwen3-vl:235b-instruct-cloud"

# --- Static Model Definitions ---
MODELS_GEMINI = [
    "gemini/gemini-3-flash-preview",
    "gemini/gemini-3.1-flash-lite-preview",
    "gemini/gemini-3.1-pro-preview",
]

MODELS_OPENAI = [
    "openai/gpt-5.1",
    "openai/o1",
    "openai/gpt-5-mini",
    "openai/gpt-4o",
]

# --- Dynamic Discovery: Ollama ---
MODELS_OLLAMA = []
try:
    res = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
    # Skip header, parse model names
    MODELS_OLLAMA = [
        f"ollama/{line.split()[0]}" for line in res.stdout.splitlines()[1:]
    ]
except (FileNotFoundError, subprocess.CalledProcessError):
    pass  # Ollama unavailable

# --- Dynamic Discovery: VLLM (HuggingFace) ---
MODELS_VLLM = []

if HUGGINGFACE_DIR.exists():
    # Parse "models--author--model" -> "hosted_vllm/author/model"
    MODELS_VLLM = [
        f"hosted_vllm/{m.name.replace('models--', '', 1).replace('--', '/', 1)}"
        for m in HUGGINGFACE_DIR.iterdir()
        if m.name.startswith("models--")
    ]


# --- Dynamic Discovery: TabbyAPI ---
MODELS_EXLLAMA = []

tabby_models_dir = DIRECTORY_TABBY / "models"
if tabby_models_dir.exists():
    MODELS_EXLLAMA = [
        f"tabby/{m.name}"
        for m in tabby_models_dir.iterdir()
        if m.name != "place_your_models_here.txt"
    ]

qwen_coder_14b_exl2 = "Qwen2.5-Coder-14B-Instruct-exl2"
EXLLAMA_CONFIG = {
    f"tabby/{qwen_coder_14b_exl2}": {"max_seq_len": 16384, "cache_mode": "Q4"}
}

# Expects API-Keys in environment variables & Huggingface tokens for tokenizer
DEFAULT_EMBEDDING_MODEL = "gemini/gemini-embedding-001"
RAG_K_DOCS = 5
