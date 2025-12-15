import os
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
    "ollama/gemini-3-pro-preview",
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

HUGGINGFACE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
MODELS_VLLM = os.listdir(HUGGINGFACE_DIR) if os.path.exists(HUGGINGFACE_DIR) else []
MODELS_VLLM = [model.replace("models--", "", 1) for model in MODELS_VLLM if model.startswith("models--")]
MODELS_VLLM = [model.replace("--", "/", 1) for model in MODELS_VLLM]
MODELS_VLLM = [f"hosted_vllm/{model}" for model in MODELS_VLLM]

ministral_14b_quantized = "hosted_vllm/cyankiwi/Ministral-3-14B-Instruct-2512-AWQ-4bit"
VLLM_STARTUP_COMMANDS = {
    ministral_14b_quantized:
        f"vllm serve {ministral_14b_quantized.replace('hosted_vllm/', '')} " +
        "--port 8000 --gpu-memory-utilization 0.95 --max-model-len 2800 " +
        "--max-num-seqs 1 --enforce-eager"
}

# Expects API-Keys in environment variables & Huggingface tokens for tokenizer
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
RAG_K_DOCS = 5

### OCR related config
MODELS_OCR_OLLAMA = ["qwen3-vl:2b", "qwen3-vl:8b"]
