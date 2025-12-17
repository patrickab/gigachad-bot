import os
import subprocess

# --- Constants ---
HOME = os.path.expanduser("~")
DIRECTORY_TABBY = os.path.join(HOME, "tabbyAPI")
HUGGINGFACE_DIR = os.path.join(HOME, ".cache", "huggingface", "hub")

# --- Model Defaults ---
NANOTASK_MODEL = "gemini/gemini-2.5-flash-lite"

# --- Static Model Definitions ---
MODELS_GEMINI = [
    "gemini/gemini-3-flash-preview",
    "gemini/gemini-2.5-flash-lite",
    "ollama/gemini-3-pro-preview",
]

MODELS_OPENAI = [
    "openai/gpt-5.1",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-4o",
]

# --- Dynamic Discovery: Ollama ---
MODELS_OLLAMA = []
try:
    res = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
    # Skip header, parse model names
    MODELS_OLLAMA = [f"ollama/{line.split()[0]}" for line in res.stdout.splitlines()[1:]]
except (FileNotFoundError, subprocess.CalledProcessError):
    pass  # Ollama unavailable

# --- Dynamic Discovery: VLLM (HuggingFace) ---
MODELS_VLLM = []

if os.path.exists(HUGGINGFACE_DIR):
    # Parse "models--author--model" -> "hosted_vllm/author/model"
    MODELS_VLLM = [
        f"hosted_vllm/{m.replace('models--', '', 1).replace('--', '/', 1)}"
        for m in os.listdir(HUGGINGFACE_DIR)
        if m.startswith("models--")
    ]


def vllm_cmd(model: str, max_tokens: int) -> str:
    """Efficiency optimizations to fit large models into small GPU."""
    # Runtime dependency check for specific quantized models
    if "bnb" in model:
        try:
            import bitsandbytes  # noqa
        except ImportError:
            # TODO: find more elegant way
            subprocess.run(["uv", "pip", "install", "bitsandbytes>=0.46.1"], check=True)

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Construct VLLM command with memory optimizations
    cmd = (
        f"vllm serve {model.replace('hosted_vllm/', '')} "
        f"--port 8000 --gpu-memory-utilization 0.95 --max-model-len {max_tokens} "
        "--max-num-seqs 1 --enforce-eager"
    )
    return cmd


ministral_14b_awq_4bit = "hosted_vllm/cyankiwi/Ministral-3-14B-Instruct-2512-AWQ-4bit"
qwen_coder_14b_bnb_4bit = "unsloth/Qwen2.5-Coder-14B-bnb-4bit"

# --- VLLM Configuration Map ---
VLLM_CONFIG = {
    ministral_14b_awq_4bit: vllm_cmd(model=ministral_14b_awq_4bit, max_tokens=2800),
    qwen_coder_14b_bnb_4bit: vllm_cmd(model=qwen_coder_14b_bnb_4bit, max_tokens=100),
}

# --- Dynamic Discovery: TabbyAPI ---
MODELS_EXLLAMA = []

if os.path.exists(os.path.join(DIRECTORY_TABBY, "models")):
    MODELS_EXLLAMA = [f"tabby/{m}" for m in os.listdir(os.path.join(DIRECTORY_TABBY, "models")) if m != "place_your_models_here.txt"]

qwen_coder_14b_exl2 = "Qwen2.5-Coder-14B-Instruct-exl2"
EXLLAMA_CONFIG = {f"tabby/{qwen_coder_14b_exl2}": {"max_seq_len": 16384, "cache_mode": "Q4"}}

# Expects API-Keys in environment variables & Huggingface tokens for tokenizer
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
RAG_K_DOCS = 5
