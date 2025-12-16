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

def vllm_cmd(model:str, max_tokens:int) -> str:
    """Efficiency optimizations to fit large models into small GPU"""
    if model==qwen_coder_14b_bnb_4bit:
        try:
            import bitsandbytes # noqa
        except ImportError:
            subprocess.run(["uv", "pip", "install", "bitsandbytes>=0.46.1"], check=True) # todo: find more elegant way

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    return (f"vllm serve {model.replace('hosted_vllm/', '')} " +
        f"--port 8000 --gpu-memory-utilization 0.95 --max-model-len {max_tokens} " +
        "--max-num-seqs 1 --enforce-eager")


ministral_14b_awq_4bit = "hosted_vllm/cyankiwi/Ministral-3-14B-Instruct-2512-AWQ-4bit"
qwen_coder_14b_bnb_4bit = "unsloth/Qwen2.5-Coder-14B-bnb-4bit"

VLLM_CONFIG = {
    ministral_14b_awq_4bit: vllm_cmd(model=ministral_14b_awq_4bit, max_tokens=2800),
    qwen_coder_14b_bnb_4bit: vllm_cmd(model=qwen_coder_14b_bnb_4bit, max_tokens=100)
}

DIRECTORY_TABBY = os.path.join(os.path.expanduser("~"), "tabbyAPI")
MODELS_EXLLAMA = os.listdir(os.path.join(DIRECTORY_TABBY, "models"))
MODELS_EXLLAMA = [f"tabby/{model}" for model in MODELS_EXLLAMA]
MODELS_EXLLAMA.remove("tabby/place_your_models_here.txt")

qwen_coder_14b_exl2 = "Qwen2.5-Coder-14B-Instruct-exl2"
EXLLAMA_CONFIG = {
    qwen_coder_14b_exl2: {"max_seq_len": 8192, "cache_mode": "Q4"}
}

# Expects API-Keys in environment variables & Huggingface tokens for tokenizer
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
RAG_K_DOCS = 5

### OCR related config
MODELS_OCR_OLLAMA = ["qwen3-vl:2b", "qwen3-vl:8b"]
