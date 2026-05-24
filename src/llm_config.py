from llm_baseclient.config import ModelConfigs

# --- Model Defaults ---
NANOTASK_MODEL = "ollama/devstral-2:123b-cloud"
DEFAULT_VISION_MODEL = "ollama/qwen3-vl:235b-instruct-cloud"

# --- Per-Model Configurations ---
MODEL_CONFIGS: ModelConfigs = {
    "exllama": {
        "tabby/Qwen2.5-Coder-14B-Instruct-exl2": {"max_seq_len": 16384, "cache_mode": "Q4"},
    },
}

# Expects API-Keys in environment variables & Huggingface tokens for tokenizer
DEFAULT_EMBEDDING_MODEL = "gemini/gemini-embedding-001"
RAG_K_DOCS = 5
