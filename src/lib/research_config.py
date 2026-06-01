import json
import tempfile
from typing import Any

EMBEDDING_DEFAULT = "ollama:nomic-embed-text"
RETRIEVER_DEFAULT = "searx"


def build_research_config(
    fast_model: str,
    smart_model: str,
    strategic_model: str,
    depth: int = 2,
    breadth: int = 4,
    reasoning_effort: str | None = None,
    retriever: str = RETRIEVER_DEFAULT,
    embedding: str = EMBEDDING_DEFAULT,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "RETRIEVER": retriever,
        "EMBEDDING": embedding,
        "FAST_LLM": f"litellm:{fast_model}",
        "SMART_LLM": f"litellm:{smart_model}",
        "STRATEGIC_LLM": f"litellm:{strategic_model}",
        "DEEP_RESEARCH_DEPTH": depth,
        "DEEP_RESEARCH_BREADTH": breadth,
    }
    if reasoning_effort and reasoning_effort != "none":
        config["REASONING_EFFORT"] = reasoning_effort
    return config


def write_research_config(config: dict[str, Any]) -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="gpt-researcher-", delete=False
    )
    json.dump(config, tmp)
    tmp.close()
    return tmp.name