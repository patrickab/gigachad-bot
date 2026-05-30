import base64
import re

from llm_client import LLMClient
from llm_config import MODEL_CONFIGS

_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient(model_configs=MODEL_CONFIGS)
    return _client


def shutdown_client() -> None:
    global _client
    if _client is not None:
        _client.kill_inference_engines()
        _client = None


def decode_image(base64_data: str | None) -> bytes | None:
    if not base64_data:
        return None
    match = re.match(r"data:image/\w+;base64,(.+)", base64_data)
    if match:
        return base64.b64decode(match.group(1))
    return base64.b64decode(base64_data)