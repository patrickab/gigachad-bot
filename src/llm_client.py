from typing import Any, Dict, Iterator, List, Union

from litellm.types.utils import ModelResponse
from llm_baseclient.client import LLMClient
from openai.types.chat import ChatCompletion

from llm_config import MODEL_CONFIGS


class LLMClient(LLMClient):
    """
    LLM Client supporting
       - Open source: vLLM / Ollama / Huggingface with local CPU/GPU inference.
       - Commercial: OpenAI, Gemini, Anthropic, etc - any provider supported by LiteLLM.

    Kept as module inside this project for future extensibility.

    Automatically spawns & manages local inference servers (vLLM, Ollama).
    Allowing dynamic switching between backends/models during runtime.

    Assumes:
        For Commercial Providers:
            - API keys in environment.
        For vLLM / Ollama:
            - Local provider software installed.
            - Local models already downloaded.

    Supports:
        - stateless & stateful interactions
        - streaming & non-streaming responses
        - text-only & multimodal inputs
        - images provided as
            (1) file paths
            (2) raw bytes
            (3) base64 data URIs
            (4) web URLs
    """

    def __init__(self, model_configs: Dict[str, Any] | None = None) -> None:
        super().__init__(model_configs=model_configs or MODEL_CONFIGS)

    def chat(self, model: str, **kwargs: Dict[str, Any]) -> Iterator[str] | ChatCompletion:
        return super().chat(model=model, **kwargs)

    def api_query(self, model: str, **kwargs: Dict[str, Any]) -> Iterator[str] | ChatCompletion:
        return super().api_query(model=model, **kwargs)

    def batch_api_query(self, model: str, **kwargs: Dict[str, Any]) -> List[Union[ModelResponse, Exception]]:
        return super().batch_api_query(model=model, **kwargs)
