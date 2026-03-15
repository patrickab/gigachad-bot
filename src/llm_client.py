import csv
import os
from typing import Any, Dict, Iterator, List, Union

from litellm.types.utils import ModelResponse
from llm_baseclient.client import LLMClient as BaseLLMClient
from openai.types.chat import ChatCompletion


class LLMClient(BaseLLMClient):
    """
    Custom Hardware-Aware LLM Client supporting
       - Open source: vLLM (GPU) / Ollama (CPU) / Huggingface.
       - Commercial: OpenAI, Google Gemini, Anthropic Claude.

    Supports:
        - stateless & stateful interactions
        - streaming & non-streaming responses
        - text-only & multimodal inputs
        - images provided as
            (1) file paths or
            (2) raw bytes

    Extended with Streamlit state management methods.
    """

    def __init__(self) -> None:
        super().__init__()
        self.messages: List[Dict[str, str]] = []  # [role, message] - only store text for efficiency
        self.sys_prompt = ""
        self._vllm_config = None
        self._exllama_config = None

    # -------------------------------- LLM Interaction -------------------------------- #
    def _load_vllm_config(self) -> Dict[str, List[str]]:
        """Lazy-load vLLM config only if needed (avoids importing torch/vllm unnecessarily)."""
        if self._vllm_config is None:
            try:
                from llm_config import VLLM_CONFIG
                self._vllm_config = VLLM_CONFIG
            except ImportError:
                self._vllm_config = {}
        return self._vllm_config

    def _load_exllama_config(self) -> Dict[str, Any]:
        """Lazy-load ExLlama config only if needed."""
        if self._exllama_config is None:
            try:
                from llm_config import EXLLAMA_CONFIG
                self._exllama_config = EXLLAMA_CONFIG
            except ImportError:
                self._exllama_config = {}
        return self._exllama_config

    def _apply_model_config(self, model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply model-specific configurations to kwargs, lazily loading heavy deps."""
        vllm_config = self._load_vllm_config()
        if model in vllm_config and "hosted_vllm/" in model:
            kwargs["vllm_cmd"] = vllm_config[model].split()

        exllama_config = self._load_exllama_config()
        if model in exllama_config and "tabby/" in model:
            kwargs["tabby_config"] = exllama_config[model]

        return kwargs

    def chat(self, model: str, **kwargs: Dict[str, Any]) -> Iterator[str] | ChatCompletion:
        """Overrides base chat method to add startup configs."""
        kwargs = self._apply_model_config(model, kwargs)
        return super().chat(model=model, **kwargs)

    def api_query(self, model: str, **kwargs: Dict[str, Any]) -> Iterator[str] | ChatCompletion:
        """Overrides base api_query to add startup configs."""
        kwargs = self._apply_model_config(model, kwargs)
        return super().api_query(model=model, **kwargs)

    def batch_api_query(self, model: str, **kwargs: Dict[str, Any]) -> List[Union[ModelResponse, Exception]]:
        """Overrides base batch_api_query to add startup configs."""
        kwargs = self._apply_model_config(model, kwargs)
        return super().batch_api_query(model=model, **kwargs)

    # -------------------------------- Streamlit State Management -------------------------------- #
    def store_history(self, filename: str) -> None:
        """Store message history to filesystem."""
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["role", "content"])
            for msg in self.messages:
                writer.writerow([msg["role"], msg["content"]])

    def load_history(self, filename: str) -> None:
        """Load message history from filesystem."""
        if not os.path.exists(filename):
            return

        with open(filename, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            messages = [{"role": row["role"], "content": row["content"]} for row in reader]
            self.messages = messages

    def reset_history(self) -> None:
        self.messages = []
