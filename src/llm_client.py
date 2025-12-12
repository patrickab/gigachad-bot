import csv
import os
from typing import List, Tuple

from llm_baseclient.client import LLMClient
from streamlit_paste_button import PasteResult

EMPTY_PASTE_RESULT = PasteResult(image_data=None)


class LLMClient(LLMClient):
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
        self.messages: List[Tuple[str, str]] = [] # [role, message] - only store text for efficiency
        self.sys_prompt = ""

    # -------------------------------- Streamlit State Management -------------------------------- #
    def store_history(self, filename: str) -> None:
        """Store message history to filesytem."""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['role', 'message'])
            for msg in self.messages:
                writer.writerow([msg["role"], msg["content"]])

    def load_history(self, filename: str) -> None:
        """Load message history from filesystem."""
        if not os.path.exists(filename):
            return

        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            messages = [{"role": row["role"], "content": row["message"]} for row in reader]
            self.messages = messages

    def reset_history(self) -> None:
        self.messages = []
