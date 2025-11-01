import os
from typing import List, Tuple

from google.genai import Client as GeminiClient
from google.genai import types
from openai import OpenAI as OpenAIClient

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODELS_OPENAI = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-4.1-mini",
    "gpt-4o",
]
MODELS_GEMINI = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

OBSIDIAN_VAULT = "/home/noob/Nextcloud/obsidian"

class LLMClient:
    """Base client for LLM chat completions."""

    def __init__(self) -> None:
        self.messages: List[Tuple[str, str]] = []  # [(role, message)]
        self.system_prompt: str = ""

        if OPENAI_API_KEY is not None:
            self.openai_client = OpenAIClient(api_key=OPENAI_API_KEY)

        if GEMINI_API_KEY is not None:
            self.gemini_client = GeminiClient(api_key=GEMINI_API_KEY)

    def _add_user_message(self, content: str) -> None:
        self.messages.append(("user", content))

    def _add_assistant_message(self, content: str) -> None:
        self.messages.append(("assistant", content))

    def _set_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    def reset_history(self) -> None:
        """Reset the chat history."""
        self.messages = []

    def write_to_md(self, filename: str, idx: int) -> None:
        """Write an assistant response to .md (idx: 0 = most recent)."""
        if not filename.endswith(".md"):
            filename += ".md"

        assistant_msgs = [msg for role, msg in self.messages if role == "assistant"]
        try:
            content = assistant_msgs[::-1][idx]
        except IndexError:
            raise IndexError("idx out of range for assistant messages.")

        file_path = os.path.join(OBSIDIAN_VAULT, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        os.makedirs("markdown", exist_ok=True)
        with open(os.path.join("markdown", filename), "w", encoding="utf-8") as f:
            f.write(content)

    def chat(self, model: str, user_message: str) -> str:
        self._add_user_message(user_message)

        if model in MODELS_OPENAI:
            messages = (
                [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
            ) + [{"role": r, "content": m} for r, m in self.messages]

            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
            )
            response = response.choices[0].message.content

        if model in MODELS_GEMINI:
            contents = [
                {
                    "role": ("model" if role == "assistant" else "user"),
                    "parts": [{"text": msg}],
                }
                for role, msg in self.messages
            ]

            response = self.gemini_client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(system_instruction=self.system_prompt, top_p=0.94, temperature=0.2),
                contents=contents,
            )
            response = getattr(response, "text", None)

        self._add_assistant_message(response)
        return response
