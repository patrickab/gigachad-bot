import os
from typing import Iterator, List, Optional, Tuple

from google.genai import Client as GeminiClient
from google.genai import types
from openai import OpenAI as OpenAIClient

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MACROTASK_MODEL = "gemini-2.5-pro"
MICROTASK_MODEL = "gemini-2.5-flash"
NANOTASK_MODEL = "gemini-2.5-flash-lite"

MODELS_GEMINI = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]
MODELS_OPENAI = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-4o",
]

OBSIDIAN_VAULT = "/home/noob/Nextcloud/obsidian"


class LLMClient:
    """Base client for LLM chat completions."""

    def __init__(self) -> None:
        self.messages: List[Tuple[str, str]] = []  # [(role, message)]
        self.sys_prompt: str = ""

        if OPENAI_API_KEY is not None:
            self.openai_client = OpenAIClient(api_key=OPENAI_API_KEY)

        if GEMINI_API_KEY is not None:
            self.gemini_client = GeminiClient(api_key=GEMINI_API_KEY)

    def _set_system_prompt(self, system_prompt: str) -> None:
        self.sys_prompt = system_prompt

    def reset_history(self) -> None:
        """Reset the chat history."""
        self.messages = []

    def api_query(
        self, model: str, user_message: str, system_prompt: str, chat_history: Optional[List[Tuple[str, str]]]
    ) -> Iterator[str] | str:
        """
        Make a single API query to the LLM.
        Supports both Gemini & OpenAI models.
        """
        if model in MODELS_OPENAI:
            messages = (
                ([{"role": "system", "content": system_prompt}])
                + [{"role": role, "content": msg} for role, msg in chat_history or []]
                + [{"role": "user", "content": user_message}]
            )
            stream = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            response = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    response += content
                    yield content

        if model in MODELS_GEMINI:
            contents = [
                {
                    "role": ("model" if role == "assistant" else "user"),
                    "parts": [{"text": msg}],
                }
                for role, msg in chat_history or []
            ] + [
                {
                    "role": "user",
                    "parts": [{"text": user_message}],
                }
            ]
            stream_resp = self.gemini_client.models.generate_content_stream(
                model=model,
                config=types.GenerateContentConfig(system_instruction=system_prompt, top_p=0.96, temperature=0.2),
                contents=contents,
            )
            response = ""
            for chunk in stream_resp:
                if chunk.parts:
                    for part in chunk.parts:
                        if part.text:
                            response += part.text
                            yield part.text

    def chat(self, model: str, user_message: str) -> Iterator[str]:
        self.messages.append(("user", user_message))
        response = ""
        for chunk in self.api_query(model=model, user_message=user_message, system_prompt=self.sys_prompt, chat_history=self.messages):
            response += chunk
            yield chunk
        self.messages.append(("assistant", response))
