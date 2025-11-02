import os
from typing import Iterator, List, Optional, Tuple

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
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
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
            content = assistant_msgs[idx]
        except IndexError:
            raise IndexError("idx out of range for assistant messages.")

        file_path = os.path.join(OBSIDIAN_VAULT, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        os.makedirs("markdown", exist_ok=True)
        with open(os.path.join("markdown", filename), "w", encoding="utf-8") as f:
            f.write(content)

    def api_query(
        self, model: str, user_message: str, system_prompt: str, stream: bool, chat_history: Optional[List[Tuple[str, str]]]
    ) -> Iterator[str]:
        """Make a single API query to the LLM."""
        if model in MODELS_OPENAI:
            messages = (
                ([{"role": "system", "content": system_prompt}])
                + [{"role": role, "content": msg} for role, msg in chat_history or []]
                + [{"role": "user", "content": user_message}]
            )
            if stream:
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
            else:
                resp = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                response = resp.choices[0].message.content
                yield response

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
            if stream:
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
            else:
                resp = self.gemini_client.models.generate_content(
                    model=model,
                    config=types.GenerateContentConfig(system_instruction=system_prompt, top_p=0.96, temperature=0.2),
                    contents=contents,
                )
                response = "".join(part.text for chunk in resp.chunks for part in chunk.parts if part.text)
                yield response

        return response

    def chat(self, model: str, user_message: str) -> Iterator[str]:
        self.messages.append(("user", user_message))
        response = yield from self.api_query(model=model, user_messages=user_message, stream=True)
        self.messages.append(("assistant", response))
