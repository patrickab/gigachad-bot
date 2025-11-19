import base64
import csv
from io import BytesIO
import os
from typing import Iterator, List, Optional, Tuple

from google.genai import Client as GeminiClient
from google.genai import types
from ollama import Client as OllamaClient
from openai import OpenAI as OpenAIClient
from streamlit_paste_button import PasteResult

from src.config import MODELS_GEMINI, MODELS_OLLAMA, MODELS_OPENAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class LLMClient:
    """Base client for LLM chat completions."""

    def __init__(self) -> None:
        self.messages: List[Tuple[str, str]] = []  # [(role, message)]
        self.sys_prompt: str = ""

        if OPENAI_API_KEY is not None:
            self.openai_client = OpenAIClient(api_key=OPENAI_API_KEY)

        if GEMINI_API_KEY is not None:
            self.gemini_client = GeminiClient(api_key=GEMINI_API_KEY)

        if MODELS_OLLAMA != []:
            self.ollama_client = OllamaClient(
                host="http://localhost:11434",
                headers={"x-some-header": "some-value"},
            )

    def _set_system_prompt(self, system_prompt: str) -> None:
        self.sys_prompt = system_prompt

    def store_history(self, filename: str) -> None:
        """Store message history to filesytem."""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['role', 'message'])
            writer.writerows(self.messages)

    def load_history(self, filename: str) -> None:
        """Load message history from filesystem."""
        if not os.path.exists(filename):
            return

        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            self.messages = [(row['role'], row['message']) for row in reader]

    def reset_history(self) -> None:
        """Reset the chat history."""
        self.messages = []

    def api_query(
        self, model: str,
        user_message: str,
        system_prompt: str,
        chat_history: Optional[List[Tuple[str, str]]],
        img: Optional[PasteResult]=None
    ) -> Iterator[str]:
        """
        Make a single API query to the LLM.
        Supports both Gemini & OpenAI models.
        """

        def convert_img_to_bytes(img: PasteResult, format: str = 'png') -> bytes:
            """Converts a Pillow Image object into bytes."""
            # 1. Save the Pillow image to an in-memory byte buffer
            buffer = BytesIO()
            img.save(buffer, format=format)
            return buffer.getvalue()

        def convert_bytes_to_base64(img_bytes: bytes) -> str:
            """Convert bytes to base64 string."""
            return base64.b64encode(img_bytes).decode('utf-8')

        if img.image_data is not None:
            byte_image = convert_img_to_bytes(img.image_data)
            base_64_image = convert_bytes_to_base64(byte_image)

        if model in MODELS_OPENAI:
            messages = (
                ([{"role": "system", "content": system_prompt}])
                + [{"role": role, "content": msg} for role, msg in chat_history or []]
            )

            user_content = [{"type": "text", "text": user_message}]
            if img.image_data is not None:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base_64_image}"
                    }
                })

            messages.append({"role": "user", "content": user_content})
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

            history = [
                {
                    "role": ("model" if role == "assistant" else "user"),
                    "parts": [types.Part(text=msg)],
                }
                for role, msg in chat_history or []
            ]

            user_parts = [types.Part(text=user_message)]
            if img.image_data is not None:
                user_parts.append(
                    types.Part.from_bytes(
                        data=byte_image,
                        mime_type="image/png")
                    )

            contents = [*history, types.Content(role="user", parts=user_parts)]

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

        if model in MODELS_OLLAMA:

            messages = [{"role": "system", "content": system_prompt}]
            if chat_history != []:
                messages.append([{"role": role, "content": msg} for role, msg in chat_history])
            user_message_payload = {"role": "user", "content": user_message}

            if img.image_data is not None:
                user_message_payload["images"] = [base_64_image]

            messages.append(user_message_payload)

            stream = self.ollama_client.chat(
                model=model,
                messages=messages,
                stream=True,
            )
            response = ""
            for chunk in stream:
                content = chunk.get("message", {}).get("content") or chunk.get("response") or ""
                if content:
                    response += content
                    yield content

    def chat(self, model: str, user_message: str, img: Optional[PasteResult]=None) -> Iterator[str]:
        response = ""
        try:
            for chunk in self.api_query(
                                model=model,
                                user_message=user_message,
                                system_prompt=self.sys_prompt,
                                chat_history=self.messages,
                                img=img):

                    response += chunk
                    yield chunk
        except Exception as e: # noqa
            # yield exception to avoid breaking the stream - will be handled in streamlit_helper.py
            # This is a workaround due to limitations in streaming error handling.
            yield Exception("Errororororor!!11!")
        self.messages.append(("user", user_message))
        self.messages.append(("assistant", response))
