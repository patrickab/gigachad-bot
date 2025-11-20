import base64
import csv
from io import BytesIO
import os
from typing import Dict, Iterator, List, Optional, Tuple

from google.genai import Client as GeminiClient
from google.genai import types
from ollama import Client as OllamaClient
from openai import OpenAI as OpenAIClient
from streamlit_paste_button import PasteResult

from src.config import MODELS_GEMINI, MODELS_OLLAMA, MODELS_OPENAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMPTY_PASTE_RESULT = PasteResult(image_data=None)

class LLMClient:
    """
    LLM client for streaming chat completions.
    Serves as wrapper for Ollama, OpenAI & Gemini clients.

    Includes:
       - System prompt management
       - Multimodal support (text + images)
       - Chat history management (store/load/reset)
    """

    def __init__(self) -> None:
        self.messages: List[Tuple[str, str]] = []  # [(role, message)]
        self.sys_prompt: str = ""

        if OPENAI_API_KEY is not None:
            self.openai_client = OpenAIClient(api_key=OPENAI_API_KEY)

        if GEMINI_API_KEY is not None:
            self.gemini_client = GeminiClient(api_key=GEMINI_API_KEY)

        if MODELS_OLLAMA != []:
            self.ollama_client = OllamaClient(host="http://localhost:11434")

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

    def _query_openai(self, model: str, messages: List[Dict]) -> Iterator[str]:
        stream = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def _query_gemini(self, model: str, system_prompt: str, contents: List) -> Iterator[str]:
        stream_resp = self.gemini_client.models.generate_content_stream(
            model=model,
            config=types.GenerateContentConfig(system_instruction=system_prompt, top_p=0.96, temperature=0.2),
            contents=contents,
        )
        for chunk in stream_resp:
            if chunk.parts:
                for part in chunk.parts:
                    if part.text:
                        yield part.text

    def _query_ollama(self, model: str, messages: List[Dict]) -> Iterator[str]:
        stream = self.ollama_client.chat(
            model=model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            content = chunk.get("message", {}).get("content") or chunk.get("response") or ""
            if content:
                yield content

    def api_query(
        self, model: str,
        user_message: str,
        system_prompt: str,
        chat_history: Optional[List[Tuple[str, str]]],
        img: Optional[PasteResult]=EMPTY_PASTE_RESULT
    ) -> Iterator[str]:
        """
        Make a single API query to the LLM.
        Supports Gemini, OpenAI, and Ollama models.
        """

        def _convert_img_to_bytes(img: PasteResult, format: str = 'png') -> bytes:
            """Converts a Pillow Image object into bytes."""
            buffer = BytesIO()
            img.save(buffer, format=format)
            return buffer.getvalue()

        def _convert_bytes_to_base64(img_bytes: bytes) -> str:
            """Convert bytes to base64 string."""
            return base64.b64encode(img_bytes).decode('utf-8')

        byte_image = None
        base_64_image = None
        if img and img.image_data:
            byte_image = _convert_img_to_bytes(img.image_data)
            base_64_image = _convert_bytes_to_base64(byte_image)

        if model in MODELS_OPENAI:

            # Load History
            messages = (
                ([{"role": "system", "content": system_prompt}])
                + [{"role": role, "content": msg} for role, msg in chat_history or []]
            )

            # Prepare user message
            user_content = [{"type": "text", "text": user_message}]
            if base_64_image:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base_64_image}"
                    }
                })

            # Assemble final message
            messages.append({"role": "user", "content": user_content})
            yield from self._query_openai(model, messages)

        if model in MODELS_GEMINI:

            # Load History
            history = [
                {
                    "role": ("model" if role == "assistant" else "user"),
                    "parts": [types.Part(text=msg)],
                }
                for role, msg in chat_history or []
            ]

            # Prepare user message
            user_parts = [types.Part(text=user_message)]
            if byte_image:
                user_parts.append(
                    types.Part.from_bytes(
                        data=byte_image,
                        mime_type="image/png")
                    )

            # Assemble final contents
            contents = [*history, types.Content(role="user", parts=user_parts)]
            yield from self._query_gemini(model, system_prompt, contents)

        if model in MODELS_OLLAMA:

            # Load History
            messages = [{"role": "system", "content": system_prompt}]
            if chat_history:
                messages.extend([{"role": role, "content": msg} for role, msg in chat_history])

            # Prepare user message
            user_message_payload = {"role": "user", "content": user_message}
            if base_64_image:
                user_message_payload["images"] = [base_64_image]

            # Assemble final message
            messages.append(user_message_payload)
            yield from self._query_ollama(model, messages)

    def chat(self, model: str, user_message: str, img: Optional[PasteResult]=EMPTY_PASTE_RESULT) -> Iterator[str]:
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
            # Error handling only works if assistant response is empty.
            yield Exception("Errororororor!!11!")
        self.messages.append(("user", user_message))
        self.messages.append(("assistant", response))
