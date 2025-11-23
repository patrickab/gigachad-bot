import base64
import csv
from io import BytesIO
import os
import shutil
import subprocess
from typing import Dict, Iterator, List, Optional, Tuple

from litellm import completion
from ollama import Client as OllamaClient
from openai import OpenAI as VLLMClient
from streamlit_paste_button import PasteResult

from src.config import MODELS_GEMINI, MODELS_OPENAI

EMPTY_PASTE_RESULT = PasteResult(image_data=None)


def _has_nvidia_gpu() -> bool:
    """
    Check for NVIDIA GPU availability using standard library only.
    Returns True if 'nvidia-smi' is found and executes successfully.
    """
    # 1. Check if the binary exists in PATH
    if not shutil.which("nvidia-smi"):
        return False
            
    # 2. Try executing it to ensure drivers are actually working
    try:
        # Run nvidia-smi to query GPU count. 
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"], 
            capture_output=True, 
            text=True, 
            timeout=3
        )

        if result.returncode == 0:
            count = int(result.stdout.strip())
            return count > 0

    except (subprocess.SubprocessError, ValueError):
        return False

    return False

API_CLIENT_INFO = {
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "gemini_api_key": os.getenv("GEMINI_API_KEY"),
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
    "is_ollama_installed": shutil.which("ollama") is not None,
    "is_gpu_available": False #_has_nvidia_gpu() # GPU behavior remains to be tested
}

class LLMClient:

    def __init__(self) -> None:

        self.messages: List[Tuple[str, str]] = [] # [role, message] - only store text for efficiency
        self.sys_prompt = ""

        if API_CLIENT_INFO["is_ollama_installed"]:
            self.ollama_client = OllamaClient(host="http://localhost:11434")
        if API_CLIENT_INFO["is_gpu_available"]:
            self.vllm_client = VLLMClient(base_url="http://localhost:8000")

    # -------------------------------- Core LLM Interaction -------------------------------- #

    def _process_image(self, img: PasteResult) -> Optional[str]:
        buffer = BytesIO()
        img.image_data.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_image}"

    def api_query(
        self, model: str,
        user_message: Optional[str] = None,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[Tuple[str, str]]]=None,
        img: Optional[PasteResult] = EMPTY_PASTE_RESULT,
        **kwargs: Dict[str, any]
    ) -> Iterator[str]:
        """
        Stateless API call using LiteLLM to unify the request format.
        Routes to the correct local/cloud settings based on __init__ detection.
        """

        # 1. Set defaults
        api_base = None
        custom_llm_provider = None

        # 2. Prepare Messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if chat_history:
            messages.extend(chat_history)
        if user_message:
            if img.image_data is None:
                # Text-only Format: Simple string
                content_payload = user_message
            else:
                base64_img = self._process_image(img)
                # Multimodal Format: List of dictionaries
                content_payload = [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": base64_img}}
                ]

            messages.append({"role": "user", "content": content_payload})

        # 3. Determine Provider
        is_cloud_model = (model in MODELS_OPENAI) or (model in MODELS_GEMINI)

        if not is_cloud_model:
            # Routing Logic: Prefer vLLM (GPU) > Ollama (CPU)
            if API_CLIENT_INFO["is_gpu_available"]:
                # Use vLLM settings
                api_base = str(self.vllm_client.base_url)
                custom_llm_provider = "openai" # vLLM mimics OpenAI
            
            else:
                # Use Ollama settings
                api_base = "http://localhost:11434"
                custom_llm_provider = "ollama"

                # LiteLLM requires 'ollama/' prefix
                if not model.startswith("ollama/"):
                    model = f"ollama/{model}"
        else:
            if model in MODELS_OPENAI:
                model = f"openai/{model}"
            elif model in MODELS_GEMINI:
                model = f"gemini/{model}"

        # 4. Execute request via LiteLLM
        try:
            assert messages[0]['role'] == 'system'
            response = completion(
                model=model,
                messages=messages,
                stream=True,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                **kwargs # Pass temperature, top_p, etc.
            )

            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except Exception as e:
            yield f"API Error: {e!s}"

    def chat(self, model: str,
        user_message: str,
        system_prompt: Optional[str] = "",
        img: Optional[PasteResult] = EMPTY_PASTE_RESULT,
        **kwargs: Dict[str, any]
    ) -> Iterator[str]:
        """Stateful Chat Wrapper"""
        response = ""
        stream = self.api_query(
            model=model,
            user_message=user_message,
            system_prompt=system_prompt,
            chat_history=self.messages,
            img=img,
            **kwargs)

        for chunk in stream:
            response += chunk
            yield chunk

        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": response})

    # -------------------------------- Streamlit State Management -------------------------------- #

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
        self.messages = []
