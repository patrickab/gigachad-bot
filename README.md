# A chatinterface for personal use.

## Core Features
- Customizable library of system prompts for recurring tasks.
- Unified access to 2100+ models from 100+ providers using [LiteLLM](https://models.litellm.ai/) 
- CPU/GPU local inference using [vLLM](https://docs.vllm.ai/en/latest/) and [Ollama](https://ollama.com/).
- Automatic VRAM ressource & localhost server management.
- Optical Character Recognition (OCR) for Screenshot-to-LaTeX extraction - including web-based editor.

## Further Features
- Token-efficient image compression for fast local inference at scale.
- Store responses directly into [Obsidian](https://obsidian.md/) vault (with automatically generated YAML-headers).
- Chat History Management & Storage on local disk.

## Installation & Setup
- Clone the repo.
- Store your `OPENAI_API_KEY` or `GEMINI_API_KEY` as environment variable  (e.g. in `~/.bashrc`).
- Alternatively you can get a free API key from Ollama.
- Create a virtual environment using `uv sync`.
- Activate your virtual environment using `source .venv/bin/activate`
- Use `./run.sh` to start the application
