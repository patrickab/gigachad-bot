# A chatinterface for personal use.

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/34727936-ddbd-44d7-b191-d57084b30984" />

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

## Tech Stack
- **Frontend**: Next.js 15 + React + Tailwind CSS v4 + Framer Motion
- **Backend**: FastAPI + SSE (Server-Sent Events) streaming

## Installation & Setup
- Clone the repo.
- Store your `OPENAI_API_KEY` or `GEMINI_API_KEY` as environment variable (e.g. in `~/.bashrc`).
- Alternatively you can get a free API key from Ollama.
- Run `uv sync` to install Python dependencies.
- Run `uv run gigachad-install` to install Node.js into the venv and frontend dependencies.
- Use `./run.sh` to start the application (backend on `:8001`, frontend on `:2999`).
