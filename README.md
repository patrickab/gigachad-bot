# GenAI Toolbox
A collection of utilities for personal use. Created for experimenting & making my workflows more efficient.

## Core Features
- Customizable library of system prompts for recurring tasks.
- Unified access to 2100+ models from 100+ providers using [LiteLLM](https://models.litellm.ai/) 
- CPU/GPU local inference using [vLLM](https://docs.vllm.ai/en/latest/) and [Ollama](https://ollama.com/).
- Automatic VRAM ressource & localhost server management.
- Retrieval Augmented Generation (RAG).
- RAG data pipeline (preparation/extraction/ingestion).
  - Visual Language Model (VLM) PDF-2-Markdown conversion using [MinerU](https://github.com/opendatalab/MinerU)
  - Markdown Post-Processing using RegEx & LLMs to guarantee RAG-compatible format.
  - Ingestion of RAG payloads into a lightweight RAG Database [see here](https://github.com/patrickab/rag-database/tree/main).
- Optical Character Recognition (OCR) for Screenshot-to-LaTeX extraction - including web-based editor.

## Demo
https://github.com/user-attachments/assets/4a977472-dfbb-4166-a53e-c3c35025a939

## Further Features
- Token-efficient image compression for fast local inference at scale.
- Store responses directly into [Obsidian](https://obsidian.md/) vault (with automatically generated YAML-headers).
- Chat History Management & Storage on local disk.
- ...and much more to explore :)

## Installation & Setup
- Clone the repo.
- Store your `OPENAI_API_KEY` or `GEMINI_API_KEY` as environment variable  (e.g. in `~/.bashrc`).
- Create a virtual environment using `uv sync`.
- Activate your virtual environment using `source .venv/bin/activate`
- Use `./run.sh` to start the application






