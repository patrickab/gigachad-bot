# Giagachad Bot

A knowledge-management assistant with flexible, customizable behavior.
Use your own chatbot on localhost via API-Calls to your preferred provider (current support: OpenAI, Gemini, Ollama - Claude support coming soon)
Permanently adjust knowledge level to your background - no more answers, that are too trivial or too complex.
The sidebar allows you to select pre-defined system prompts and swap models on the fly, giving you fine-grained control over your assistants behavior.

## Core Features
- Retrieval Augmented Generation (RAG).
- Optical Character Recognition (OCR) and img-to-LaTeX including webbased editor.
- Clean markdown/LaTeX formatting.
- Automatic adjustment of the answer length to the complexity of your query.
- Fine-grained control over answer length via flexible system prompts.
- Chat History Management & Storage on your local disk.
- Support for all [Ollama](https://ollama.com/search) open-source models.
- Support for Gemini API & OpenAI API (easily extendable to any API).
- Manage your own prompt library & flexibly adjust chatbot behavior as you talk.
- Store responses from the website directly in your [Obsidian](https://obsidian.com) vault.
- Automatically generated YAML-headers for your Obsidian notes.

## Demo

https://github.com/user-attachments/assets/c21ec4b0-2c92-419c-a46e-36453efce309

## Installation & Setup
- Clone the repo.
- Store your `OPENAI_API_KEY` or `GEMINI_API_KEY` as environment variable  (e.g. in `~/.bashrc`).
- Create a virtual environment using `uv sync`.
- Activate your virtual environment using `source .venv/bin/activate`
- Use `./run.sh` to start the application

## Work in Progress
- OCR-based PDF miner for conversion of entire books into embeddings.

