# GigaChat Bot

A minimalist, local-first LLM web interface. Designed for focus and custom workflows.

<img width="500" height="500" alt="Gigachad Interface" src="https://github.com/user-attachments/assets/34727936-ddbd-44d7-b191-d57084b30984" />

---

## Features

- **Private cloud integration**: Histories and artifacts can be stored on private infrastructure. 
- **Cloud & Local inference**: Support for 2100+ models from 100+ providers via [LiteLLM](https://github.com/BerriAI/litellm).
- **Persistent memory**: Extract facts and user preferences from conversations via toolcalling.
- **PDF parsing**: local VLM text extraction via [MinerU](https://github.com/opendatalab/mineru).
- **Modes**:
  - **Deep Research**: multi-step reports via [GPT-Researcher](https://github.com/assafelovic/gpt-researcher).
  - **Web Search**: Supports domain filters. Citation-mapped search via [Vane](https://github.com/ItzCrazyKns/Vane)/[SearXNG](https://github.com/searxng/searxng).
- **Project Mode**: isolated memory, kanban board, and a per-project document library.
- **Chat branching**: fork a conversation from any message, merge branches back.
- **Obsidian vaults**: integrate Obsidian vaults into project documents. Files can be attached as live references. Edits write back to the source.

---

## Tech Stack

- **Frontend**: Next.js 15, React 19, Tailwind CSS v4
- **Backend**: FastAPI (Python 3.12), LiteLLM, MinerU, GPT-Researcher

---

## Setup

- Clone the repository.
- Store your `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, or `OPENROUTER_API_KEY` in environment variables.
- Run `uv sync` to install Python dependencies.
- Run `uv run gigachad-install` to install Node.js into the venv and frontend dependencies.
- Execute `./run.sh` to start the app (backend on `:8001`, frontend on `:2999`).
