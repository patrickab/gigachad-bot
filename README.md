# Gigachad

A minimalist, local-first LLM web interface. Designed for focus and custom workflows.

<img width="500" height="500" alt="Gigachad Interface" src="https://github.com/user-attachments/assets/34727936-ddbd-44d7-b191-d57084b30984" />

---

## Features

- **Cloud & Local**: 2100+ models from 100+ providers via LiteLLM alongside local Ollama and vLLM.
- **Project Mode**: Tab isolation, scoped file assets, and Kanban boards.
- **Flat File Storage**: Chat histories and uploads organized in flat, human-readable JSON and Markdown files. Fits perfectly with Nextcloud or Obsidian.
- **Specialized Workflows**:
  - **Deep Research**: Detailed reports via GPT-Researcher with live step tracing.
  - **Morphic Search**: Real-time citation-mapped web search.
  - **PDF Study**: Document parsing to learning goals, summaries, and notebook articles.
  - **OCR**: Screenshot-to-LaTeX extraction with side-by-side editing.

---

## Tech Stack

- **Frontend**: Next.js 15, React 19, Tailwind CSS v4, Framer Motion
- **Backend**: FastAPI (Python 3.12), LiteLLM, MinerU

---

## Setup

- Clone the repository.
- Store your `OPENAI_API_KEY` or `GEMINI_API_KEY` in environment variables.
- Run `uv sync` to install Python dependencies.
- Run `uv run gigachad-install` to install Node.js into the venv and frontend dependencies.
- Execute `./run.sh` to start the app (backend on `:8001`, frontend on `:2999`).


