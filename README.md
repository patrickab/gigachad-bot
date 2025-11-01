# Study Assistant

A study assistant with flexible, customizable behavior.
Use your own chatbot on localhost via API-Calls.
Permanently adjust knowledge level to your background - no more answers, that are too complex/trivial
The sidebar allows you to select pre-defined system prompts and swap models on the fly, giving you fine-grained control over the assistant.

## Core Features
- Clean markdown/LaTeX formatting.
- Adjust the length, complexity of the answer.
- LLM responses with table of contents incl. links to response sections
- Support for Gemini API & OpenAI API (easily extendable to any API)
- Manage your own prompt library & flexibly adjust chatbot behavior as you talk.
- Store responses from the website directly in your [Obsidian](https://obsidian.com) vault

## Demo
The image below shows response behavior for **the same input** with different **model/system prompt**
<img width="2345" height="1439" alt="chatbot-demo" src="https://github.com/user-attachments/assets/77f87cc0-a14c-4b23-9410-0413bc3f5bd0" />

## Installation & Setup
- Clone the repo.
- Store your `OPENAI_API_KEY` or `GEMINI_API_KEY` as environment variable  (e.g. in `~/.bashrc`).
- Create a virtual environment using `uv sync`.
- Activate your virtual environment using `source .venv/bin/activate`
- Use `./run.sh` to start the application

## Work in Progress
- Explore repositories visually and enrich your LLM-queries with context-aware code snippets by using Retrieval Augmented Generation (RAG).
- Keep your conversations tidy with a compact, expandable history
- Flexibly finetune LLM-behavior for brainstorming, generation of markdown learning material (markdown & Jupyter notebook) & explanations directly in your browser.

