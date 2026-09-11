# GigaChat Bot

A minimalist, local-first LLM web interface. Designed for privacy, focus and custom workflows.

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
- **Infinite Canvas**: whiteboard for drawing, pdfs attach to canvas, canvas can be rendered as jpg and attached to chat
- **Chat branching**: fork a conversation from any message, merge branches back.
- **Obsidian vaults**: integrate Obsidian vaults into project documents. Files can be attached as live references. Edits write back to the source.

---

## Demo

https://github.com/user-attachments/assets/57864372-dac2-49f3-97f7-5bceecf53c49

---

## Tech Stack

- **Frontend**: Next.js 15, React 19, Tailwind CSS v4
- **Backend**: FastAPI (Python 3.12), LiteLLM, MinerU, GPT-Researcher

---

## Setup

- Clone the repository.
- Store your `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, or `OPENROUTER_API_KEY` in environment variables.

### Developer launchers

The launchers install the dependencies they own and run in development mode:

| Command | Use |
| --- | --- |
| `./run-frontend.sh` | Run only the local Next.js UI on `:2999`. It installs frontend dependencies first. |
| `./run-backend.sh` | Run only the local FastAPI backend on `127.0.0.1:8001` with reload enabled. It syncs Python dependencies first. |
| `./run.sh` | Run both local services; this is the usual full-app developer command. |

For a private Tailnet development endpoint, opt in with
`./run-backend.sh --tailscale` or `./run.sh --tailscale`. This assumes Tailscale
is already installed and the machine is already joined to the intended Tailnet.
The option configures private Tailscale Serve for the loopback backend only; it
does not install Tailscale, log in or join a Tailnet, use `sudo`, configure
systemd, enable Funnel, or provide public access.

These are developer launchers, not a production deployment method. Production
uses the systemd backend service and Vercel/Tailscale Serve deployment described
in [the deployment runbook](.technical-docs/deployment.md).

### Nextcloud WebDAV storage

By default `GIGACHAD_STORAGE=local` retains the historical
`~/Nextcloud/linux/Documents` layout. To make the app-owned stores use
Nextcloud directly (without the desktop client), configure its WebDAV
**Documents** collection and a Nextcloud app password:

```bash
export GIGACHAD_STORAGE=webdav
export GIGACHAD_WEBDAV_URL="https://cloud.example.com/remote.php/dav/files/USER/linux/Documents"
export GIGACHAD_WEBDAV_USER="USER"
export GIGACHAD_WEBDAV_PASSWORD="APP_PASSWORD"
```

Create the app password in Nextcloud's Security settings. Chats, projects,
memory, prompts, and architecture graphs use one `DataStore` interface and
logical keys relative to `Documents`; the WebDAV adapter sends ETags on
revision-aware writes and returns a conflict rather than silently overwriting a
newer file. External FileVault/Obsidian roots remain external local
integrations.
