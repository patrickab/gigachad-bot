# GigaChat Bot

A minimalist, local-first LLM web interface. Designed for privacy, focus and custom workflows.

<img width="500" height="500" alt="Gigachad Interface" src="https://github.com/user-attachments/assets/34727936-ddbd-44d7-b191-d57084b30984" />

---

## Features

- **Private cloud integration**: Histories and artifacts can be stored on private infrastructure. 
- **Cloud & Local inference**: Support for 2100+ models from 100+ providers via [LiteLLM](https://github.com/BerriAI/litellm).
- **Adaptive model selector**: collapses to a circular icon button when the full pill would overlap the chat column. Hover or keyboard focus smoothly reveals the model name, respecting reduced-motion preferences.
- **Persistent memory**: Extract facts and user preferences from conversations via toolcalling.
- **PDF parsing**: local VLM text extraction via [MinerU](https://github.com/opendatalab/mineru).
- **Modes**:
  - **Deep Research**: multi-step reports via [GPT-Researcher](https://github.com/assafelovic/gpt-researcher).
  - **Web Search**: Brave LLM Context search with visible, citation-mapped source labels.
- **Project workspaces**: one stable sidebar groups sessions, canvases, architecture, and files under each project, alongside isolated memory and a kanban board.
  - Select a project to restore its last-opened workspace in the current session. Use its chevron to expand or collapse navigation without switching views.
  - Open items to change the workspace and contextual tools. Category expansion never changes the current view.
  - The sidebar uses one text-only recursive tree with consistent indentation. Hover or keyboard-focus a row to reveal creation and management actions. Empty categories add no placeholder rows.
  - A borderless **project / category / file** switcher is centered in the existing toolbar row. It fades out at rest, reappears on hover or keyboard focus, and stays visible while its menu is open. Click the arrows on the left to advance through the tab list with wraparound, or toggle directly when two tabs are open. The arrows are disabled with one tab. Click or tap the breadcrumb to open the centered tab list, including with the sidebar collapsed. The breadcrumb no longer collapses on overlap, and neighboring controls never shift. Tabs stay naturally sorted by project, category, and filename, with **general** last. The dropdown fits long labels within the workspace and offers search for long tab lists. Unsaved canvases display as `new.canvas`.
  - Switching tabs crossfades workspace content over 160ms with a subtle upward settle and breadcrumb fade. The sidebar and toolbar stay anchored, rapid switches interrupt immediately, and reduced-motion preferences disable the effect.
  - Background chats retain their last visible column width and composer spacing, preventing layout jumps and unnecessary plot resizing when returning to a tab.
  - Loose chats, canvases, and external file roots remain accessible under **general**, always listed after projects. The global memory button stays above the project list, without a mode heading.
  - Open workspace commands with `/` or `Ctrl/Cmd+K`. The focused command popup works with the sidebar collapsed, supports filtering and keyboard selection, and closes with Escape.
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
- In the deploying user's shell, set:

  ```bash
  REPO="$HOME/git/gigachad-bot"
  PROD="$HOME/git/gigachad-bot-prod"
  ENV="${XDG_CONFIG_HOME:-"$HOME/.config"}/gigachad-bot/env"
  ```

- Keep backend credentials in `~/.secrets` as `export NAME=value` lines.
  `run-backend.sh` sources it on every launch, including the production systemd
  service. Create `$ENV` for shared nonsecret configuration; see the
  [deployment runbook](.technical-docs/deployment.md#1-prepare-the-private-host)
  for its strict `KEY=value` format.
### Developer launchers

The launchers install the dependencies they own and run in development mode:

| Command | Use |
| --- | --- |
| `./run-frontend.sh` | Run only the local Next.js UI on `:2999`. It installs frontend dependencies first. |
| `./run-backend.sh` | Run only the local FastAPI backend on `127.0.0.1:8001` with reload enabled. It syncs Python dependencies first. |
| `./run.sh` | Run both local services; this is the usual full-app developer command. |

Both `run-backend.sh` and `run-frontend.sh` load the shared configuration
file. The backend runner first sources `~/.secrets`; the strict config loader
never evaluates its file and leaves already-exported environment variables
unchanged. This lets development and production share configuration and the
default Documents store while production uses separate code.

The development backend accepts requests that carry no Tailscale login as the
local OS user, so `./run.sh` needs no proxy. It binds `127.0.0.1` only, and the
production launcher always sets `GIGACHAD_ENV=production`, so an inherited
development setting can never relax the deployed backend.

For a private Tailnet development endpoint, opt in with
`./run-backend.sh --tailscale` or `./run.sh --tailscale`. This assumes Tailscale
is already installed and the machine is already joined to the intended Tailnet.
The option configures private Tailscale Serve for the loopback backend only; it
does not install Tailscale, log in or join a Tailnet, use `sudo`, configure
systemd, enable Funnel, or provide public access.

These are developer launchers, not a production deployment method. The deployment
runbook currently describes a Vercel/Tailscale setup. Update it to serve the UI
on the private origin before using it for a browser-portable production deployment.

### Private deployment browser compatibility

Serve the production UI and `/api` from the same private Tailscale HTTPS origin.
Use a relative production API base (`/api`), not a public Vercel UI that calls a
Tailscale/private API directly. Firefox-family browsers classify that as
local-network access, so CORS and per-browser permission grants are not a
durable fix.

OMP's gateway token is read once at backend startup and cached for the
process lifetime: restart `gigachad-bot.service` after installing or
restarting `gigachad-bot-omp.service`, or OMP model calls will 401.

The development backend runs on `127.0.0.1:8001`; the systemd backend runs on
`127.0.0.1:8002`. They may run concurrently, each using its own isolated
PostgreSQL container and data volume (`gigachad-dev` vs `gigachad-prod`). The
production user unit uses systemd's `%h` specifier for the deploying user's
home directory and assumes `$PROD`; its full setup and update steps are in
the runbook.

The unit carries only `GIGACHAD_ENV_FILE` with the nonsecret `$ENV` path. Its
production runner sources `~/.secrets`, then invokes the strict
`deploy/load-env.sh` loader before it execs loopback Uvicorn.
