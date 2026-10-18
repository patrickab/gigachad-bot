# GigaChat Bot

A self-hosted LLM workspace for chat, research, and working with your own documents. Use cloud or local models and keep histories and artifacts on infrastructure you control.

<img width="500" height="500" alt="GigaChat Bot interface" src="https://github.com/user-attachments/assets/34727936-ddbd-44d7-b191-d57084b30984" />

## What you can do

- **Choose your models**: connect cloud and local providers through [LiteLLM](https://github.com/BerriAI/litellm).
- **Watch subscription limits**: the sidebar shows how much of each [OMP](https://github.com/can1357/oh-my-pi) login's 5h, 7d and monthly limits you have used.
- **Review changes**: open the git diff of any mounted file directory from the sidebar.
- **Work in worktrees**: a mounted git clone lists its worktrees, switches the mount to one, creates new ones from a pulled base branch and removes them.
- **Code with omp agents**: open omp-web sessions in tabs. A project linked to a file mount lists the sessions of that directory under "code", the "OMP" section lists every other omp workspace.
- **Organize projects**: keep conversations, files, canvases, architecture diagrams, and project memory together. Each chat uses its containing project's memory, independently of the selected tab.
- **Explore conversations**: branch from any message and merge branches back.
- **Work with documents**: extract PDF content with [MinerU](https://github.com/opendatalab/mineru), attach live file references, and edit Obsidian vault files in place.
- **Think visually**: draw on an infinite canvas, attach PDFs, and share canvas snapshots with a model.
- **Research and remember**: search the web, generate multi-step research reports, and extract facts and preferences into persistent memory.

## Demo

https://github.com/user-attachments/assets/57864372-dac2-49f3-97f7-5bceecf53c49

## Run locally

Requires Linux, Bash, Python 3.12 or 3.13, [uv](https://docs.astral.sh/uv/), and Docker with Compose. Docker must be running and accessible to your user.

1. Clone the repository and enter its directory.
2. Add a database password to `~/.secrets`:

   ```bash
   export POSTGRES_PASSWORD_DEV='replace-with-a-strong-password'
   ```

   Add credentials for your chosen model providers to the same file as `export NAME=value` lines. Keep this file private and outside the repository.

3. Start the app:

   ```bash
   ./run.sh
   ```

4. Open [localhost:2999](http://localhost:2999). The development API runs on `127.0.0.1:8001`.

The launcher installs dependencies, including Node.js, and starts the development services and PostgreSQL. The default dependency install includes MinerU's large local OCR stack, so allow extra time and disk space on the first run.

To run services separately, use `./run-frontend.sh` or `./run-backend.sh`. Optional nonsecret configuration goes in `${XDG_CONFIG_HOME:-$HOME/.config}/gigachad-bot/env` as plain `KEY=value` lines.

`./run.sh` also serves omp-web on `127.0.0.1:30177` for the omp tabs. It needs `omp` on `PATH` and the ompweb fork checked out at `~/git/ompweb` (branch `custom`). The first start builds it in a separate worktree, which takes a few minutes. Set `GIGACHAD_OMP_WEB=0` to skip it.

## Further reading

- [System overview](.technical-docs/system-overview.md): architecture, storage, and backend services.
- [Remote inference](.technical-docs/remote-inference.md): connect inference services on another machine.
- [Deployment runbook](.technical-docs/deployment.md): private-host configuration and service management. The runbook currently describes a split Vercel/Tailscale setup. For browser-compatible production access, serve the UI and `/api` from the same private HTTPS origin instead.

The local launchers are for development, not production. Keep the backend private rather than exposing it directly to the internet.
