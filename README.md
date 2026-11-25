# GigaChad Bot

A self-hosted LLM workspace for chat, research, and working with your own documents. Use cloud or local models and keep histories and artifacts on infrastructure you control.

<img width="500" height="500" alt="GigaChad Bot interface" src="https://github.com/user-attachments/assets/34727936-ddbd-44d7-b191-d57084b30984" />

## What you can do

- **Choose your models**: use the cloud providers your [Pi](https://github.com/earendil-works/pi) login and API keys give you, or local [Ollama](https://ollama.com) models.
- **Watch subscription limits**: the sidebar shows how much of your [Pi](https://github.com/earendil-works/pi) login's 5h, 7d and monthly limits you have used.
- **Review changes**: open the git diff of any mounted file directory from the sidebar.
- **Work in worktrees**: a mounted git clone lists its worktrees, switches the mount to one, creates new ones from a pulled base branch and removes them.
- **Code with Pi agents**: open Pi sessions in tabs. The **Code** section holds one row per repository or file mount, with its **Worktrees**, **Sessions** and files. A new session starts in that directory.
- **Organize projects**: keep conversations, files, canvases, architecture diagrams, and project memory together. With the Memory tool on, each chat uses its containing project's memory, independently of the selected tab.
- **Rename projects**: use the project's Rename action. The slug follows the name, and stored chats, documents, the project's memory, architecture, attachments and the links between them move with it. A name whose slug another project already uses is rejected. Open tabs follow the project and reload its content. Linked disk folders keep their original names. Deep links and URLs copied outside the app are not redirected.
- **Archive projects**: hover a project row to archive it without deleting its content. Hover **Projects** and open the archive icon for an anchored popover of all projects, then select **Restore** to return an archived project to the active sidebar.
- **Archive directories**: the archive icon and **Restore** popover of **Code** work like the project archive. The archive is kept per device.
- **Explore conversations**: branch from any message and merge branches back.
- **Work with documents**: extract PDF content with [MinerU](https://github.com/opendatalab/mineru), attach live file references, and edit Obsidian vault files in place.
- **Think visually**: draw on an infinite canvas, attach PDFs, and share canvas snapshots with a model.
- **Annotate PDFs**: write and highlight on PDFs with pen or mouse. Every copy of a PDF shares its annotations, on a canvas and in its own tab.
- **Research and remember**: search the web, generate multi-step research reports, and keep facts and preferences in persistent memory that the assistant reads and updates itself.

## Demo

https://github.com/user-attachments/assets/57864372-dac2-49f3-97f7-5bceecf53c49

## Run locally

Requires Linux, Bash, Python 3.12 or 3.13, [uv](https://docs.astral.sh/uv/), and Docker with Compose. Docker must be running and accessible to your user.

1. Clone the repository and enter its directory.
2. Start the app:

   ```bash
   ./run.sh
   ```

   Startup checks `POSTGRES_PASSWORD_DEV` in `~/.secrets` or your environment. If missing, it runs `scripts/setup-postgres.sh`, which asks before generating a password and creating the database. Declining stops startup without changes. Credentials are saved privately in `~/.secrets`, and PostgreSQL starts through Docker Compose on `127.0.0.1:5432`. Existing credentials and database volumes are reused. If an existing database has no password configured, restore its original password rather than generating a replacement.

   Add credentials for your chosen model providers to `~/.secrets` as `export NAME=value` lines. Keep this file private and outside the repository. You can still run `./scripts/setup-postgres.sh` separately.

3. Open [localhost:2999](http://localhost:2999). The development API runs on `127.0.0.1:8001`.

The launcher installs dependencies, including Node.js, and starts the development services and PostgreSQL. The default dependency install includes MinerU's large local OCR stack, so allow extra time and disk space on the first run.

To run services separately, use `./run-frontend.sh` or `./run-backend.sh`. Optional nonsecret configuration goes in `${XDG_CONFIG_HOME:-$HOME/.config}/gigachad-bot/env` as plain `KEY=value` lines.

`./run.sh` also starts pi-host on `127.0.0.1:8300`, which runs the Pi sessions of the code tabs and workbench chats. It needs `pi` on `PATH` at the version `pi-host/package.json` pins. Set `GIGACHAD_PI_HOST=0` to skip it. See the [repository documentation index](AGENTS.md#documentation) for operational guidance.

`./run.sh` also starts the factory preview gate on `127.0.0.1:30185`, the one address a browser reaches factory previews through. Factory tasks run as pi-host sessions. Set `GIGACHAD_FACTORY=0` to skip the gate.

## Further reading

- [System overview](.docs/system-overview.md): purpose, concepts, and main workflows.
- [Architecture](.docs/architecture.md): components, data flow, persistence, and integration contracts.
- [Module map](.docs/module-map.md): ASCII source tree, ownership, and implementation entry points.
- [Repository documentation index](AGENTS.md#documentation): task-specific operational runbooks.
- Prefer one private HTTPS origin for the UI and `/api` in production. The existing split Vercel/Tailscale setup remains documented for current deployments.

The local launchers are for development, not production. Keep the backend private rather than exposing it directly to the internet.

## License

GigaChad Bot is licensed under the [GNU AGPL-3.0-only](LICENSE), with an additional permission for MinerU in [LICENSE-EXCEPTION](LICENSE-EXCEPTION).
