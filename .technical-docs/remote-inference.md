# Remote inference & deployment configuration

The backend is a thin client toward every heavy service: LLM inference, OCR,
and web search are all addressed through environment variables. This means the
webapp (or the desktop AppImage) can run on a laptop while all inference lives
on a private server, reached over SSH tunnels — no code changes per deployment.

## Environment variable surface

| Variable | Default | Purpose |
|---|---|---|
| `MINERU_SERVER_URL` | unset | External MinerU OCR server (`mineru.cli.fast_api`). When unset, the backend spawns one locally per parse from its own Python environment. **Required for OCR in the desktop app** — the frozen sidecar does not bundle the ML stack. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint for local-model inference. |
| `BRAVE_API_KEY` | unset | Secret used by backend web search. |
| `GIGACHAD_CORS_ORIGINS` | `*` | Comma-separated CORS allow-list for the FastAPI backend. The Tauri shell sets this to its own webview origins; server deployments can restrict it to their public origin. |
| `GIGACHAD_BASE_DIR` | `~/Nextcloud/linux` | Root for all runtime data (chat histories, uploads, prompts, MinerU output). `src/config.py` expands it into `REMOTE_ROOT`. The desktop app defaults it to the XDG app-data dir (the AppImage mount is read-only) but respects a value exported before launch, so pointing it at the webapp's root shares one data store. |

`GIGACHAD_BASE_DIR` is a development and desktop knob only. `deployment.md` forbids
setting it in any production artifact (environment file, service unit) because
development and production deliberately share the same default Documents tree, and the
worktree is meant to isolate code rather than application state. Override it locally when
you need a scratch data root, never in a deployed configuration.

## Hosting MinerU OCR on a remote server

On the server (from a checkout of this repo, or any env with `mineru[all]`):

```bash
cd src && uv run python -m mineru.cli.fast_api --host 127.0.0.1 --port 8003
```

On the client machine, tunnel and point the app at it:

```bash
ssh -N -L 8003:127.0.0.1:8003 your-server &
MINERU_SERVER_URL=http://127.0.0.1:8003 ./run.sh          # webapp
MINERU_SERVER_URL=http://127.0.0.1:8003 ./gigachad-bot_*.AppImage  # desktop
```

The client protocol is plain HTTP (upload PDF → poll task → download result
zip), so it behaves identically through a tunnel. The fast_api server has **no
authentication** — keep it bound to loopback on the server and reach it only
via SSH tunnel or VPN; do not expose it publicly.

Backend auto-selection (`hybrid-auto-engine` vs `pipeline`) checks for
`nvidia-smi` on the *client* machine; when OCR runs remotely on different
hardware, pin the backend explicitly in the UI/request if the default guess is
wrong for the server.

## Hosting Ollama remotely

```bash
ssh -N -L 11434:127.0.0.1:11434 your-server &
OLLAMA_BASE_URL=http://127.0.0.1:11434 ./run.sh
```

Web search uses Brave directly from the backend and requires no local search service.

## Web vs. desktop build behavior

One codebase, two targets, decided automatically:

- **Web** (`npm run build` / `./run.sh`): Next.js builds `output: "standalone"`
  — `npm run start` works, SSR/API routes remain possible, CORS defaults to
  `*`. Nothing desktop-related runs in the browser (the Tauri API is a
  dynamically imported chunk that is never fetched).
- **Desktop** (`./builds.sh`): Tauri sets `TAURI_ENV_*` during its
  `beforeBuildCommand`, which flips Next.js to `output: "export"` for embedding
  in the webview. The PyInstaller sidecar bundles the FastAPI backend with the
  MinerU *client* only — the ~12 GB torch/CUDA/vllm OCR server stack is
  excluded (it could not run from a frozen app anyway, since MinerU spawns its
  server via `sys.executable`). The Tauri shell narrows CORS to its own
  origins via `GIGACHAD_CORS_ORIGINS`.
