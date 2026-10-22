# Remote inference

- Point the backend at remote LLM/OCR services through environment variables.
- Use private SSH tunnels for remote inference without application changes.
- Keep Brave web-search calls server-side.
- Follow [deployment](deployment.md) for production identity and service setup.

## Configuration

| Variable | Default | Contract |
| --- | --- | --- |
| `MINERU_SERVER_URL` | unset | OCR endpoint. Unset starts local MinerU per parse. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama inference endpoint |
| `BRAVE_API_KEY` | unset | Backend-only web-search credential in `~/.secrets` |
| `GIGACHAD_CORS_ORIGINS` | `*` in development | Exact allowed browser origins. Production rejects `*` and empty lists. |
| `GIGACHAD_BASE_DIR` | `~/Nextcloud/linux` | PDF/MinerU mirror root, expanded to `REMOTE_ROOT` by `src/config.py`. PostgreSQL remains authoritative for histories, uploads, and prompts. |

- Override `GIGACHAD_BASE_DIR` only for development, never production artifacts.
- Retain the shared default Documents mirror in production.

## MinerU OCR

### Start the server

- Use this checkout or an environment with `mineru[all]`:

```bash
cd src && uv run python -m mineru.cli.fast_api --host 127.0.0.1 --port 8003
```

### Connect the client

```bash
ssh -N -L 8003:127.0.0.1:8003 your-server &
MINERU_SERVER_URL=http://127.0.0.1:8003 ./run.sh
```

| Concern | Contract |
| --- | --- |
| HTTP protocol | Upload PDF, poll task, download result ZIP |
| Access | MinerU has no authentication. Bind loopback, use SSH/private VPN, never public ingress. |
| Hardware selection | Automatic selection checks client `nvidia-smi`, not remote hardware. Explicitly select the request/UI backend when hardware differs. |

## Ollama

- Tunnel to the remote server's loopback listener, then launch the client:

```bash
ssh -N -L 11434:127.0.0.1:11434 your-server &
OLLAMA_BASE_URL=http://127.0.0.1:11434 ./run.sh
```

## Web build

- Build the Next.js frontend with `npm run build` (`output: "standalone"`).
- Serve it with `npm run start`.
