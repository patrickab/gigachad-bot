# Private deployment: Vercel + Tailscale Serve

This runbook deploys the Next.js frontend to Vercel while keeping FastAPI, its
runtime data, and heavy services on one private host. Browsers call FastAPI
directly through Tailscale Serve:

```text
Tailnet-enrolled browser ── HTTPS ──> <backend-host>.<tailnet>.ts.net
                                      Tailscale Serve ──> 127.0.0.1:8001 FastAPI
Vercel ── serves the Next.js bundle ──> browser
```

The Vercel project is public web hosting, but its frontend is not an API proxy.
**Vercel cannot proxy to a Tailnet-only origin:** its infrastructure is not a
member of this Tailnet, and this frontend's `NEXT_PUBLIC_API_BASE` is used by
the browser for direct requests. Every browser that uses the app must therefore
run on a device enrolled in the Tailnet and permitted by its ACLs to reach the
backend's Serve HTTPS endpoint.

Do not use Tailscale Funnel for this deployment. Public access is outside this
design; it requires a separately designed, authenticated gateway and a security
review. The backend must remain bound to loopback.

## Values to choose

Replace these placeholders once and use the resulting values consistently:

| Placeholder | Meaning | Example |
| --- | --- | --- |
| `<tailnet-hostname>` | Stable MagicDNS name assigned to the host | `gigachad-backend.example.ts.net` |
| `<vercel-production-origin>` | Exact production browser origin, with scheme and no path/trailing slash | `https://gigachad-bot.vercel.app` |
| `<vercel-preview-origin>` | Exact preview origin to authorize temporarily | `https://gigachad-bot-git-main-team.vercel.app` |
| `<release-repository-url>` | Authenticated Git URL for this repository | `ssh://git@example/gigachad-bot.git` |

The API base provided to Vercel is always:

```text
https://<tailnet-hostname>/api
```

## 1. Prepare the private host

### Install and join Tailscale on Arch / Omarchy

Omarchy is Arch-based. Do **not** use Tailscale's universal installer here: it
can fail to identify Omarchy. Install the Arch package and enable its daemon
instead:

```bash
sudo pacman -Syu tailscale
sudo systemctl enable --now tailscaled
sudo tailscale up --hostname=gigachad-backend
sudo tailscale status
```

`tailscale up` prints a URL when interactive Tailnet login is required; open it
and complete the login in the intended Tailnet. Confirm the host appears in the
Tailnet admin console and that the Tailnet ACL permits the users/devices that
will use this app to reach this host over HTTPS. Record the MagicDNS hostname
shown by `tailscale status` or the admin console as `<tailnet-hostname>`.

Tailscale Serve requires HTTPS certificates to be enabled for the Tailnet. Its
first configuration can prompt for the required Tailnet consent; complete that
prompt as the Tailnet administrator. Do not enable Funnel.

### Install the application as its service user

The commands below create a non-login service account, a durable data root, and
a checkout owned by that account. They assume `uv`, Git, and a supported Python
are already available on the host; obtain them through the host's normal
package-management policy.

```bash
sudo useradd --system --create-home --home-dir /opt/gigachad-bot \
  --shell /usr/bin/nologin gigachad
sudo install -d -o gigachad -g gigachad -m 0750 /var/lib/gigachad-bot
sudo -u gigachad git clone <release-repository-url> /opt/gigachad-bot
sudo -u gigachad sh -lc 'cd /opt/gigachad-bot && uv sync --no-dev --no-group ocr'
```

For an existing service account or checkout, omit the creation/clone command
that is already satisfied. Do not use `run.sh` in production: it installs
dependencies, enables reload mode, and starts a development frontend.

The standard backend release excludes the roughly 12 GB local `ocr` group;
configure `MINERU_SERVER_URL` when OCR is provided by a separately hosted
MinerU service.

All persistent application state belongs below `GIGACHAD_BASE_DIR`. With the
value below, the backend creates and uses
`/var/lib/gigachad-bot/Documents` for chat histories, uploads, prompts,
projects, document libraries, OCR outputs, and other mutable application data.
Keep this directory on persistent storage, owned by `gigachad`; do not place it
inside a release checkout.

### Create the root-only environment file

Create the configuration directory and edit the environment file without
putting its contents in shell history:

```bash
sudo install -d -m 0750 /etc/gigachad-bot
sudoedit /etc/gigachad-bot/gigachad-bot.env
sudo chown root:root /etc/gigachad-bot/gigachad-bot.env
sudo chmod 0600 /etc/gigachad-bot/gigachad-bot.env
```

Use this as the starting content. Substitute real values before starting the
service; values shown as placeholders are not credentials.

```dotenv
# Required deployment settings — neither value is a secret.
GIGACHAD_BASE_DIR=/var/lib/gigachad-bot
GIGACHAD_CORS_ORIGINS=<vercel-production-origin>

# Configure these when their corresponding host-local capabilities are used.
# Keep each target private/loopback or otherwise reachable only from this host.
OLLAMA_BASE_URL=http://127.0.0.1:11434
VANE_URL=http://127.0.0.1:3001
SEARX_URL=http://127.0.0.1:8888
# MINERU_SERVER_URL=http://127.0.0.1:8003
# EMBEDDING_MODEL=ollama/bge-m3:latest

# Optional provider credentials. Uncomment only for providers in use.
# GEMINI_API_KEY=<secret>
# OPENROUTER_API_KEY=<secret>
# OPENAI_API_KEY=<secret>
```

| Variable | Required | Secret? | Purpose |
| --- | --- | --- | --- |
| `GIGACHAD_BASE_DIR` | Yes | No | Absolute durable root for all mutable app state. |
| `GIGACHAD_CORS_ORIGINS` | Yes | No | Comma-separated, exact Vercel browser origins; do not use `*` in this deployment. |
| `OLLAMA_BASE_URL` | When using local models/embeddings | No | Host-local Ollama endpoint. |
| `VANE_URL` | When using web search | No | Host-local Vane endpoint. |
| `SEARX_URL` | When using deep research | No | Host-local SearXNG endpoint. |
| `MINERU_SERVER_URL` | When OCR runs as a separate persistent MinerU service | No | Host-local MinerU API endpoint. If absent, the backend may launch MinerU work itself. |
| `EMBEDDING_MODEL` | Optional | No | Vane embedding-model identifier. |
| `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY` | Only for their providers | **Yes** | Provider credentials; keep only in this root-only host file, never in Vercel. |

The URL and model settings are configuration, not secrets. Provider API keys are
secrets. Do not commit this environment file, paste it into tickets, or set any
provider key as a Vercel environment variable. Model-provider choices and other
application state are persisted under `GIGACHAD_BASE_DIR`, so preserve that
root across releases.

Ensure any host-managed Ollama, Vane, SearXNG, and MinerU services are started
before their dependent features are used and listen only on loopback or another
host-private interface. They are backend dependencies, not separately exposed
Tailnet services.

### Provision the FastAPI systemd service

Create `/etc/systemd/system/gigachad-bot.service` with the following content:

```ini
[Unit]
Description=GigaChad Bot FastAPI backend
Wants=network-online.target
After=network-online.target tailscaled.service

[Service]
Type=simple
User=gigachad
Group=gigachad
WorkingDirectory=/opt/gigachad-bot
EnvironmentFile=/etc/gigachad-bot/gigachad-bot.env
ExecStart=/opt/gigachad-bot/.venv/bin/uvicorn src.backend.server:app --host 127.0.0.1 --port 8001
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then load and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gigachad-bot.service
sudo systemctl status gigachad-bot.service
curl --fail --show-error http://127.0.0.1:8001/healthz
```

The last command must return `{"status":"ok"}`. If it does not, inspect the
service rather than exposing another listener:

```bash
sudo journalctl -u gigachad-bot.service -n 100 --no-pager
```

The `ExecStart` deliberately has neither `--reload` nor a non-loopback host.
Do not bind Uvicorn to `0.0.0.0`, a LAN address, or the Tailscale address.

## 2. Publish the private HTTPS ingress

On the backend host, inspect any existing Serve configuration before changing
it. This runbook assumes the host is dedicated to this app at its Serve root.

```bash
tailscale serve status
```

Configure a persistent private HTTPS reverse proxy from the Serve hostname to
FastAPI. The root proxy preserves `/api` and `/healthz`, so no path rewrite is
needed:

```bash
tailscale serve --bg http://127.0.0.1:8001
tailscale serve status
```

The status output gives the actual URL. It should show a proxy to
`http://127.0.0.1:8001` and report that it is available within the Tailnet.
`--bg` makes this Serve configuration survive the command session and resume
across Tailscale restarts/reboots. Use `tailscale serve`, never `tailscale
funnel`.

From a different device that is enrolled in the Tailnet and allowed by the ACL,
verify the external leg:

```bash
curl --fail --show-error https://<tailnet-hostname>/healthz
curl --fail --show-error https://<tailnet-hostname>/api/config
```

A device outside the Tailnet must not be used for this check and must not be
given a fallback public URL.

## 3. Configure the Vercel frontend

In the Vercel project settings:

| Setting | Value |
| --- | --- |
| Framework preset | Next.js (auto-detected is fine) |
| Root Directory | `src/frontend` |
| Install Command | `npm ci` |
| Build Command | `npm run build` |
| Output Directory | Leave at Vercel's Next.js default; do not configure a static export directory. |

Add this environment variable for **Production** and **Preview** before each
corresponding deployment:

```text
NEXT_PUBLIC_API_BASE=https://<tailnet-hostname>/api
```

`NEXT_PUBLIC_API_BASE` is intentionally public build-time configuration, not a
secret. A change to it requires a new Vercel deployment because it is bundled
into browser code. Vercel does not need Tailnet access to build this frontend;
the browser, not Vercel, makes API calls.

For production, set `GIGACHAD_CORS_ORIGINS` to the exact production Vercel
origin (and any intentional production custom-domain origin), for example:

```dotenv
GIGACHAD_CORS_ORIGINS=https://gigachad-bot.vercel.app,https://app.example.com
```

Origins include scheme and host only: no `/api`, path, or trailing slash. Apply
a changed backend environment file with:

```bash
sudo systemctl restart gigachad-bot.service
```

### Preview deployments and CORS

A preview bundle still calls the same private API base. Its browser origin must
also be listed in `GIGACHAD_CORS_ORIGINS` before testing it. The backend's CORS
allow-list is exact; comma-separated values are supported, but Vercel preview
wildcards are not. For each temporary preview:

1. Obtain its complete `https://…vercel.app` origin from Vercel.
2. Append that exact origin to `GIGACHAD_CORS_ORIGINS` in the host file.
3. Restart `gigachad-bot.service`.
4. Deploy/test the preview from a Tailnet-enrolled browser.
5. Remove the preview origin and restart the service when the preview is no
   longer needed.

Do not replace the allow-list with `*` to make previews convenient. CORS is not
a substitute for Tailnet ACLs, but the allow-list prevents arbitrary web origins
from reading API responses in an enrolled browser.

## 4. Deployment sequence

Use this order for a first deployment or an update:

1. On the host, make the intended source revision available at
   `/opt/gigachad-bot` as `gigachad`, then run the appropriate dependency sync
   for that reviewed revision. Do not run `run.sh`.
2. Confirm `/etc/gigachad-bot/gigachad-bot.env` has the durable base path,
   exact production CORS origin(s), host-local dependency URLs, and any needed
   provider secrets.
3. Start/restart `gigachad-bot.service`; confirm the loopback health check.
4. Join Tailscale, configure `tailscale serve --bg http://127.0.0.1:8001`, and
   confirm Tailnet HTTPS health from another enrolled device.
5. Configure Vercel's root directory, build settings, and
   `NEXT_PUBLIC_API_BASE`; deploy production.
6. From an enrolled browser, open the Vercel production URL and use the app.
   A browser that is not in the Tailnet will be unable to reach its API by
   design.

Before an upgrade that could affect stored state, stop the service and take a
non-destructive backup of the durable root, then start it again:

```bash
sudo systemctl stop gigachad-bot.service
sudo install -d -o gigachad -g gigachad -m 0750 /var/backups/gigachad-bot
sudo rsync -aHAX /var/lib/gigachad-bot/ /var/backups/gigachad-bot/pre-upgrade-$(date +%F-%H%M%S)/
sudo systemctl start gigachad-bot.service
```

Keep the backup outside the release checkout and do not use `--delete` on the
backup command.

## 5. Verify the deployed path

Perform all browser-facing checks from a device that is both Tailnet-enrolled
and allowed by the Tailnet ACL.

1. **Health and path routing.** Confirm both `https://<tailnet-hostname>/healthz`
   and `https://<tailnet-hostname>/api/config` return successfully.
2. **CORS.** Check the production browser origin is accepted:

   ```bash
   curl --silent --show-error --dump-header - --output /dev/null \
     --request OPTIONS https://<tailnet-hostname>/api/config \
     --header 'Origin: <vercel-production-origin>' \
     --header 'Access-Control-Request-Method: GET' \
     --header 'Access-Control-Request-Headers: content-type'
   ```

   The response must include
   `access-control-allow-origin: <vercel-production-origin>`. Repeat
   with a preview origin only while that preview is intentionally authorized.
3. **SSE.** In the deployed Vercel UI, send a short chat using a configured
   model and watch the browser network entry for `POST /api/chat`: it must
   remain open while streamed events arrive, then complete. This verifies the
   browser → Tailnet Serve → FastAPI streaming route; a health check alone does
   not.
4. **Persistence.** In the deployed UI, create a clearly labelled disposable
   chat, refresh the page, and reopen the chat. Its messages must remain.
   Delete the disposable chat afterwards. On the host, confirm its data is
   under `/var/lib/gigachad-bot/Documents`, not under `/opt/gigachad-bot`.
5. **Boundary.** Confirm `sudo ss -ltnp '( sport = :8001 )'` shows FastAPI only
   on `127.0.0.1:8001`, and `tailscale serve status` shows the private proxy.
   Do not test with Funnel.

## 6. Roll back safely

Keep the current Serve hostname and loopback port unchanged during a normal
application rollback; this lets clients retain
`https://<tailnet-hostname>/api`.

1. If an update is unhealthy, stop the backend:

   ```bash
   sudo systemctl stop gigachad-bot.service
   ```

2. Restore the last known-good application revision in `/opt/gigachad-bot` and
   its matching dependencies as `gigachad`. Restore the previous root-only
   environment file if configuration changed. Restore the durable-data backup
   only when the incident requires reverting data; do not overwrite newer user
   data merely to roll back code.
3. Start the backend and repeat the loopback and Tailnet health checks:

   ```bash
   sudo systemctl start gigachad-bot.service
   curl --fail --show-error http://127.0.0.1:8001/healthz
   ```

4. In Vercel, promote/redeploy the last known-good deployment and ensure its
   baked `NEXT_PUBLIC_API_BASE` remains the private Serve API URL. If its origin
   differs, restore the corresponding exact CORS allow-list and restart the
   backend.
5. If private ingress itself must be withdrawn immediately, run
   `tailscale serve off`; this removes Serve exposure but does not make the
   loopback backend public. Re-establish it later with the Serve command in
   section 2.

After rollback, repeat the health, CORS, SSE, and persistence checks before
resuming use.
