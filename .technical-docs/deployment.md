# Private deployment: Vercel + Tailscale Serve

This runbook deploys the Next.js frontend to Vercel while FastAPI and its local
runtime stay on one private host. Production runs from a code-isolated worktree
while deliberately sharing the deploying user's default Documents store with
development.

```text
Tailnet-enrolled browser ── HTTPS ──> <tailnet-hostname>
                                      Tailscale Serve ──> 127.0.0.1:8001 FastAPI
Vercel ── serves the Next.js bundle ──> browser
```

The Vercel project hosts the frontend; it is not an API proxy. The browser calls
`NEXT_PUBLIC_API_BASE` directly, so each browser using the app must be enrolled
in the Tailnet and allowed by its ACLs to reach the Serve HTTPS endpoint.

Do not use Tailscale Funnel. Public ingress is outside this design and requires
a separately designed authenticated gateway and security review. Keep FastAPI
bound to loopback.

## Values to choose

Replace these placeholders once and use the resulting values consistently:

| Placeholder | Meaning | Example |
| --- | --- | --- |
| `<tailnet-hostname>` | Stable MagicDNS name assigned to the host | `gigachad-backend.example.ts.net` |
| `<vercel-production-origin>` | Exact production browser origin, with scheme and no path or trailing slash | `https://gigachad-bot.vercel.app` |
| `<vercel-preview-origin>` | Exact preview origin to authorize temporarily | `https://gigachad-bot-git-main-team.vercel.app` |

The Vercel API base is always:

```text
https://<tailnet-hostname>/api
```

## 1. Prepare the private host

In the deploying user's shell, define:

```bash
REPO="$HOME/git/gigachad-bot"
PROD="$HOME/git/gigachad-bot-prod"
ENV="${XDG_CONFIG_HOME:-"$HOME/.config"}/gigachad-bot/env"
```

### Install and join Tailscale on Arch / Omarchy

Omarchy is Arch-based. Do not use Tailscale's universal installer here: it can
fail to identify Omarchy. Install the Arch package and enable its daemon:

```bash
sudo pacman -Syu tailscale
sudo systemctl enable --now tailscaled
sudo tailscale up --hostname=gigachad-backend
sudo tailscale status
```

Complete any interactive Tailnet login and confirm the Tailnet ACL permits the
intended users and devices to reach this host over HTTPS. Record the MagicDNS
hostname as `<tailnet-hostname>`. Tailscale Serve requires Tailnet HTTPS
certificates; complete any first-use Tailnet consent prompt. Do not enable
Funnel.

### Create the production worktree

Keep the regular checkout at `$REPO` for development.
Create a separate production worktree that tracks `origin/master`:

```bash
git -C "$REPO" fetch origin master
git -C "$REPO" worktree add --track -b production "$PROD" origin/master
```

For each reviewed production update, fast-forward that worktree and install its
production dependencies:

```bash
git -C "$PROD" fetch origin master
git -C "$PROD" merge --ff-only origin/master
cd "$PROD" && uv sync --python 3.12 --no-dev
```

Do not run `run.sh` in production: it installs development dependencies,
enables reload mode, and starts a development frontend. The production sync
includes the local OCR runtime. Leave `MINERU_SERVER_URL` unset to parse PDFs
locally; set it only when intentionally using a remote MinerU service.

Development and production intentionally use the same default Documents tree.
Do not configure `GIGACHAD_BASE_DIR` in the environment file, service unit, or
another production artifact. The worktree isolates code, not application state.

### Create the shared runtime environment file

There is exactly one private runtime configuration file for both development
and production: `$ENV`.

Create its parent directory and protect the file as the deploying user:

```bash
install -d -m 0700 "$(dirname "$ENV")"
umask 077
${EDITOR:?Set EDITOR} "$ENV"
chmod 0600 "$ENV"
```

Its syntax is deliberately strict. Only whitespace-only lines and lines that
begin with `#` are ignored. Every other line must be `NAME=value`, where `NAME`
is a shell variable name; values are literal and untrimmed, and may contain
`=`. Do not use `export`, shell quotes, command substitution, variable
expansion, or any other shell syntax. The shared loader used by both developer
launchers rejects every malformed record and never sources or evaluates this
file. It loads only variables absent from the inherited environment, so an
already-exported value takes precedence; duplicate non-inherited names resolve
in file order.

Use this as a starting file, substituting real values before use:

```dotenv
# Browser origins allowed to call the backend. Keep the production origin and
# both local frontend origins because this one file is shared with development.
GIGACHAD_CORS_ORIGINS=<vercel-production-origin>,http://127.0.0.1:2999,http://localhost:2999

# Local frontend API base. Vercel receives its own public build-time value below.
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8001/api

# Configure only host-local capabilities in use.
OLLAMA_BASE_URL=http://127.0.0.1:11434
VANE_URL=http://127.0.0.1:3001
SEARX_URL=http://127.0.0.1:8888
# EMBEDDING_MODEL=ollama/bge-m3:latest

# Provider credentials: uncomment only for providers in use.
# GEMINI_API_KEY=<secret>
# OPENROUTER_API_KEY=<secret>
# OPENAI_API_KEY=<secret>
```

| Variable | Required | Secret? | Purpose |
| --- | --- | --- | --- |
| `GIGACHAD_CORS_ORIGINS` | Yes | No | Comma-separated exact production and local browser origins. Do not use `*`. |
| `NEXT_PUBLIC_API_BASE` | For local frontend development | No | Local browser API base; it is intentionally public configuration. |
| `OLLAMA_BASE_URL` | When using local models or embeddings | No | Host-local Ollama endpoint. |
| `VANE_URL` | When using web search | No | Host-local Vane endpoint. |
| `SEARX_URL` | When using deep research | No | Host-local SearXNG endpoint. |
| `MINERU_SERVER_URL` | No | No | Remote MinerU endpoint. Leave unset for required local OCR. |
| `EMBEDDING_MODEL` | Optional | No | Vane embedding-model identifier. |
| Provider API keys | Only for their providers | **Yes** | Credentials used by the backend. |

URL and model settings are configuration, not secrets. Provider API keys are
secrets: keep them only in this `0600` private file; never commit or paste the
file, put keys in a ticket, or configure them in Vercel. `NEXT_PUBLIC_API_BASE`
is public browser configuration and is not a secret. Ensure host-managed Ollama,
Vane, and SearXNG are started before their dependent features and listen only on
loopback or another host-private interface.

### Provision the FastAPI systemd service

Install, reload, and enable the repository user unit. It validates the tracked
production runner and virtual-environment Uvicorn executable before installing
the unit, and deliberately does not start the backend:

```bash
"$PROD/deploy/install-systemd-service.sh"
```

To keep the user unit running after logout and start it at boot, enable
lingering once:

```bash
loginctl enable-linger "$USER"
```

After the shared environment file and production worktree are ready, start it
explicitly and verify the loopback health endpoint:

```bash
systemctl --user start gigachad-bot.service
systemctl --user status gigachad-bot.service
curl --fail --show-error http://127.0.0.1:8001/healthz
```

The user unit uses systemd's `%h` home-directory specifier. Its only
environment entry is the nonsecret configuration path:

```text
GIGACHAD_ENV_FILE=$ENV
```

It has no `EnvironmentFile`. Its `ExecStart` runs
`deploy/run-production-backend.sh`, whose strict `deploy/load-env.sh` loader
parses that shared file before it execs loopback Uvicorn. Provider values are
never parsed or loaded by the systemd manager; only the backend child receives
them.

It intentionally has neither `--reload` nor a non-loopback host. It retains
restart handling, a stop timeout, `NoNewPrivileges=true`, `PrivateTmp=true`, and
`UMask=0077` hardening. Do not bind Uvicorn to `0.0.0.0`, a LAN address, or the
Tailscale address.

**Only one backend may run at a time.** The development backend and the systemd
backend both bind `127.0.0.1:8001` and share the same Documents state. Stop the
systemd service before using a development backend, and stop the development
backend before starting or restarting the service.

If the loopback health check fails, inspect the service rather than exposing
another listener:

```bash
journalctl --user -u gigachad-bot.service -n 100 --no-pager
```

## 2. Publish the private HTTPS ingress

On the backend host, inspect any existing Serve configuration before changing
it. This runbook assumes the host is dedicated to this app at its Serve root:

```bash
tailscale serve status
```

Configure the persistent private HTTPS root proxy with the tracked helper. It
preserves `/api` and `/healthz`, so no path rewrite is needed:

```bash
"$PROD/deploy/tailscale-serve.sh"
tailscale serve status
```

The helper runs exactly `tailscale serve --bg --https=443 http://127.0.0.1:8001`.

The status output should show a proxy to `http://127.0.0.1:8001` available only
inside the Tailnet. `--bg` persists the Serve configuration across Tailscale
restarts and reboots. Use `tailscale serve`, never `tailscale funnel`.

From a different device enrolled in the Tailnet and permitted by its ACL,
verify the external leg:

```bash
curl --fail --show-error https://<tailnet-hostname>/healthz
curl --fail --show-error https://<tailnet-hostname>/api/config
```

Do not use a device outside the Tailnet for this check or supply a public
fallback URL.

## 3. Configure the Vercel frontend

In the Vercel project settings:

| Setting | Value |
| --- | --- |
| Framework preset | Next.js (auto-detected is fine) |
| Root Directory | `src/frontend` |
| Install Command | `npm ci` |
| Build Command | `npm run build` |
| Output Directory | Leave at Vercel's Next.js default; do not configure a static export directory. |

Add this variable for Production and Preview before the corresponding Vercel
deployment:

```text
NEXT_PUBLIC_API_BASE=https://<tailnet-hostname>/api
```

This Vercel value is public build-time configuration, not a credential. A
change requires a new Vercel deployment because it is bundled into browser
code. Vercel does not need Tailnet access to build the frontend; the browser,
not Vercel, makes API calls.

The shared environment file must retain the exact production Vercel origin and
both local origins in `GIGACHAD_CORS_ORIGINS`. Origins have scheme and host
only—no `/api`, path, or trailing slash. Restart the backend after changing its
environment file:

```bash
systemctl --user restart gigachad-bot.service
```

### Preview deployments and CORS

A preview bundle calls the same private API base, so its exact browser origin
must also be temporarily added to `GIGACHAD_CORS_ORIGINS` before testing it.
The allow-list is exact and comma-separated; Vercel preview wildcards are not
supported.

1. Obtain the complete `https://…vercel.app` preview origin.
2. Append that exact origin to the shared environment file.
3. Restart `gigachad-bot.service`.
4. Deploy and test from a Tailnet-enrolled browser.
5. Remove the preview origin and restart the service when it is no longer
   needed.

Do not replace the allow-list with `*` for previews. CORS is not a substitute
for Tailnet ACLs, but it prevents arbitrary web origins from reading API
responses in an enrolled browser.

## 4. Deployment sequence

For a first deployment or an update:

1. Create or fast-forward the production worktree from `origin/master`, then
   run `uv sync --python 3.12 --no-dev` in it. Do not run `run.sh`.
2. Confirm `$ENV` has the exact production and local CORS origins, host-local
   dependency URLs, and only the required provider secrets. Leave
   `GIGACHAD_BASE_DIR` and `MINERU_SERVER_URL` unset.
3. Start or restart `gigachad-bot.service` and confirm its loopback health
   check.
4. Run `"$PROD/deploy/tailscale-serve.sh"` and confirm Tailnet HTTPS health
   from another enrolled device.
5. Configure Vercel's root directory, build settings, and public
   `NEXT_PUBLIC_API_BASE`, then deploy production.
6. From an enrolled browser, open the Vercel production URL and use the app. A
   browser outside the Tailnet cannot reach its API by design.

Before an update that could affect stored state, take a non-destructive backup
of the shared Documents tree. Do not use `--delete` on the backup command or
mistake the production worktree for application data.

## 5. Verify the deployed path

Perform browser-facing checks from a device enrolled in the Tailnet and allowed
by its ACL.

1. **Health and path routing.** Confirm both
   `https://<tailnet-hostname>/healthz` and
   `https://<tailnet-hostname>/api/config` return successfully.
2. **CORS.** Check the production browser origin is accepted:

   ```bash
   curl --silent --show-error --dump-header - --output /dev/null \
     --request OPTIONS https://<tailnet-hostname>/api/config \
     --header 'Origin: <vercel-production-origin>' \
     --header 'Access-Control-Request-Method: GET' \
     --header 'Access-Control-Request-Headers: content-type'
   ```

   The response must include
   `access-control-allow-origin: <vercel-production-origin>`. The local
   `http://127.0.0.1:2999` and `http://localhost:2999` origins are also
   intentionally accepted for development. Repeat with a preview origin only
   while it is intentionally authorized.
3. **SSE.** In the deployed Vercel UI, send a short chat using a configured
   model and confirm `POST /api/chat` stays open while streamed events arrive,
   then completes. This verifies the browser → Tailnet Serve → FastAPI
   streaming route; a health check alone does not.
4. **Persistence.** Create a clearly labelled disposable chat, refresh, and
   reopen it. Its messages must remain. Delete the disposable chat afterwards;
   it belongs in the shared default Documents tree, not either code worktree.
5. **Boundary.** Confirm `sudo ss -ltnp '( sport = :8001 )'` shows FastAPI only
   on `127.0.0.1:8001` and `tailscale serve status` shows the private proxy. Do
   not test with Funnel.

## 6. Roll back safely

Keep the Serve hostname and loopback port unchanged during a normal application
rollback so clients retain `https://<tailnet-hostname>/api`.

1. Stop the backend:

   ```bash
   systemctl --user stop gigachad-bot.service
   ```

2. Restore the last known-good `origin/master` revision in `$PROD`, run its
   matching production sync, and restore the previous shared environment file
   only if configuration changed. Restore a Documents backup only when the
   incident requires reverting data; do not overwrite newer user data merely to
   roll back code.
3. Start the backend and repeat loopback and Tailnet health checks:

   ```bash
   systemctl --user start gigachad-bot.service
   curl --fail --show-error http://127.0.0.1:8001/healthz
   ```

4. In Vercel, promote or redeploy the last known-good deployment and ensure its
   baked `NEXT_PUBLIC_API_BASE` remains the private Serve API URL. If its origin
   differs, restore the corresponding exact CORS allow-list and restart the
   backend.
5. If private ingress must be withdrawn immediately, run `tailscale serve off`.
   This removes Serve exposure without making the loopback backend public.
   Re-establish it later with the Serve command in section 2.

After rollback, repeat the health, CORS, SSE, and persistence checks before
resuming use.
