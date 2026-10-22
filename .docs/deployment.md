# Private deployment: Vercel + Tailscale Serve

## Deployment model

- Prefer serving the UI and `/api` from one private HTTPS origin for browser
  compatibility. See [README](../README.md).
- Use this split-origin runbook for existing Vercel + private Tailscale deployments.

| Boundary | Contract |
| --- | --- |
| Vercel | Serve the frontend bundle, not an API proxy |
| Browser | Call the private API directly. Require Tailnet enrollment and ACL permission. |
| Backend | Bind loopback. Never use Funnel or a public fallback. |
| Production isolation | Separate code worktree and PostgreSQL container/volume |
| Shared state | Development/production share the default Documents/PDF mirror, not PostgreSQL data. Code rollback does not roll back state. |

## Values to choose

| Placeholder | Meaning |
| --- | --- |
| `<tailnet-hostname>` | Host's stable MagicDNS name, e.g. `gigachad-backend.example.ts.net` |
| `<vercel-production-origin>` | Exact HTTPS production origin, without path or trailing slash |
| `<vercel-preview-origin>` | Exact preview origin, authorized only while testing |

- Set the split-origin API base to `https://<tailnet-hostname>/api`.
- Define paths in the deploying user's shell:

```bash
REPO="$HOME/git/gigachad-bot"
PROD="$HOME/git/gigachad-bot-prod"
ENV="${XDG_CONFIG_HOME:-"$HOME/.config"}/gigachad-bot/env"
```

## 1. Prepare the private host

### Install and join Tailscale on Arch / Omarchy

- Install the Arch package. The universal installer may not recognize Omarchy.

```bash
sudo pacman -Syu tailscale
sudo systemctl enable --now tailscaled
sudo tailscale up --hostname=gigachad-backend
sudo tailscale status
```

- Complete login and permit intended devices through Tailnet ACLs.
- Record the MagicDNS hostname and enable HTTPS certificates when prompted.

### Create the production worktree

Keep development at `$REPO`. Create production from `origin/master`:

```bash
git -C "$REPO" fetch origin master
git -C "$REPO" worktree add --track -b production "$PROD" origin/master
```

For each reviewed update:

```bash
git -C "$PROD" fetch origin master
git -C "$PROD" merge --ff-only origin/master
cd "$PROD" && uv sync --python 3.12 --locked --no-dev
```

- Never use `run.sh` in production: it installs dev dependencies and starts reload
  mode and a development frontend. Services do not resync Python on restart.
- Production sync includes local OCR. Leave `MINERU_SERVER_URL` unset unless
  intentionally using remote OCR.
- Never set `GIGACHAD_BASE_DIR` in production artifacts. Keep the shared default
  Documents mirror, the worktree isolates code rather than that filesystem data.

### Dependency supply chain

- Install from `uv.lock`, `src/frontend/package-lock.json`, and
  `src/c4/package-lock.json` with `uv sync --locked` and `npm ci`.
- Development launchers repair missing/outdated locks with `uv lock` without
  `--upgrade` or `npm install --package-lock-only`, then retry installation once.
  Review and commit regenerated locks with manifests. Production never regenerates
  locks, other installation failures stop startup.
- Retain the seven-day release cooldown (`exclude-newer`, `.npmrc`'s
  `min-release-age`) and disabled npm install scripts (`ignore-scripts`). Review
  required scripts before explicitly running them. Neither cooldown nor automated
  lock resolution replaces security review. Review Dependabot diffs for all three locks.

### Create the shared runtime configuration file

| File | Contents |
| --- | --- |
| `$ENV` | Shared development/production nonsecret configuration |
| `~/.secrets` | Backend credentials as `export NAME=value`: provider keys, `BRAVE_API_KEY`, `POSTGRES_PASSWORD_DEV`, `POSTGRES_PASSWORD_PROD` |

```bash
install -d -m 0700 "$(dirname "$ENV")"
umask 077
${EDITOR:?Set EDITOR} "$ENV"
chmod 0600 "$ENV"
```

- Write literal, untrimmed `NAME=value` records. Allow `=` inside values,
  whitespace-only lines, and comments starting with `#`.
- Do not use `export`, quotes, expansion, or command substitution in `$ENV`.
- Preserve loader semantics: no evaluation, malformed-record rejection,
  inherited-variable precedence.
- Source `~/.secrets` first. Its values win without entering the systemd manager.

Example, substituting your production origin:

```dotenv
GIGACHAD_CORS_ORIGINS=<vercel-production-origin>,http://127.0.0.1:2999,http://localhost:2999
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8001/api
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

- `GIGACHAD_CORS_ORIGINS`: required exact origins. Production rejects `*` and empty
  lists because Tailnet identity alone would let arbitrary pages act as the user.
- `NEXT_PUBLIC_API_BASE`: public, build-time frontend configuration. The example
  serves local development, configure Vercel's private HTTPS value separately.
- `OLLAMA_BASE_URL`: needed only for Ollama. `MINERU_SERVER_URL`: optional remote OCR.
- URLs and model names are configuration, not credentials. Brave requests run
  server-side, its API key must never reach the browser.

### Provision the systemd services

Install the backend and optional OMP model-source supervisor without starting them:

```bash
"$PROD/deploy/install-systemd-service.sh"
loginctl enable-linger "$USER"
```

- Require installer validation of production runners and the venv's Uvicorn.
- Enable lingering to retain services after logout and start them at boot.
- Start only after configuration is ready:

```bash
systemctl --user start gigachad-bot.service
systemctl --user status gigachad-bot.service gigachad-bot-omp.service
curl --fail --show-error http://127.0.0.1:8002/healthz
```

- `gigachad-bot.service` runs `run-backend.sh --prod`, loading credentials/config,
  starting or reusing production PostgreSQL, and serving loopback Uvicorn without
  reload. Keep `NoNewPrivileges`, `PrivateTmp`, `UMask=0077`, restart handling,
  and stop timeout. Never bind FastAPI to a LAN, Tailnet, or wildcard address.
- Units use `%h` and `GIGACHAD_ENV_FILE`, not `EnvironmentFile`, to avoid systemd
  parsing credentials. The OMP unit also supplies a `PATH` including mise shims.
- The backend orders startup after `gigachad-bot-omp.service`. Its runner starts
  or reuses OMP's auth broker/gateway, hides the model source when OMP or its
  credentials are absent, and exits successfully in that case. Credential root:
  `~/.omp/agent/agent.db`, overridden by `PI_CONFIG_DIR`.
- OMP runner: `deploy/run-production-omp.sh` → `run-omp.sh`. Override binds with
  `GIGACHAD_OMP_BROKER_BIND` and `GIGACHAD_OMP_GATEWAY_BIND` if needed.
- The backend starts the stateless LikeC4 service, points `C4_SERVICE_URL` at it,
  and stops it on exit. Each start uses `npm ci` against its lockfile. If unavailable,
  architecture routes return 503, other features remain available.

| Service | Development | Production |
| --- | --- | --- |
| FastAPI | `127.0.0.1:8001` | `127.0.0.1:8002` |
| LikeC4 | `127.0.0.1:8011` | `127.0.0.1:8012` |
| PostgreSQL | `gigachad-dev` container/volume | `gigachad-prod` container/volume |
| OMP auth broker / gateway | `127.0.0.1:8765` / `127.0.0.1:4000` | Same defaults, reuse existing listeners |

Inspect failures rather than exposing another listener:

```bash
journalctl --user -u gigachad-bot.service -n 100 --no-pager
journalctl --user -u gigachad-bot-omp.service -n 100 --no-pager
```

### Provision the code sandbox

- Run workspace agents and image-based plot requests through OMP in
  [Agents-in-a-box](https://github.com/patrickab/Agents-in-a-box), using rootless Docker and gVisor.

```bash
git clone https://github.com/patrickab/Agents-in-a-box.git "$HOME/git/Agents-in-a-box"
cd "$HOME/git/Agents-in-a-box"
./scripts/setup_agent_sandbox.sh
cd "$PROD" && uv run agent-sandbox doctor --profile gigachad
```

- Setup installs Docker/runsc, builds the image, and configures host services.
  Existing installations need only `./scripts/setup_host_services.sh`.
- Ollama stays on loopback, a persistent alias/socket proxy exposes it to sandbox
  containers at `10.200.200.1:11434`.
- The bundled `gigachad` profile uses absolute paths for the real OMP binary,
  Python interpreters, and `~/.omp`. Do not mount a mise shim or edit installed
  profiles, `uv sync` replaces them. Put custom profiles in a directory named by
  `AGENT_SANDBOX_PROFILE_PATH`, searched before bundled profiles. Doctor must pass.

### Provision omp-web (omp tabs)

- Use private [patrickab/ompweb](https://github.com/patrickab/ompweb), branch `custom`.
- Put `omp` on `PATH` and authorize Git credentials for the fork.

```bash
git clone --branch custom https://github.com/patrickab/ompweb.git "$HOME/git/ompweb"
git -C "$HOME/git/ompweb" remote add upstream https://github.com/kahme247/ompweb
git -C "$HOME/git/ompweb" remote set-url --push upstream no-push
```

- Use `upstream` for rebasing. Follow the fork's `custom/AGENTS.md`.
- Let `run-omp-web.sh` serve a detached worktree and resolve its ref on each start.
- Install/build initially and rebuild on source changes.
- Restart after updates: an existing listener is reused.

| Variable | Default / purpose |
| --- | --- |
| `GIGACHAD_OMP_WEB_REPO` | `~/git/ompweb`, fork checkout |
| `GIGACHAD_OMP_WEB_DIR` | `~/.local/share/gigachad-bot/ompweb`, served worktree |
| `GIGACHAD_OMP_WEB_REF` | `custom`, use `origin/custom` in production |
| `GIGACHAD_OMP_WEB_BIND` | `127.0.0.1:30177` |
| `GIGACHAD_OMP_WEB_FRAME_ANCESTORS` | Local `:2999` origins, production units default to `GIGACHAD_CORS_ORIGINS` |
| `GIGACHAD_OMP_WEB_URL` | `http://127.0.0.1:30177`, upstream for backend/gate |
| `GIGACHAD_OMP_WEB_FRAME_URL` | Defaults to upstream URL, use the gate's HTTPS URL in production |
| `GIGACHAD_OMP_WEB_GATE_PORT` | `30179`, loopback identity gate |
| `GIGACHAD_OMP_WEB_AUTH` | Optional trusted upstream auth header |
| `GIGACHAD_OMP_WEB` | `1`, set `0` to skip development startup |

#### Production: units and the Tailnet gate

```text
private Serve :8443 -> identity gate 127.0.0.1:30179 -> omp-web 127.0.0.1:30177
```

- Treat OMP as host-user command execution. Never publish omp-web directly.
- Require host capability in `src/backend/omp_web_gate.py` before forwarding.
- Do not rely on omp-web's `SameSite=Lax` password cookie for a cross-site iframe.
- Configure allowed frame origins. Missing origins produce a blank frame.

Add to `$ENV`:

```dotenv
GIGACHAD_OMP_WEB_FRAME_URL=https://<tailnet-hostname>:8443
GIGACHAD_OMP_WEB_REF=origin/custom
```

```bash
"$PROD/deploy/install-systemd-service.sh"
systemctl --user start gigachad-bot-omp-web.service gigachad-bot-omp-web-gate.service
"$PROD/deploy/tailscale-serve.sh" 30179 8443
systemctl --user restart gigachad-bot.service
```

- Run both units through `deploy/run-production-omp-web.sh` with `server` or `gate`.
- Keep normal `/tmp`, sudo, and file modes available to agents. The server
  deliberately lacks sandboxing flags.
- Refuse non-host logins at the gate.
- Find login names with `tailscale status --json`, field `User.*.LoginName`.

#### Update omp-web

- Keep protocol, palette, Q&A cards, and Markdown in `src/frontend/packages/chat-ui/`.
- Install versioned `custom/vendor/*.tgz` through the fork's lockfile.
  Require no sibling checkout and no repacking during ordinary deployment.
- For shared-package changes, bump its version and run from `src/frontend`:

```bash
npm run sync:ompweb
npm run check:ompweb
```

- Use project npm (`.venv/bin` first on `PATH`) for reproducible tarball bytes.
- Commit fork tarball, manifest, and lock together. Deploy matching sides.
- Check `EMBED_PROTOCOL_VERSION`: no runtime handshake exists, so mismatches
  silently drop messages.

```bash
grep -h 'EMBED_PROTOCOL_VERSION =' \
  "$REPO/src/frontend/packages/chat-ui/src/protocol.ts" \
  "$HOME/.local/share/gigachad-bot/ompweb/node_modules/@gigachad/chat-ui/dist/protocol.js"
```

1. Ensure no session is running before restart, which ends agent processes.
   `curl -s http://127.0.0.1:30177/api/sessions` must show no `runningSessionIds`.
2. Fetch rather than pull the rebased/force-pushed fork:
   `git -C "$HOME/git/ompweb" fetch origin custom`.
3. Restart `gigachad-bot-omp-web.service` and verify
   `curl -sf http://127.0.0.1:30177/login`. The worktree follows its configured ref.
4. Update Gigachad to the matching commit and reload the frontend.
5. Smoke session loading, history, turn rendering, and tab switching. Test prompt
   sending, Stop, model/thinking controls, and completion only in an explicitly
   approved disposable session, never an existing conversation.

**Known limitation:** after a fresh frame load, a session created by the first tab
uses omp-web's last folder rather than that tab's folder. Later tabs are unaffected.

### Provision the agent factory

| Concern | Contract |
| --- | --- |
| Unit | `gigachad-bot-factory.service` runs `run-factory.sh --prod` on loopback `:8090` |
| Dependencies | Agents-in-a-box checkout and production PostgreSQL `factory` database |
| Backend upstream | Set `GIGACHAD_FACTORY_URL` to the engine base URL, default `http://127.0.0.1:8090` |
| Startup | Provision missing checkout/dependencies/config/database. Preserve existing state. `run.sh` never starts this production engine. |
| Previews | Disable in production: remote browsers cannot reach host-loopback ports 8100–8199. Boards, diffs, messages, and merges remain usable. |

- Configure once per production host, before service startup:

```bash
install -d -m 0700 ~/.config/agent-sandbox/factory
printf 'preview: false\n' > ~/.config/agent-sandbox/factory/defaults.yaml
"$PROD/deploy/install-systemd-service.sh"
systemctl --user start gigachad-bot-factory.service
curl -sf http://127.0.0.1:8090/v1/repos
```

- Update with `git -C ~/git/Agents-in-a-box pull --ff-only`.
- Restart `gigachad-bot-factory.service` only when no task is running.
  Restart ends an active turn.
- Follow [Agent factory](system-overview.md#agent-factory) for repository contracts.

## 2. Publish the private HTTPS ingress

- Inspect existing Serve configuration. Require this app to own the host's Serve root.
- Use the helper to preserve `/api` and `/healthz`:

```bash
tailscale serve status
"$PROD/deploy/tailscale-serve.sh" 8002
tailscale serve status
```

- Expect persistent private HTTPS from the helper:
  `tailscale serve --bg --https=443 --accept-app-caps=gigachad.local/cap/host http://127.0.0.1:8002`.
- Verify from another permitted Tailnet device. Never add a public fallback:

```bash
curl --fail --show-error https://<tailnet-hostname>/healthz
curl --fail --show-error https://<tailnet-hostname>/api/config
```

### Host access

| Identity | Authority |
| --- | --- |
| Host-capable login | Host-user execution through OMP, Neovim, factory, and files outside mounts, excluding credential stores |
| Other login | Own mounted roots only |
| Trusted ingress | Tailscale v1.92+ accepts `gigachad.local/cap/host` and forwards `Tailscale-App-Capabilities`, stripping client-supplied copies |
| Backend / gate | Trust only private ingress to loopback (`backend/identity.py`) |

- Grant host access in Tailnet policy, not `$ENV`.
- Add Access controls without removing existing rules:

```jsonc
"hosts": { "gigachad-backend": "<backend tailnet IP>" },
"grants": [
  // ...existing grants...
  {
    "src": ["<your Tailscale login>"],
    "dst": ["gigachad-backend"],
    "app": { "gigachad.local/cap/host": [{}] }
  }
]
```

- Fail closed on missing grants or unsupported Serve flags.
- Check `/api/config`: `omp_web_url` is set for granted logins, `null` otherwise.

## 3. Configure the Vercel frontend

| Setting | Value |
| --- | --- |
| Framework | Next.js |
| Root Directory | `src/frontend` |
| Install / build | `npm ci` / `npm run build` |
| Output Directory | Vercel default, not a static-export directory |
| Production/Preview environment | `NEXT_PUBLIC_API_BASE=https://<tailnet-hostname>/api` |

- Redeploy after changing the API base: it is baked into the bundle.
- Give Tailnet access to browsers, not Vercel. Browsers make API requests.
- Retain exact production/local origins in `$ENV`'s `GIGACHAD_CORS_ORIGINS`.
- Restart after backend configuration changes:

```bash
systemctl --user restart gigachad-bot.service
```

### Preview deployments and CORS

1. Append the exact preview origin and restart the backend.
2. Test from an enrolled browser.
3. Remove the origin and restart when finished.

- Reject wildcard shortcuts, including `*`, in production.
- Keep Tailnet ACLs. CORS complements identity, never replaces it.

## 4. Deployment sequence

1. Update the production worktree and install locked dependencies.
2. Confirm nonsecret `$ENV` and separate `~/.secrets`. Leave `GIGACHAD_BASE_DIR`
   unset, leave `MINERU_SERVER_URL` unset when using local OCR.
3. Provision sandbox/optional OMP/factory services, then pass sandbox doctor.
4. Start or restart the backend and verify loopback health.
5. Publish private Serve ingress and verify it from another permitted device.
6. Deploy the frontend with the matching API base and authorized origin.
7. Perform the verification below before use.

- Back up PostgreSQL and the shared Documents/PDF mirror before state-affecting updates.
- Avoid destructive backup synchronization.
- Do not treat code worktrees as application-data backups.

## 5. Verify the deployed path

From a permitted Tailnet device:

1. Check HTTPS `/healthz` and `/api/config`.
2. Check CORS for the exact authorized browser origin:

   ```bash
   curl --silent --show-error --dump-header - --output /dev/null \
     --request OPTIONS https://<tailnet-hostname>/api/config \
     --header 'Origin: <vercel-production-origin>' \
     --header 'Access-Control-Request-Method: GET' \
     --header 'Access-Control-Request-Headers: content-type'
   ```

   Require `access-control-allow-origin: <vercel-production-origin>`. Local
   `:2999` origins remain intentional, preview origins only while authorized.
3. Send a disposable chat and verify `/api/chat` SSE streams and completes.
   Health checks alone do not prove streaming.
4. Refresh and reopen that chat to verify persistence, then delete it.
5. Confirm `sudo ss -ltnp '( sport = :8002 )'` shows loopback only and
   `tailscale serve status` shows private ingress. Never test with Funnel.

## 6. Roll back safely

Keep the private hostname and loopback port stable:

1. `systemctl --user stop gigachad-bot.service`.
2. Restore a known-good code revision in `$PROD` and its matching locked
   dependencies. Restore configuration only if changed, restore database or
   mirror backups only when reverting data is necessary. Do not overwrite newer
   user data merely to roll back code.
3. Start the service and repeat loopback/Tailnet health checks:

   ```bash
   systemctl --user start gigachad-bot.service
   curl --fail --show-error http://127.0.0.1:8002/healthz
   ```

4. Restore the matching frontend deployment, preserving its private API base and
   exact CORS origin. Restart the backend if its origin configuration changes.
5. To withdraw ingress immediately, use `tailscale serve off`. Restore it with
   the section 2 helper, never by exposing the backend directly.

Repeat health, CORS, SSE, and persistence verification before resuming use.
