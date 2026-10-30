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
- `run.sh` syncs Python once (with the `sandbox-plot` extra) and exports
  `GIGACHAD_DEPS_SYNCED=1`, so `run-backend.sh`/`run-frontend.sh` skip their own
  sync. `uv sync` is exact by default and removes extras it was not given, so a
  standalone `uv sync --locked` drops `polars`. Run tests with `uv run --no-sync`.
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
- Exact origins also enable credentialed CORS. A startup read that times out (15 s)
  switches later API calls to credentialed requests, which Chrome sends over a separate
  connection pool. This recovers from a dead HTTP/2 connection that Tailscale Serve
  dropped without a reset (tailscaled logs `packet was not handled`).
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
| omp-web / its gate | `127.0.0.1:30177` / - | Same omp-web, gate `127.0.0.1:30179` (Serve `:8443`) |
| LearnHarness / LiteLLM proxy / gate | `127.0.0.1:30181` / `127.0.0.1:30183` / - | Same defaults, one instance per machine, gate `127.0.0.1:30182` (Serve `:8444`) |

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

- Use unmodified upstream [kahme247/ompweb](https://github.com/kahme247/ompweb).
  gigachad-bot's side is the overlay `ompweb-overlay/` in this repository.
- Put `omp` on `PATH`.

```bash
git clone https://github.com/kahme247/ompweb.git "$HOME/git/ompweb_original"
```

- Serve a ref containing the upstream seams the overlay uses. Until PR #332
  (issue #314) is merged that is the local branch `integration_patrick_ab`
  (upstream `main` plus #332's commits), which a fresh clone lacks: create it
  from `patrickab/ompweb-fork` branch `refactor/314-ui-customization-seams`,
  then use `origin/main`.
- `run-omp-web.sh` serves a detached worktree reset to the ref on each start,
  applies `ompweb-overlay/seams.patch`, copies the overlay to `custom/` and packs
  `@gigachad/chat-ui` into it. The source checkout stays untouched.
- Packing compiles with `src/frontend`'s TypeScript and types, so production
  needs `src/frontend/node_modules`. `redeploy.sh` installs it (`npm ci
  --ignore-scripts`) when missing or when the lockfile changed.
- Before serving, `run-omp-web.sh` fetches `origin` and, when the ref branch
  lacks `origin/main`, rebases a copy onto it in a temporary worktree, runs the
  overlay checks (`tsc`, overlay node tests) and moves the branch (and its clean
  worktree) there. Local only, never pushed. A conflict, failed check, dirty
  branch worktree or offline start keeps the branch and prints one
  `omp-web auto-update:` line why; the start goes on. Details:
  `.docs/integration-ompweb.md`, "Automatic upstream rebase".
- Install/build initially and rebuild on source changes.
- Restart after updates: an existing listener is reused, its update applies at
  its next start.

| Variable | Default / purpose |
| --- | --- |
| `GIGACHAD_OMP_WEB_REPO` | `~/git/ompweb_original`, upstream checkout |
| `GIGACHAD_OMP_WEB_DIR` | `~/.local/share/gigachad-bot/ompweb-upstream`, served worktree |
| `GIGACHAD_OMP_WEB_REF` | `integration_patrick_ab`, a remote-tracking ref (`origin/main`) is fetched by `redeploy.sh` |
| `GIGACHAD_OMP_WEB_BIND` | `127.0.0.1:30177` |
| `GIGACHAD_OMP_WEB_FRAME_ANCESTORS` | Local `:2999` origins, production units default to `GIGACHAD_CORS_ORIGINS` |
| `GIGACHAD_OMP_WEB_URL` | `http://127.0.0.1:30177`, upstream for backend/gate |
| `GIGACHAD_OMP_WEB_FRAME_URL` | Defaults to upstream URL, use the gate's HTTPS URL in production |
| `GIGACHAD_OMP_WEB_GATE_PORT` | `30179`, loopback identity gate |
| `GIGACHAD_OMP_WEB_AUTH` | Optional trusted upstream auth header |
| `GIGACHAD_OMP_WEB` | `1`, set `0` to skip development startup |
| `GIGACHAD_AUTO_UPDATE` | `1`, set `0` to serve the ref as is (no fetch, no rebase) |

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

- Keep protocol, palette, Q&A cards, and Markdown in `src/frontend/packages/chat-ui/`,
  the embed UI in `ompweb-overlay/`. Both ship with the gigachad-bot commit, so
  `redeploy.sh` restarts omp-web when they or the served ref change.
- The protocol is unversioned: a stale omp-web silently drops messages it does
  not know until restarted.

Manual update:

1. Ensure no session is running before restart, which ends agent processes.
   `curl -s http://127.0.0.1:30177/api/sessions` must show no `runningSessionIds`.
2. Update Gigachad, whose overlay the restart applies. The restart fetches
   `origin` and rebases `integration_patrick_ab` onto `origin/main` itself
   (fetch by hand for a remote-tracking ref such as `origin/main`).
3. Restart `gigachad-bot-omp-web.service` and verify
   `curl -sf http://127.0.0.1:30177/login`. The worktree follows its configured ref;
   `journalctl --user -u gigachad-bot-omp-web.service | grep auto-update` shows
   whether the rebase applied or why it was kept.
4. Reload the frontend.
5. Smoke session loading, history, turn rendering, and tab switching. Test prompt
   sending, Stop, model/thinking controls, and completion only in an explicitly
   approved disposable session, never an existing conversation.

**Known limitation:** after a fresh frame load, a session created by the first tab
uses omp-web's last folder rather than that tab's folder. Later tabs are unaffected.

### Provision learn mode (LearnHarness)

Units, environment, gate, backup and restore: `.docs/integration-learnharness.md`, "Production deployment".

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
- The frame gates get their own HTTPS ports with the same helper: omp-web
  `30179 8443`, LearnHarness `30182 8444` (see their sections). Serve accepts any
  HTTPS port, only Funnel is limited to 443, 8443 and 10000.
- Verify from another permitted Tailnet device. Never add a public fallback:

```bash
curl --fail --show-error https://<tailnet-hostname>/healthz
curl --fail --show-error https://<tailnet-hostname>/api/config
```

### Host access

| Identity | Authority |
| --- | --- |
| Host-capable login | Host-user execution through OMP, Neovim, factory, and files outside mounts, excluding credential stores. Learn mode (LearnHarness) |
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
- Check `/api/config`: `omp_web_url` and `learnharness_url` are set for granted
  logins, `null` otherwise.

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
3. Provision sandbox/optional OMP/factory/learn mode services, then pass sandbox doctor.
4. Start or restart the backend and verify loopback health.
5. Publish private Serve ingress and verify it from another permitted device.
6. Deploy the frontend with the matching API base and authorized origin.
7. Perform the verification below before use.

- Back up PostgreSQL and the shared Documents/PDF mirror before state-affecting updates.
  LearnHarness data has its own daily backup (section 1, "Back up and restore
  LearnHarness data"), and the launcher snapshots it before a ref bump.
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
   user data merely to roll back code. Applied `db/schema/` steps stay applied:
   older code ignores additive ones such as `0011` (document `meta` and
   `sha256` columns, `STORAGE EXTERNAL`), and chat metadata it leaves stale is
   rederived from the chat body once newer code runs again.
3. Start the service and repeat loopback/Tailnet health checks:

   ```bash
   systemctl --user start gigachad-bot.service
   curl --fail --show-error http://127.0.0.1:8002/healthz
   ```

4. Restore the matching frontend deployment, preserving its private API base and
   exact CORS origin. Restart the backend if its origin configuration changes.
5. To withdraw ingress immediately, use `tailscale serve off`. Restore it with
   the section 2 helper, never by exposing the backend directly.

Schema steps stay applied after a code rollback. Older code runs against them
unchanged: step 0013 only deleted superseded `changes` rows (replay output is
identical) and added an index, older code merely stops pruning. The sync
stream tolerates a frontend and backend from different revisions in both
directions, an older backend just replays from `since=0` again.

A rolled-back LearnHarness ref serves data a newer commit may have changed, and
upstream keeps no backward compatibility for it. Stop the LearnHarness unit, move
its home aside, copy the matching `snapshots/<old-sha>-<utc>/` back into a new
home (its `data/`, `auth.json` and credential files) and write `<old-sha>` to its
`served-commit`, or restore a backup taken under that commit. Then start it
(`.docs/integration-learnharness.md`, "Back up and restore LearnHarness data").

Repeat health, CORS, SSE, and persistence verification before resuming use.
