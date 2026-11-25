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

Fetch over HTTPS, push over SSH. The settings menu's redeploy runs without an SSH agent, and both repositories are public:

```bash
git -C "$REPO" remote set-url origin https://github.com/patrickab/gigachad-bot.git
git -C "$REPO" remote set-url --push origin git@github.com:patrickab/gigachad-bot.git
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
- `run.sh` syncs Python once and exports `GIGACHAD_DEPS_SYNCED=1`, so
  `run-backend.sh`/`run-frontend.sh` skip their own sync, which would race it on the
  shared venv. The default groups (`dev`, `ocr`, `sandbox-plot`) come with every
  `uv sync` and `uv run`, so no sync drops `polars` (production's `--no-dev` drops
  only `dev`). `pyproject.toml` requires uv 0.13 or newer (`required-version`): an
  older uv refuses instead of rewriting `uv.lock` to an older revision.
- Development npm installs (`install_deps.py`) run `npm ci` only when `package.json`,
  the lock or `.npmrc` changed since the last successful install (stamp in
  `node_modules/.gigachad-npm-inputs.sha256`), as `npm ci` rewrites `node_modules`
  (25s+ for the frontend). Production still runs `npm ci` on every start.
- `run.sh` installs the frontend once up front and exports
  `GIGACHAD_FRONTEND_DEPS_INSTALLED=1`, so `run-frontend.sh` skips it. A running
  install breaks the chat-ui pack (`tsc` from `src/frontend/node_modules`) and makes
  the backend's reloader restart. Started separately, run `run-frontend.sh` to
  completion of its install before the sidecars.
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

Install the backend, pi-host and the other units without starting them:

```bash
"$PROD/deploy/install-systemd-service.sh"
loginctl enable-linger "$USER"
```

- Require installer validation of production runners and the venv's Uvicorn.
- Enable lingering to retain services after logout and start them at boot.
- Start only after configuration is ready:

```bash
systemctl --user start gigachad-bot-pi-host.service gigachad-bot.service
systemctl --user status gigachad-bot.service gigachad-bot-pi-host.service
curl --fail --show-error http://127.0.0.1:8002/healthz
```

- `gigachad-bot.service` runs `run-backend.sh --prod`, loading credentials/config,
  starting or reusing production PostgreSQL, and serving loopback Uvicorn without
  reload. Keep `NoNewPrivileges`, `PrivateTmp`, `UMask=0077`, restart handling,
  and stop timeout. Never bind FastAPI to a LAN, Tailnet, or wildcard address.
- Units use `%h` and `GIGACHAD_ENV_FILE`, not `EnvironmentFile`, to avoid systemd
  parsing credentials. The pi-host unit also supplies a `PATH` with mise shims (`pi`) and linuxbrew (`bun`).
- The backend orders startup after `gigachad-bot-pi-host.service` and reaches it
  through `GIGACHAD_PI_HOST_URL=http://127.0.0.1:8301`, set in its unit (and the
  LiteLLM proxy's), not in the shared `$ENV`: development's pi-host holds the
  default `127.0.0.1:8300`.
- pi-host runner: `deploy/run-production-pi-host.sh` → `run-pi-host.sh` with
  `PI_HOST_PORT=8301` from the unit. It loads `~/.secrets` (the gateway's API-key
  providers) and `$ENV`, installs `pi-host/`'s locked dependencies when they are
  missing or older than its manifest or lockfile. It runs whatever `pi` is installed (M13: no
  pin, mise updates Pi). Pi sessions get only pi-host's environment allowlist.
- Both pi-hosts share the token file (`~/.local/state/gigachad-bot/pi-host.token`,
  `PI_HOST_TOKEN_FILE` overrides) and `~/.pi/agent` (logins, sessions): one host
  user, like `$ENV` and `~/.secrets`. That path is the one `src/lib/host_access.py`
  and the LearnHarness sandbox hide. pi-host's session-file lock (M1) keeps one
  writer per session across both: a session live in one answers `409 locked` in the other.
- The pi-host unit has no `NoNewPrivileges`, `PrivateTmp` or
  `UMask`: it runs the user's Pi agents.
- `redeploy.sh` restarts pi-host before the backend, only when `pi-host/` changed
  (code, its manifest, the lockfile), the installed `pi --version` differs from the
  one `/health` reports, or pi-host is down (M14). It never restarts while a Pi
  session runs a turn (`/health`; an older pi-host: a `pi` child in the unit's cgroup) or a
  factory runner is alive (`pgrep -f` on `agent-factory/src/runner.ts` and the old `extensions/factory/runner.ts`), and prints the
  restart command instead (M4). Without any `pi` on `PATH` the redeploy stops before a restart.
  The installer stops, disables and removes the retired `gigachad-bot-omp.service`, and the omp-web units once
  no omp session runs (see "Retire omp-web").
- The backend starts the stateless LikeC4 service, points `C4_SERVICE_URL` at it,
  and stops it on exit. Each start uses `npm ci` against its lockfile. If unavailable,
  architecture routes return 503, other features remain available.

| Service | Development | Production |
| --- | --- | --- |
| FastAPI | `127.0.0.1:8001` | `127.0.0.1:8002` |
| LikeC4 | `127.0.0.1:8011` | `127.0.0.1:8012` |
| PostgreSQL | `gigachad-dev` container/volume | `gigachad-prod` container/volume |
| pi-host (Pi sessions, completion gateway, usage) | `127.0.0.1:8300` (`run.sh`) | `127.0.0.1:8301`, same token file and `~/.pi/agent` |
| LearnHarness / LiteLLM proxy / gate | `127.0.0.1:30181` / `127.0.0.1:30183` / - | Same defaults, one instance per machine, gate `127.0.0.1:30182` (Serve `:8444`) |
| Factory preview gate | `127.0.0.1:30185` | Same default, one instance per machine, behind Serve `:8445` |

Inspect failures rather than exposing another listener:

```bash
journalctl --user -u gigachad-bot.service -n 100 --no-pager
journalctl --user -u gigachad-bot-pi-host.service -n 100 --no-pager
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
- Ollama stays on loopback (setup refuses to bridge a port bound to `0.0.0.0`), a
  persistent alias/socket proxy exposes it to sandbox containers at `10.200.200.1:11434`.
- The bundled `gigachad` profile uses `${HOME}`-relative absolute paths for the real
  OMP binary, Python interpreters, and `~/.omp`. Its OMP path assumes mise's
  `github-can1357-oh-my-pi` install directory: override that mount in a custom profile
  when `mise` names it differently. Do not mount a mise shim or edit installed
  profiles, `uv sync` replaces them. Put custom profiles in a directory named by
  `AGENT_SANDBOX_PROFILE_PATH`, searched before bundled profiles. Doctor must pass.

### Retire omp-web (once per host)

Pi tabs replaced the omp tabs: the omp-web units, their runner and the Tailnet gate are gone from the
repository. On a host that still runs them, after the redeploy that removed them:

1. Reinstall the units. `install-systemd-service.sh` stops, disables and removes
   `gigachad-bot-omp-web-gate.service`, `gigachad-bot-omp-web.service` and `gigachad-bot-omp-web-host.service`
   only when no omp session runs (omp-web's `runningSessionIds`, else the processes the host daemon runs besides
   itself). Otherwise it keeps them and says how many sessions run: rerun it once they are idle.

   ```bash
   "$PROD/deploy/install-systemd-service.sh"
   systemctl --user list-unit-files 'gigachad-bot-omp-web*'   # expect none
   ```

2. Turn off Tailscale Serve on `:8443`, which pointed at the removed gate:

   ```bash
   tailscale serve --https=8443 off
   tailscale serve status   # expect :443, :8444 and :8445 only
   ```

3. Remove the `GIGACHAD_OMP_WEB_*` lines from `$ENV`, nothing reads them.
4. Optional cleanup: `~/.local/share/gigachad-bot/ompweb-upstream*` (served worktree, daemon log and lock) and
   `~/git/ompweb_original`. Old omp sessions stay in `~/.omp/agent/sessions`, reachable through the `omp` CLI.

Development: `run.sh` no longer starts omp-web. An instance an earlier `run.sh` started on `127.0.0.1:30177` ends
with that run, its `ompweb host` daemon loop (started with `setsid`) keeps running until stopped by hand once its
omp sessions are idle: `pkill -f 'omp-web.js host'`.

### Provision learn mode (LearnHarness)

Units, environment, gate, backup and restore: `.docs/integration-learnharness.md`, "Production deployment".

### Provision the agent factory

Factory tasks belong to the agent-factory package of the service user's Pi: each runs in a detached runner
(`bun runner.ts`, under systemd in its own `systemd-run --user --scope`, so a pi-host restart leaves it running), its
files under `~/.pi/agent/factory/`. pi-host only calls the package's core. The only factory unit is the preview gate.

The factory needs two things this repo does not ship. Both must be current for the service user on every host that
runs factory tasks:

| Dependency | Source | Installed as | Without it |
| --- | --- | --- | --- |
| `agent-factory` (core, `factory` CLI, Pi extension) | [patrickab/agent-factory](https://github.com/patrickab/agent-factory) | a clone registered with `pi install <clone>`; its `factory` CLI named in the environment file (below) | No task starts, the board reports `factory_plugin_missing` with this link |
| `agent-worktree` | [`.os`: `chezmoi/dot_local/bin/executable_agent-worktree`](https://github.com/patrickab/.os/blob/master/chezmoi/dot_local/bin/executable_agent-worktree) | `~/.local/bin/agent-worktree` (`chezmoi apply`, or copy the script to a directory on the service's `PATH`) | No worktree task starts (tasks in the checkout still do) |

The pi-host unit finds the CLI through one line in `~/.config/gigachad-bot/env`:
`AGENT_FACTORY_CLI=<absolute path of the factory binary>` (a clone's binary is `<clone>/src/cli.ts`).
`run-production-pi-host.sh` loads that file through `deploy/load-env.sh` before it starts pi-host; a variable the unit
already sets wins over the file. pi-host reads it at start, so restart `gigachad-bot-pi-host.service` after the change.
Without the line pi-host looks for `factory` on the unit's `PATH` (set in `gigachad-bot-pi-host.service`, not the login
shell's), which has no `~/.bun/bin`, where `bun install -g` and `bun link` put it. The binary's `#!/usr/bin/env bun`
finds `bun` on the unit's `PATH` through mise's shims or linuxbrew, as pi-host itself does.

pi-host runs `factory --core-path` once and imports the file it prints. A package update reaches a task when its
runner next starts; pi-host imports `core.ts` again when its mtime changes, a change only in another module of the
package reaches pi-host when it restarts. Runners the old plugin location (`~/.pi/agent/extensions/factory/`) started
keep running, and `redeploy.sh` counts both.

| Concern | Contract |
| --- | --- |
| Previews | `gigachad-bot-preview-gate.service` runs `run-preview-gate.sh --prod` on loopback `:30185`. Serve `:8445` points at it, `GIGACHAD_PREVIEW_URL` names that address for the backend. Preview processes stay on loopback ports 8100–8199. |
| Preview database | Each preview gets a throwaway database on the server of `GIGACHAD_DATABASE_URL`, so the backend's PostgreSQL role needs `CREATEDB`. The role `scripts/setup-postgres.sh` creates (the image's `POSTGRES_USER`) is a superuser and has it. |
| Worktrees | Task worktrees come from `agent-worktree`, which must be on the backend's `PATH`. |

- Configure once per production host, before service startup:

```bash
"$PROD/deploy/install-systemd-service.sh"
systemctl --user start gigachad-bot-preview-gate.service
"$PROD/deploy/tailscale-serve.sh" 30185 8445
```

- Add to `$ENV` and restart the backend:

```dotenv
GIGACHAD_PREVIEW_URL=https://<tailnet-hostname>:8445
```

- Check: `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30185/` prints `403` (no Tailnet login).
- A host that ran the Agents-in-a-box factory engine: `install-systemd-service.sh` stops, disables and removes
  `gigachad-bot-factory.service`. Remove `GIGACHAD_FACTORY_URL` from `$ENV`, nothing reads it. Optional cleanup: the
  `factory` database in the production PostgreSQL and `~/.config/agent-sandbox/factory.yaml`.
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
- The frame gates get their own HTTPS ports with the same helper: LearnHarness
  `30182 8444`, factory previews `30185 8445` (see their sections). Serve accepts any
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
- Check `/api/config`: `learnharness_url` is set for granted logins, `null` otherwise.

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
3. Provision sandbox/pi-host/factory/learn mode services, then pass sandbox doctor.
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
5. Confirm `sudo ss -ltnp '( sport = :8002 or sport = :8301 )'` shows loopback only and
   `tailscale serve status` shows private ingress. Never test with Funnel.
6. Check pi-host with its token, it answers the installed Pi version (no `Origin` header, which it refuses):

   ```bash
   curl --fail --show-error -H "Authorization: Bearer $(cat ~/.local/state/gigachad-bot/pi-host.token)" \
     http://127.0.0.1:8301/health
   ```

   Then open the Pi page, start a disposable session, send one prompt and delete it.

## 6. Roll back safely

Keep the private hostname and loopback port stable:

1. `systemctl --user stop gigachad-bot.service`. Stop `gigachad-bot-pi-host.service`
   too when the rollback crosses a `pi-host/` change, after its Pi sessions are idle:
   stopping ends their turns, the sessions themselves persist.
2. Restore a known-good code revision in `$PROD` and its matching locked
   dependencies. Restore configuration only if changed, restore database or
   mirror backups only when reverting data is necessary. Do not overwrite newer
   user data merely to roll back code. Applied `db/schema/` steps stay applied:
   older code ignores additive ones such as `0011` (document `meta` and
   `sha256` columns, `STORAGE EXTERNAL`), and chat metadata it leaves stale is
   rederived from the chat body once newer code runs again.
3. Start the service and repeat loopback/Tailnet health checks:

   ```bash
   systemctl --user start gigachad-bot-pi-host.service gigachad-bot.service
   curl --fail --show-error http://127.0.0.1:8002/healthz
   ```

4. Restore the matching frontend deployment, preserving its private API base and
   exact CORS origin. Restart the backend if its origin configuration changes.
5. To withdraw ingress immediately, use `tailscale serve off`. Restore it with
   the section 2 helper, never by exposing the backend directly.
6. pi-host refuses a `pi` other than its revision's pin (`pi-host/package.json`):
   install that version (mise) before starting it. Its dependencies reinstall on
   start from the restored lockfile. A revision from before pi-host runs the OMP
   model source instead: reinstall that revision's units with its
   `deploy/install-systemd-service.sh` and disable `gigachad-bot-pi-host.service`.
7. A revision from before the omp-web removal serves the omp tabs through omp-web again: reinstall that revision's
   units with its `deploy/install-systemd-service.sh`, start `gigachad-bot-omp-web.service`,
   `gigachad-bot-omp-web-host.service` and `gigachad-bot-omp-web-gate.service`, and restore Serve with its
   `deploy/tailscale-serve.sh 30179 8443`.

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
