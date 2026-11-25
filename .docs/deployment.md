# Private deployment: Vercel + Tailscale Serve

## Deployment model

- Prefer one private HTTPS origin serving the UI and `/api` (see [README](../README.md)). This runbook covers existing split-origin Vercel + private Tailscale deployments.

| Boundary | Contract |
| --- | --- |
| Vercel | Serves the frontend bundle, not an API proxy |
| Browser | Calls the private API directly. Requires Tailnet enrollment and ACL permission |
| Backend | Binds loopback. Never Funnel, never a public fallback |
| Production isolation | Separate code worktree and PostgreSQL container/volume |
| Shared state | Development and production share the default Documents/PDF mirror, not PostgreSQL data. Code rollback does not roll back state |

## Values to choose

| Placeholder | Meaning |
| --- | --- |
| `<tailnet-hostname>` | Host's stable MagicDNS name, e.g. `gigachad-backend.example.ts.net` |
| `<vercel-production-origin>` | Exact HTTPS production origin, no path or trailing slash |
| `<vercel-preview-origin>` | Exact preview origin, authorized only while testing |

- Split-origin API base: `https://<tailnet-hostname>/api`.
- Paths, in the deploying user's shell:

```bash
REPO="$HOME/git/gigachad-bot"
PROD="$HOME/git/gigachad-bot-prod"
ENV="${XDG_CONFIG_HOME:-"$HOME/.config"}/gigachad-bot/env"
```

## 1. Prepare the private host

### Install and join Tailscale (Arch / Omarchy)

Use the Arch package; the universal installer may not recognize Omarchy.

1. Install and join:

   ```bash
   sudo pacman -Syu tailscale
   sudo systemctl enable --now tailscaled
   sudo tailscale up --hostname=gigachad-backend
   sudo tailscale status
   ```

2. Complete login. Permit intended devices through Tailnet ACLs.
3. Record the MagicDNS hostname. Enable HTTPS certificates when prompted.

### Create the production worktree

1. Keep development at `$REPO`. Create production from `origin/master`:

   ```bash
   git -C "$REPO" fetch origin master
   git -C "$REPO" worktree add --track -b production "$PROD" origin/master
   ```

2. Fetch over HTTPS, push over SSH (the settings menu's redeploy runs without an SSH agent):

   ```bash
   git -C "$REPO" remote set-url origin https://github.com/patrickab/gigachad-bot.git
   git -C "$REPO" remote set-url --push origin git@github.com:patrickab/gigachad-bot.git
   ```

3. For each reviewed update:

   ```bash
   git -C "$PROD" fetch origin master
   git -C "$PROD" merge --ff-only origin/master
   cd "$PROD" && uv sync --python 3.12 --locked --no-dev
   ```

- **Never use `run.sh` in production**: it installs dev dependencies, starts reload mode and a development frontend.
- Services do not resync Python on restart.
- Production sync includes local OCR. Leave `MINERU_SERVER_URL` unset unless you intend remote OCR.
- **Never set `GIGACHAD_BASE_DIR`** in production artifacts: keep the shared default Documents mirror.

### Dependency supply chain

- Install from `uv.lock`, `src/frontend/package-lock.json` and `src/c4/package-lock.json` with `uv sync --locked` and `npm ci`.
- Development launchers repair missing/outdated locks with `uv lock` (without `--upgrade`) or `npm install --package-lock-only`, then retry installation once. Review and commit regenerated locks with their manifests.
- Production never regenerates locks; any other installation failure stops startup.
- `run.sh` syncs Python once and exports `GIGACHAD_DEPS_SYNCED=1`, so `run-backend.sh`/`run-frontend.sh` skip their own sync (it would race on the shared venv).
- Default groups (`dev`, `ocr`, `sandbox-plot`) come with every `uv sync` and `uv run`; production's `--no-dev` drops only `dev`, so `polars` stays.
- `pyproject.toml` requires uv 0.13+ (`required-version`): an older uv refuses rather than rewriting `uv.lock`.
- Development npm installs (`src/backend/install_deps.py`) run `npm ci` only when `package.json`, the lock or `.npmrc` changed since the last successful install (stamp: `node_modules/.gigachad-npm-inputs.sha256`). Production runs `npm ci` on every start.
- `run.sh` installs the frontend once and exports `GIGACHAD_FRONTEND_DEPS_INSTALLED=1`, so `run-frontend.sh` skips it. A running install breaks the chat-ui pack (`tsc` from `src/frontend/node_modules`) and restarts the backend's reloader. Started separately: let `run-frontend.sh` finish its install before starting the sidecars.
- Keep the seven-day release cooldown (`exclude-newer`, `.npmrc`'s `min-release-age`) and disabled npm install scripts (`ignore-scripts`). Review required scripts before running them explicitly.
- Cooldown and automated lock resolution do not replace security review. Review Dependabot diffs for all three locks.

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

- `$ENV` format: literal, untrimmed `NAME=value` records. `=` allowed inside values; whitespace-only lines and `#` comments allowed.
- **No** `export`, quotes, expansion or command substitution in `$ENV`.
- Loader semantics: no evaluation, malformed records rejected, inherited variables win.
- `~/.secrets` is sourced first; its values win without entering the systemd manager.

Example (substitute your production origin):

```dotenv
GIGACHAD_CORS_ORIGINS=<vercel-production-origin>,http://127.0.0.1:2999,http://localhost:2999
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8001/api
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

| Variable | Rule |
| --- | --- |
| `GIGACHAD_CORS_ORIGINS` | Required, exact origins. Production rejects `*` and empty lists. Exact origins enable credentialed CORS: a startup read timing out (15 s) switches later API calls to credentialed requests on a separate Chrome connection pool, recovering from a dead HTTP/2 connection Tailscale Serve dropped (tailscaled logs `packet was not handled`) |
| `NEXT_PUBLIC_API_BASE` | Public, build-time. The example is for local development; configure Vercel's private HTTPS value separately |
| `OLLAMA_BASE_URL` | Only for Ollama |
| `MINERU_SERVER_URL` | Optional remote OCR |

- URLs and model names are configuration, not credentials. Brave requests run server-side: its API key must never reach the browser.

### Provision the systemd services

1. Install the units (backend, pi-host, others) without starting them; enable lingering so services survive logout and start at boot:

   ```bash
   "$PROD/deploy/install-systemd-service.sh"
   loginctl enable-linger "$USER"
   ```

   The installer must validate the production runners and the venv's Uvicorn.

2. Only once configuration is ready, start and check:

   ```bash
   systemctl --user start gigachad-bot-pi-host.service gigachad-bot.service
   systemctl --user status gigachad-bot.service gigachad-bot-pi-host.service
   curl --fail --show-error http://127.0.0.1:8002/healthz
   ```

`gigachad-bot.service`:

- Runs `run-backend.sh --prod`: loads credentials/config, starts or reuses production PostgreSQL, serves loopback Uvicorn without reload.
- Keep `NoNewPrivileges`, `PrivateTmp`, `UMask=0077`, restart handling and stop timeout.
- **Never bind FastAPI to a LAN, Tailnet or wildcard address.**
- Starts after `gigachad-bot-pi-host.service`, reaching it via `GIGACHAD_PI_HOST_URL=http://127.0.0.1:8301`, set in its unit (and the LiteLLM proxy's), **not** in `$ENV`: development's pi-host holds the default `127.0.0.1:8300`.
- Starts the stateless LikeC4 service, points `C4_SERVICE_URL` at it, stops it on exit; `npm ci` against its lockfile on each start. If unavailable, architecture routes return 503, everything else works.

Units use `%h` and `GIGACHAD_ENV_FILE`, not `EnvironmentFile` (systemd must not parse credentials).

`gigachad-bot-pi-host.service`:

- Runner: `deploy/run-production-pi-host.sh` → `run-pi-host.sh`, `PI_HOST_PORT=8301` from the unit.
- Unit `PATH` includes mise shims (`pi`) and linuxbrew (`bun`).
- Loads `~/.secrets` (gateway API-key providers) and `$ENV`; installs `pi-host/`'s locked dependencies when missing or older than its manifest or lockfile.
- Runs whatever `pi` is installed (M13: no pin, mise updates Pi). Pi sessions get only pi-host's environment allowlist.
- No `NoNewPrivileges`, `PrivateTmp` or `UMask`: it runs the user's Pi agents.
- Development and production pi-hosts share the token file `~/.local/state/gigachad-bot/pi-host.token` (`PI_HOST_TOKEN_FILE` overrides) and `~/.pi/agent` (logins, sessions), like `$ENV` and `~/.secrets`. `src/lib/host_access.py` and the LearnHarness sandbox hide that path.
- The session-file lock (M1) allows one writer per session across both: a session live in one answers `409 locked` in the other.

`redeploy.sh` and pi-host (M4, M14):

- Restarts pi-host before the backend, only when `pi-host/` changed (code, manifest, lockfile), the installed `pi --version` differs from `/health`'s, or pi-host is down.
- Never restarts while a Pi session runs a turn (`/health`; older pi-host: a `pi` child in the unit's cgroup) or a factory runner is alive (`pgrep -f` on `agent-factory/src/runner.ts` and the old `extensions/factory/runner.ts`); prints the restart command instead.
- Without any `pi` on `PATH`, stops before a restart.
- The installer stops, disables and removes the retired `gigachad-bot-omp.service`, and the omp-web units once no omp session runs (see "Retire omp-web").

| Service | Development | Production |
| --- | --- | --- |
| FastAPI | `127.0.0.1:8001` | `127.0.0.1:8002` |
| LikeC4 | `127.0.0.1:8011` | `127.0.0.1:8012` |
| PostgreSQL | `gigachad-dev` container/volume | `gigachad-prod` container/volume |
| pi-host (Pi sessions, completion gateway, usage) | `127.0.0.1:8300` (`run.sh`) | `127.0.0.1:8301`, same token file and `~/.pi/agent` |
| LearnHarness / LiteLLM proxy / gate | `127.0.0.1:30181` / `127.0.0.1:30183` / - | Same defaults, one instance per machine, gate `127.0.0.1:30182` (Serve `:8444`) |
| Factory preview gate | `127.0.0.1:30185` | Same default, one instance per machine, behind Serve `:8445` |

On failure, read the logs; do not expose another listener:

```bash
journalctl --user -u gigachad-bot.service -n 100 --no-pager
journalctl --user -u gigachad-bot-pi-host.service -n 100 --no-pager
```

### Provision the code sandbox

Workspace agents and image-based plot requests run through OMP in [Agents-in-a-box](https://github.com/patrickab/Agents-in-a-box) (rootless Docker, gVisor).

```bash
git clone https://github.com/patrickab/Agents-in-a-box.git "$HOME/git/Agents-in-a-box"
cd "$HOME/git/Agents-in-a-box"
./scripts/setup_agent_sandbox.sh
cd "$PROD" && uv run agent-sandbox doctor --profile gigachad
```

- Setup installs Docker/runsc, builds the image, configures host services. Existing installations need only `./scripts/setup_host_services.sh`.
- Ollama stays on loopback (setup refuses to bridge a port bound to `0.0.0.0`); a persistent alias/socket proxy exposes it to sandbox containers at `10.200.200.1:11434`.
- The bundled `gigachad` profile mounts `${HOME}`-relative absolute paths for the real OMP binary, Python interpreters and `~/.omp`. Its OMP path assumes mise's `github-can1357-oh-my-pi` install directory: if `mise` names it differently, override that mount in a custom profile.
- **Do not** mount a mise shim or edit installed profiles (`uv sync` replaces them). Put custom profiles in the directory named by `AGENT_SANDBOX_PROFILE_PATH` (searched before bundled ones).
- Doctor must pass.

### Retire omp-web (once per host)

On a host still running omp-web, after the redeploy that removed it:

1. Reinstall the units. `install-systemd-service.sh` stops, disables and removes `gigachad-bot-omp-web-gate.service`, `gigachad-bot-omp-web.service` and `gigachad-bot-omp-web-host.service` only when no omp session runs (omp-web's `runningSessionIds`, else the processes the host daemon runs besides itself). Otherwise it keeps them and reports how many sessions run: rerun once they are idle.

   ```bash
   "$PROD/deploy/install-systemd-service.sh"
   systemctl --user list-unit-files 'gigachad-bot-omp-web*'   # expect none
   ```

2. Turn off Tailscale Serve on `:8443` (pointed at the removed gate):

   ```bash
   tailscale serve --https=8443 off
   tailscale serve status   # expect :443, :8444 and :8445 only
   ```

3. Remove the `GIGACHAD_OMP_WEB_*` lines from `$ENV`.
4. Optional cleanup: `~/.local/share/gigachad-bot/ompweb-upstream*` (served worktree, daemon log, lock) and `~/git/ompweb_original`. Old omp sessions stay in `~/.omp/agent/sessions`, reachable through the `omp` CLI.

Development: `run.sh` no longer starts omp-web. An instance an earlier `run.sh` started on `127.0.0.1:30177` ends with that run; its `ompweb host` daemon loop (started with `setsid`) runs until you stop it, once its omp sessions are idle: `pkill -f 'omp-web.js host'`.

### Provision learn mode (LearnHarness)

Units, environment, gate, backup and restore: `.docs/integration-learnharness.md`, "Production".

### Provision the agent factory

- Tasks run in the service user's Pi agent-factory package: each in a detached runner (`bun runner.ts`, in its own `systemd-run --user --scope`, so a pi-host restart leaves it running), files under `~/.pi/agent/factory/`. pi-host only calls the package's core.
- The only factory unit is the preview gate.

Required on every host that runs factory tasks, current for the service user:

| Dependency | Source | Installed as | Without it |
| --- | --- | --- | --- |
| `agent-factory` (core, `factory` CLI, Pi extension) | [patrickab/agent-factory](https://github.com/patrickab/agent-factory) | A clone registered with `pi install <clone>`; its `factory` CLI named in `$ENV` (below) | No task starts; the board reports `factory_plugin_missing` with this link |
| `agent-worktree` | [`.os`: `chezmoi/dot_local/bin/executable_agent-worktree`](https://github.com/patrickab/.os/blob/master/chezmoi/dot_local/bin/executable_agent-worktree) | `~/.local/bin/agent-worktree` (`chezmoi apply`, or copy the script to a directory on the service's `PATH`) | No worktree task starts (tasks in the checkout still do) |

Factory CLI:

- Add to `~/.config/gigachad-bot/env`: `AGENT_FACTORY_CLI=<absolute path of the factory binary>` (a clone's binary is `<clone>/src/cli.ts`). Then restart `gigachad-bot-pi-host.service` (read at start).
- `run-production-pi-host.sh` loads that file through `deploy/load-env.sh`; a variable the unit already sets wins.
- Without the line, pi-host looks for `factory` on the unit's `PATH` (set in `gigachad-bot-pi-host.service`, not the login shell's), which lacks `~/.bun/bin`, where `bun install -g` and `bun link` put it.
- The binary's `#!/usr/bin/env bun` finds `bun` on the unit's `PATH` through mise's shims or linuxbrew.
- pi-host runs `factory --core-path` once and imports the printed file. Updates: a task gets a package update when its runner next starts; pi-host reimports `core.ts` when its mtime changes; changes only in another package module need a pi-host restart.
- Runners started from the old plugin location (`~/.pi/agent/extensions/factory/`) keep running; `redeploy.sh` counts both.

| Concern | Contract |
| --- | --- |
| Previews | `gigachad-bot-preview-gate.service` runs `run-preview-gate.sh --prod` on loopback `:30185`. Serve `:8445` points at it; `GIGACHAD_PREVIEW_URL` names that address for the backend. Preview processes stay on loopback ports 8100–8199 |
| Preview database | Each preview gets a throwaway database on the server of `GIGACHAD_DATABASE_URL`: the backend's PostgreSQL role needs `CREATEDB`. The role `scripts/setup-postgres.sh` creates (the image's `POSTGRES_USER`) is a superuser and has it |
| Worktrees | `agent-worktree` must be on the backend's `PATH` |

Once per production host, before service startup:

1. Install units, start the gate, publish it:

   ```bash
   "$PROD/deploy/install-systemd-service.sh"
   systemctl --user start gigachad-bot-preview-gate.service
   "$PROD/deploy/tailscale-serve.sh" 30185 8445
   ```

2. Add to `$ENV`, then restart the backend:

   ```dotenv
   GIGACHAD_PREVIEW_URL=https://<tailnet-hostname>:8445
   ```

3. Check: `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30185/` prints `403` (no Tailnet login).

- Host that ran the Agents-in-a-box factory engine: `install-systemd-service.sh` stops, disables and removes `gigachad-bot-factory.service`. Remove `GIGACHAD_FACTORY_URL` from `$ENV`. Optional cleanup: the `factory` database in production PostgreSQL and `~/.config/agent-sandbox/factory.yaml`.
- Repository contracts: [Agent factory](system-overview.md#agent-factory).

## 2. Publish the private HTTPS ingress

1. Inspect existing Serve configuration; this app must own the host's Serve root. Publish with the helper (preserves `/api` and `/healthz`):

   ```bash
   tailscale serve status
   "$PROD/deploy/tailscale-serve.sh" 8002
   tailscale serve status
   ```

   Expected result: `tailscale serve --bg --https=443 --accept-app-caps=gigachad.local/cap/host http://127.0.0.1:8002`.

2. Verify from another permitted Tailnet device. **Never add a public fallback.**

   ```bash
   curl --fail --show-error https://<tailnet-hostname>/healthz
   curl --fail --show-error https://<tailnet-hostname>/api/config
   ```

- Frame gates get their own HTTPS ports with the same helper: LearnHarness `30182 8444`, factory previews `30185 8445`. Serve accepts any HTTPS port; only Funnel is limited to 443, 8443 and 10000.

### Host access

| Identity | Authority |
| --- | --- |
| Host-capable login | Host-user execution through OMP, Neovim, factory, and files outside mounts, excluding credential stores. Learn mode (LearnHarness) |
| Other login | Own mounted roots only |
| Trusted ingress | Tailscale v1.92+ accepts `gigachad.local/cap/host` and forwards `Tailscale-App-Capabilities`, stripping client-supplied copies |
| Backend / gate | Trust only private ingress to loopback (`src/backend/identity.py`) |

1. Grant host access in Tailnet policy, not `$ENV`. Add to Access controls without removing existing rules:

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

2. Check `/api/config`: `learnharness_url` is set for granted logins, `null` otherwise.

- Missing grants or unsupported Serve flags fail closed.

## 3. Configure the Vercel frontend

| Setting | Value |
| --- | --- |
| Framework | Next.js |
| Root Directory | `src/frontend` |
| Install / build | `npm ci` / `npm run build` |
| Output Directory | Vercel default, not a static-export directory |
| Production/Preview environment | `NEXT_PUBLIC_API_BASE=https://<tailnet-hostname>/api` |

- Redeploy after changing the API base (baked into the bundle).
- Tailnet access goes to browsers, not Vercel: browsers make the API requests.
- Keep exact production/local origins in `$ENV`'s `GIGACHAD_CORS_ORIGINS`.
- Restart after backend configuration changes:

```bash
systemctl --user restart gigachad-bot.service
```

### Preview deployments and CORS

1. Append the exact preview origin and restart the backend.
2. Test from an enrolled browser.
3. Remove the origin and restart when finished.

- **No wildcard shortcuts**, including `*`, in production.
- Keep Tailnet ACLs: CORS complements identity, never replaces it.

## 4. Deployment sequence

1. Update the production worktree and install locked dependencies.
2. Confirm nonsecret `$ENV` and separate `~/.secrets`. `GIGACHAD_BASE_DIR` unset; `MINERU_SERVER_URL` unset when using local OCR.
3. Provision sandbox/pi-host/factory/learn mode services; pass sandbox doctor.
4. Start or restart the backend; verify loopback health.
5. Publish private Serve ingress; verify from another permitted device.
6. Deploy the frontend with the matching API base and authorized origin.
7. Run the verification below before use.

- Back up PostgreSQL and the shared Documents/PDF mirror before state-affecting updates. LearnHarness data has its own daily backup (`.docs/integration-learnharness.md`, "Backup and restore"); the launcher snapshots it before a ref bump.
- Avoid destructive backup synchronization.
- Code worktrees are not application-data backups.

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

   Require `access-control-allow-origin: <vercel-production-origin>`. Local `:2999` origins are intentional; preview origins only while authorized.
3. Send a disposable chat; verify `/api/chat` SSE streams and completes (health checks do not prove streaming).
4. Refresh and reopen that chat (persistence), then delete it.
5. Confirm `sudo ss -ltnp '( sport = :8002 or sport = :8301 )'` shows loopback only and `tailscale serve status` shows private ingress. **Never test with Funnel.**
6. Check pi-host with its token (no `Origin` header: it refuses one); it answers the installed Pi version:

   ```bash
   curl --fail --show-error -H "Authorization: Bearer $(cat ~/.local/state/gigachad-bot/pi-host.token)" \
     http://127.0.0.1:8301/health
   ```

   Then open the Pi page, start a disposable session, send one prompt, delete it.

## 6. Roll back safely

Keep the private hostname and loopback port stable.

1. `systemctl --user stop gigachad-bot.service`. If the rollback crosses a `pi-host/` change, also stop `gigachad-bot-pi-host.service` once its Pi sessions are idle (stopping ends their turns; sessions persist).
2. Restore a known-good code revision in `$PROD` and its matching locked dependencies.
   - Restore configuration only if changed.
   - Restore database or mirror backups only when reverting data is necessary. **Do not overwrite newer user data merely to roll back code.**
   - Applied `db/schema/` steps stay applied: older code ignores additive ones such as `0011` (document `meta` and `sha256` columns, `STORAGE EXTERNAL`); chat metadata it leaves stale is rederived from the chat body once newer code runs again.
3. Start the services and repeat loopback/Tailnet health checks:

   ```bash
   systemctl --user start gigachad-bot-pi-host.service gigachad-bot.service
   curl --fail --show-error http://127.0.0.1:8002/healthz
   ```

4. Restore the matching frontend deployment, keeping its private API base and exact CORS origin. Restart the backend if its origin configuration changes.
5. To withdraw ingress immediately: `tailscale serve off`. Restore it with the section 2 helper, **never by exposing the backend directly**.
6. pi-host refuses a `pi` other than its revision's pin (`pi-host/package.json`): install that version (mise) before starting it. Its dependencies reinstall on start from the restored lockfile. A revision from before pi-host runs the OMP model source instead: reinstall that revision's units with its `deploy/install-systemd-service.sh` and disable `gigachad-bot-pi-host.service`.
7. A revision from before the omp-web removal serves omp tabs through omp-web: reinstall that revision's units with its `deploy/install-systemd-service.sh`, start `gigachad-bot-omp-web.service`, `gigachad-bot-omp-web-host.service` and `gigachad-bot-omp-web-gate.service`, and restore Serve with its `deploy/tailscale-serve.sh 30179 8443`.

Schema and sync after a code rollback:

- Step 0013 only deleted superseded `changes` rows (replay output identical) and added an index; older code just stops pruning.
- The sync stream tolerates frontend and backend from different revisions either way; an older backend replays from `since=0`.

LearnHarness rollback (upstream keeps no backward compatibility for data a newer commit changed):

1. Stop the LearnHarness unit.
2. Move its home aside.
3. Copy the matching `snapshots/<old-sha>-<utc>/` back into a new home (its `data/`, `auth.json` and credential files) and write `<old-sha>` to its `served-commit` — or restore a backup taken under that commit.
4. Start it (`.docs/integration-learnharness.md`, "Backup and restore").

Repeat health, CORS, SSE and persistence verification before resuming use.
