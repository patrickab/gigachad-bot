# LearnHarness integration (learn mode)

Read only when working on learn mode. Why a sidecar and not a fork: [ADR 0001](adr/0001-learnharness-as-unmodified-sidecar.md). Planning history (local, gitignored): `~/git/gigachad-bot/.wayfinder/learnharness-integration/`.

## Goal

- [LearnHarness](https://github.com/amosblomqvist/learnharness) (LH, AGPL-3.0-only, Next.js + pi SDK, JSON/JSONL files) probes what the learner knows, plans a dependency tree of concepts, teaches with checked understanding and keeps a global Knowledge graph. gigachad's tutoring prompt cannot.
- **Receive every LH feature as upstream ships it, with gigachad's UI customized as far as that allows.** No fork, no copy: each start serves the newest release tag passing the overlay checks, a pinned ref holds a version. Every LH function stays reachable, only the UI around it differs.
- Host-only, single learner. Knowledge is global and LH's: gigachad chats never read it, gigachad never feeds it.

## Decisions

| # | Decision | Why |
|---|---|---|
| D1 | LH runs **unmodified** as a sidecar, plus a private overlay copied in as `custom/` and a seam patch of a few lines | A fork turns every upstream change into maintenance |
| D2 | gigachad-bot is AGPL-3.0-only plus a section 7 permission for MinerU (`LICENSE`, `LICENSE-EXCEPTION`, [ADR 0002](adr/0002-agpl-3-0-only.md)) | License-clean overlay and upstream contributions; MinerU >= 3.1 adds a usage threshold AGPL does not allow |
| D3 | **Hide and drive**: gigachad's model selector, sidebar and look drive LH's hidden UI through upstream's own writes, events and dialogs | Maximum customization without reimplementing LH logic |
| D4 | LH's own composer, docks and stats bar stay (restyled); a shell composer was dropped | It coupled to an untyped upstream quirk and cost ~900 lines for little gain |
| D5 | gigachad's model selector stays; LH's model pills hidden but mounted; the session model is frozen (read-only) | User picks from gigachad's catalog; one session is taught by one model |
| D6 | LH's sidebar and title bar are hidden: session title = tab name, learn sessions are a category in gigachad's sidebar, each of their features has a shell path | User decision. Cost: each new upstream feature there needs a shell path; the inventory guard fails until it has one |
| D7 | Prefer opening upstream's own UI inside the frame (popups, panels, dialogs) over re-rendering it; plain data and API actions go through the backend adapter | Upstream keeps rendering and evolving it |
| D8 | Models only from gigachad, through **pi-host's completion gateway** as LH's one custom endpoint; snapshot generated from one login's catalog; no provider key, endpoint or subscription UI in LH | One place for keys and models; any keyed builtin provider would add its models to LH's menus |
| D9 | LH's tiers kept: **strong** (prober, teacher; picked per draft, stamped per session), **cheap** (researcher, demo builder, extractor, auditor), **vision** (visual maker, PDF reader); cheap and vision set in LH Settings → Agent | Background work stays affordable, figures and PDFs need an image model |
| D10 | LH runs in a **sandboxed user unit** (environment allowlist, `InaccessiblePaths` denylist, private HOME); production adds a host-capability gate | Its agent reads any absolute path and has no auth |
| D11 | Projects group learn sessions (`learn_sessions` in `project.json`); deleting a project deletes its sessions with `preserve-graph` | Like chats, without losing Knowledge |
| D12 | "Learn this": chat tool `start_learn_session(goal)`, the chat model condenses the conversation into the goal, the session uses LH's current strong tier | Move from exploring to learning without retyping |

## Architecture

| Part | Where | Role |
|---|---|---|
| Shell | `src/frontend/features/learn/` | Model/thinking selectors in the learn top bar, Plan button, learn sessions in gigachad's sidebar (rows, chips, filter, rename, delete, export, panels, background agents), one persistent frame |
| Protocol | `src/frontend/packages/chat-ui/src/learn-protocol.ts` | postMessage protocol shared by both sides |
| Overlay | `learnharness-overlay/` → `custom/` | `root.tsx` (entry), `framing.ts` (frame-ancestors CSP), `next.config.ts` (`withCustom` module overrides), `embed/` (bridge, channel, drive mode, drivers `model.tsx`, `freeze.ts`, `chrome.tsx`), `overrides/components/` (model-picker, sidebar, top-bar, `ai-elements/` markdown wrapper), `theme/` (embed-only CSS, hides Providers UI), `seams.patch`, `*.test.mjs` |
| Backend adapter | `src/lib/learnharness.py`, `src/backend/routes/learn.py` | `/api/learn` (host only): session list with row states, create + first turn, rename, delete (`preserve-graph` / `delete-nodes`), Markdown export, project mapping |
| Model generator | `src/lib/learnharness_config.py` (`write-config`) | LH's `settings.json` custom provider and tiers |
| Gate | `src/backend/learnharness_gate.py` | Production Tailnet identity gate in front of the frame |
| Launcher | `run-learnharness.sh` | Update/select ref, build, serve LH in its sandbox unit, refuse an unguarded ref |

`~/git/learnharness` is the upstream clone (read only); the served worktree and the LH home (see Configuration) are owned by the launcher, never edited by hand (`custom/` is overwritten every start).

Styling: overlay CSS applies under `html.cf-embed` only (standalone LH keeps its look): gigachad palette, fonts (Outfit, JetBrains Mono), type and radius ladders, markdown via a thin `MessageResponse` wrapper. A learner's LH Appearance colours and prose size override the embed tokens.

## Protocol contract (`learn-protocol.ts`, v3, channel `learn-embed`)

- Shell → frame: `hello`, `set-theme`, `open-session`, `new-session {kind}`, `open-home`, `set-model`, `set-thinking`, `open-settings {section}`, `open-panel {panel, sessionId}`, `close-modal`.
- Frame → shell: `ready`, `view`, `model`, `models`, `model-open`, `modal`, `session-meta`, `session-plan`, `activity`, `chrome-key`.
- `IS_STATE`: messages that are replayed state (the frame replays them on every `hello`). `ANY_VERSION_SHELL_MESSAGE`: messages that work without a version match (theme, navigation). Keep both exhaustive.
- Both sides validate every message and drop malformed ones.
- **Drive mode** (`html.cf-shell-drive`) starts only after a version-matched handshake (`ready`/`hello` carry `LEARN_PROTOCOL_VERSION`). On mismatch upstream's UI stays fully visible and the shell shows "LearnHarness needs a restart", with no shell model selector.
- A change handles the message on both sides (`embed/channel.ts` + driver, `features/learn/bridge.ts`) in one commit; a breaking change bumps `LEARN_PROTOCOL_VERSION`. Restart frame and shell together.

## Shell contracts

- **One frame**: one persistent iframe for all learn tabs, switched by `open-session`/`new-session`/`open-home`; never reloaded, moved in the DOM, or duplicated per tab. Learn entry and "new session" open `/chat/new`.
- **Keyboard**: upstream's shortcuts work in the frame. In drive mode the frame swallows the hidden chrome's shortcuts (⌘B, ⌘⇧F, F2) and posts `chrome-key`; the freeze turns ⌘K into `model-open` (never on a session's read-only pill). ⇧Tab cycles the draft's strong-tier thinking, swallowed on a session. A chord with nothing to do keeps its browser default.
- **Models**: the draft selector offers catalog models LH serves (`set-model`/`set-thinking` → upstream's write, moves the strong tier globally; running sessions keep their stamp). A session's model is a read-only pill; an unavailable stamped model turns it back into the selector (the freeze lets that pick through). Cheap/vision tiers and every other setting: learn header Settings → `open-settings` → upstream's dialog.
- **Modal**: while the frame reports `modal.open`, the shell dims its chrome with a scrim whose hole tracks the frame; scrim click or Escape sends `close-modal`.
- **Panels**: row menu Plan (when `session-plan` says so) → `open-panel plan` → upstream's PlanPopup; Coverage → upstream's `CoverageRingDetail` and `CoverageMenuBlock`; background agents folder (`activity`) → `open-panel activity` → upstream's `ActivitySection`.
- **Rows**: title from `session-meta`; state chip from `session-meta.status`, else the list fields; export → `GET /api/learn/sessions/{id}/markdown`; delete (undo) → preserve graph, or "Delete with graph traces" → `DELETE ?mode=delete-nodes`. Deleting the open session sends the tab to learn home and the frame to `/`.
- **Sidebar polling** (`features/learn/useLearnSessions.ts`): read on start and window focus, and every 15 s only while the page is visible, the last read succeeded, and a learn tab shows the frame or a session is `busy`. A 503 (LH down or disabled) shows no error and stops polling until next focus. Never add an idle-polling timer.
- **Projects**: deleting a project deletes its LH sessions (preserve-graph) and aborts on any failure but 404. The Learn section lists unassigned sessions only; sessions opened from a project join it.
- **"Learn this"** `start_learn_session(goal)`: host-only; creates a session, assigns it to the chat's project, submits the goal.
- **Tool-choice storage** (compatibility, still running): `localStorage` `gigachad-enabled-tools` holds `{enabled, known}`; `known` lists every tool at save time, so a new `defaultEnabled` tool starts enabled once and a later opt-out sticks. A legacy bare array counts as knowing every tool except `ADDED_AFTER_BARE_LISTS`. Never extend that list for new tools.

### Accepted gaps

- gigachad documents and images never reach a learn session; LH takes what its own composer takes. Dictation works in the frame (`allow="microphone"`) but keyless (no Deepgram key in the sandbox).
- LH's keymap rebinding (`lh:keymap`) applies in the frame only.
- Draft model pick is global (see Models).
- Catalog lag: a new gigachad model reaches LH on its next start or redeploy.
- A coverage or activity card stays open until Escape or a click inside the frame. A row's Plan entry appears once the frame has shown that session.

## Rules

- **Every feature kept**: a hidden upstream control gets a shell path, preferably a frame action opening upstream's own UI, else the backend adapter.
- **Seam policy**: never change LH's server API, sessions, teaching logic or storage. `seams.patch` only loads the overlay (`app/layout.tsx`, `next.config.ts`). Each override is evidence of a missing seam: it goes upstream as an issue or PR, posted by the user (drafts: `.wayfinder/learnharness-integration/spec/upstream-proposals.md`).
- **Never vendor LH code** (components, hooks, CSS); write against its API.
- **Adapters own all LH knowledge**, one place per side:
  - Backend: `src/lib/learnharness.py` (sessions API, contract models) and `src/lib/learnharness_config.py`. Routes map their errors, never build LH URLs. The gate's only LH knowledge is the refused credential routes.
  - Frontend: `src/frontend/features/learn/` only; eslint lets only it import `learn-protocol.ts`, other features import its public files (`crossFeature`).
  - The browser never calls LH's API; the backend does, server to server.
- **Models only from gigachad**: no provider key, custom endpoint or subscription for LH. Environment allowlist, gate refuses those writes, generator owns `customProvider` and first tier seeding.
- **Data**: upstream keeps no backward compatibility for stored data; never bypass the launcher's pre-bump snapshot by editing the served worktree.
- **Commits**: one concern each; the body names the LH contract it relies on.

### Override rules (upgrade safety)

An override is `learnharness-overlay/overrides/<path>`, replacing upstream `<path>` for every importer (`withCustom`). Each rule is enforced by a test or `tsc`:

- **Wrap, never copy**: `export *` from the upstream module by relative path (`../../../components/X`), shadow only what changes, typed against upstream (`ComponentProps<typeof Upstream>`). Never hand-list re-exports.
- **Hidden, still mounted**: hidden upstream UI stays rendered (`hidden`, `inert`, `display: none`), its logic keeps running.
- **Hide only by our markers**: `data-cf-*` or under `html.cf-shell-drive`, never upstream class names. Surfaces are marked by content (sessions sidebar = `<aside>` holding `data-testid="new-session"`, title bar = TopBar holding `phase-chip`), so if upstream drops the content the surface stays visible. A surface not overridden stays visible. Exception: `theme/settings.css` hides the Providers UI by upstream test ids (the marker test catches renames).
- **Exhaustive tables over upstream unions**: every `KeyAction` is classified in `inventory.test.mjs` `KEY_ACTIONS`; `TopBar` props are mapped there.
- **Server/client**: an override imported by a server component (today only `components/sidebar`, via `app/layout.tsx`) has no `"use client"` and no JSX, only `export *` plus re-exports of client wrappers from a client module under `embed/` (`embed/chrome.tsx`). Otherwise star exports are undefined on the server or the wrapper renders twice. Other overrides are normal `"use client"` modules.
- **Aliases**: `overrideAliases` registers `@/X`, `@/X.tsx`, `@/X.ts` and bare sibling `./X` only, never `./X.tsx`/`./X.ts` (Turbopack lets them capture `./embed/X` and `@/custom/embed/X`).

## Upgrade guards

| Guard | Catches |
|---|---|
| `tsc --noEmit` | renamed/removed export, changed prop type |
| `inventory.test.mjs` check 2 | override without `export *` or missing an upstream export; upstream import bypassing the aliases; bare alias key capturing another module |
| check 3 | new/removed keyboard action (`KEY_ACTIONS`); hidden chrome keys out of step with `LEARN_CHROME_KEYS` |
| check 5 | new/removed `lh:` event, shortcut or storage name; an `lh:` name the overlay uses that upstream dropped |
| check 6 (`look.test.mjs`) | type or radius rung without gigachad's value |
| check 7 | upstream API path the overlay calls that no route serves |
| props, hidden chrome | new `TopBar` prop; new control (test id, aria label, Hint, component, unnamed button/input/handler) in the hidden sidebar or title bar without a shell path |
| markers, exports | content marker, test id, data attribute or imported upstream export gone |
| `chrome`/`model`/`freeze`/`embed.test.mjs` | each driver against upstream's real modules |
| launcher `upstream_markers` | served ref lacks a name or URL the overlay or backend adapter relies on |

All but the launcher markers run in a check checkout. The launcher checks every `file|text` in `upstream_markers` (overlay exports, content markers, URLs, the adapter's routes and delete modes) plus the `lh:kbd-*` names `embed/freeze.ts` swallows, after copying the overlay and before install or build, on every start; a miss refuses the ref ("the learn-mode overlay needs updating for this ref").

A failing guard means a feature moved: give it a shell path, a frame action, or leave it visible upstream. Never delete an entry to make a guard pass; update the marker list together with the code relying on it.

## Workflows

### Checks for a gigachad-bot change

- Shell (`src/frontend`): `npx tsc --noEmit`, `npx eslint .`, `npx vitest run features/learn features/navigation`.
- Backend: `.venv/bin/python3 -m pytest -q tests/test_learnharness_sessions.py tests/test_learnharness_models.py tests/test_learnharness_gate.py tests/test_learnharness_launcher.py`.
- Overlay: in a private check checkout, never by rsync into the served worktree.

### Private overlay check checkout

```bash
task=<task>; G=~/git/gigachad-bot; C=~/.local/share/gigachad-bot/learnharness-check-$task; ref=v0.4.0
git -C ~/git/learnharness worktree add --detach "$C" "$ref"                      # once per ref
cd "$C" && ELECTRON_SKIP_BINARY_DOWNLOAD=1 npm ci && git apply "$G/learnharness-overlay/seams.patch"   # once
# every check:
rsync -a --delete "$G/learnharness-overlay/" "$C/custom/"
T=$(cd "$G/src/frontend" && node packages/chat-ui/pack.mjs --out "$C/node_modules/.gigachad-chat-ui") \
  && npm install --no-save --no-audit --no-fund --ignore-scripts "$T"
node_modules/.bin/tsc --noEmit && node --test custom/*.test.mjs
GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS=http://127.0.0.1:<shell port> NODE_ENV=production npm run build
# done: git -C ~/git/learnharness worktree remove --force "$C"
```

`npm run build` catches what `tsc` does not (Next config, CSS imports, Turbopack aliases, server-component prerender). Frame ancestors are baked into the build: another shell origin needs its own build.

### Concurrency

- LH on `30181` (served worktree, default home) is one instance per machine; development reuses production's. Agents never stop or rebuild it, never send turns in existing sessions (live tests only in a throwaway session, deleted afterwards).
- An agent's own LH: `./run-learnharness.sh` supervised with `GIGACHAD_LEARNHARNESS_{BIND,DIR,HOME,FRAME_ANCESTORS}` pointing at a spare port and `~/.cache/learn-mode/<task>/` (delete it afterwards, 2-4 GB), or `next start` from a check checkout with scratch `LH_DATA_DIR`/`LH_AUTH_FILE`. **Never point a scratch LH or generator at the default home**: `write-config` rewrites its `settings.json` and `auth.json`. Never run a scratch backend's model routes against the user's database.
- Never `pkill`/`pgrep -f` by script name (matches other agents' launchers). Name agent worktrees of `~/git/learnharness` in the report; remove only with the user's approval.
- Serialize edits to `learn-protocol.ts`, `seams.patch`, `run-learnharness.sh`, `next.config.ts`, `root.tsx`, `theme/index.ts`, `inventory.test.mjs`. One driver/override/CSS/test set, shell files, the adapter's HTTP section and the generator are parallel-safe.

### Restart and smoke

1. No session busy: `curl -s http://127.0.0.1:30181/api/chats` has no `busy: true`.
2. Development: restarting `run.sh` restarts the LH it started; a reused instance needs its launcher stopped first (Ctrl-C, or `systemctl --user stop gigachad-learnharness-30181`). Production: `systemctl --user restart gigachad-bot-learnharness.service` or `redeploy.sh`.
3. A change rebuilds (~2 min). `.next/gigachad-build` is the only proof of a complete build.
4. Smoke at `http://localhost:2999` with a throwaway session: Learn opens `/chat/new` with upstream composer and stats bar, no model pills, no LH sidebar or title bar; send, Stop, switch learn tabs (no reload), theme, row panels, rename, start from a project (lands in its learn folder), delete.

## Ref selection and updates

### Automatic (default, `GIGACHAD_LEARNHARNESS_REF` unset)

Every real start of `run-learnharness.sh` (`run.sh`; production `gigachad-bot-learnharness.service` via `deploy/run-production-learnharness.sh server`) updates first. An LH already listening is reused untouched.

1. `git fetch --force --tags origin` in `GIGACHAD_LEARNHARNESS_REPO` (120 s, no prompts).
2. Release tags (`v*`) newer than `served-commit` are tried newest first in `<GIGACHAD_LEARNHARNESS_DIR>-check` (keeps its `node_modules`): `seams.patch` applies, launcher markers, `tsc --noEmit`, `node --test custom/*.test.mjs`, from the build's allowlisted environment. The first passing one is served.
3. Results are cached per commit and overlay digest in `update-checks/` (failures with a `.log`); an overlay fix rechecks. Failed installs (network) are not cached.
4. The pre-bump snapshot runs before the worktree moves. If the accepted release fails install or build, it is recorded as failed and the last working commit is served.
5. A failure never blocks the start: it serves the last working commit and prints e.g. `LearnHarness not updated, serving v0.4.0 (dc4c398): v0.5.0 failed tsc (log: …)`. A switch prints `LearnHarness update: <old> -> <new>`. `GIGACHAD_AUTO_UPDATE=0` skips fetch and checks.
6. `redeploy.sh` runs `run-learnharness.sh --select` (prints the commit a start would serve, while LH keeps serving) and restarts LH when it differs from `served-commit` and no session is busy.

The checks are a gate, not a review: after an update, do manual steps 1, 3, 4 when convenient. A failing release needs an overlay fix (its log names the check).

### Manual (pinned ref)

For a non-release commit (`origin/main`) or to hold a version. Stop at the first failure:

1. Read `git -C ~/git/learnharness log --oneline <old>..<new>`, especially storage, `app/api/chats`, `app/api/chat`, settings, `components/{chat,model-picker,sidebar,top-bar}.tsx`.
2. Check checkout at the new ref: `tsc`, overlay tests, `npm run build`.
3. Map every guard failure; fix `upstream_markers` in the same commit. Recheck outside the overlay: chat API shapes in `src/lib/learnharness.py`, the gate's refused routes (`app/api/keys`, `custom-provider`, `subscriptions`), the `settings.json` keys and `auth.json` binding the generator writes.
4. Live smoke on your own stack (check-checkout LH on a spare bind, scratch home with copies of the served `settings.json`/`auth.json`, own backend/frontend ports, throwaway sessions), walking the shell contracts above, including an old-protocol frame showing upstream UI and the restart banner.
5. Production: set `GIGACHAD_LEARNHARNESS_REF` in `$ENV`, restart LH when no session is busy (`redeploy.sh` does this when the ref moved), smoke one tab. A pinned ref is never updated and its failure ends the start (no fallback). Unset it to return to automatic.

### Pre-bump snapshot and rollback

- On a start at a commit other than `served-commit`, the launcher copies `data/`, `auth.json` and `*-credentials.json*` to `snapshots/<old-sha>-<utc>/`. An identical snapshot for that commit is not taken twice. `served-commit` moves only once the new commit listens.
- Roll back: stop LH, copy the snapshot into a fresh home, write the old sha to `served-commit`, pin the old ref ([deployment](deployment.md), "Roll back safely").

## Models: generator and gateway

- LH's one custom endpoint: pi-host's gateway (`pi_host.gateway_url()` = `GIGACHAD_PI_HOST_URL` + `/v1`; production `127.0.0.1:8301`). Only providers Pi serves, Ollama via Pi's generated `ollama` provider (`lib.ollama_models.sync_pi_provider`). LH authenticates with `CUSTOM_OPENAI_API_KEY` = `pi_host.gateway_token()`, which opens `/v1/*` only.
- `python -m lib.learnharness_config write-config` runs right before the unit starts and in `redeploy.sh` (`deploy/run-production-learnharness.sh models`). It reads one login's catalog (`GIGACHAD_LEARNHARNESS_LOGIN`, else the OS user; read-only) and installed Ollama models, syncs Pi's Ollama provider, reads the gateway's `/v1/models`, and writes under a lock (staged, renamed):
  - `settings.json` `customProvider`: gateway base URL, binding id, models with `image` from the row's `input`, `reasoning` from `thinking_levels`, `context`/`maxTokens` from `context_window`/`max_tokens`, else 128000/16384 (LH would fall back to 8192/4096). The `strong`/`cheap`/`vision` tiers only when absent or no longer served. Other keys untouched.
  - the `auth.json` binding record, without the key.
- LH model id = gateway id, `pi_models.gateway_model_id(selector)` (`litellm_proxy/x` → `x`, `gemini/m` → `google/m`, else unchanged); the shell maps both ways with `gatewayModelId` (`src/frontend/lib/models.ts`, `features/learn/model.ts`) and offers catalog entries in `models.served`.
- Compatibility: a tier still naming a stored selector (LiteLLM-proxy era) keeps its pick and level under the gateway id.
- A catalog model the gateway does not serve is skipped with a warning. Transient failure (gateway down or serving none of the catalog, database, unknown login, empty catalog) writes nothing and exits 1; LH starts with the last config. Ollama down only drops its models.
- Fresh home: cheap and vision are seeded from gigachad's defaults when served, else the first served (image) model. A served model whose provider has no credits makes background jobs and extraction fail silently: pick others in LH Settings → Agent.

## Sandbox

`run-learnharness.sh` runs LH as transient user unit `gigachad-learnharness-<port>` (`systemd-run --user`, `UMask=0077`):

- **Environment allowlist** (`env -i`): `PATH`, `HOME` (`<home>/home`), `LANG`, `NODE_ENV`, `LH_DATA_DIR`, `LH_AUTH_FILE`, `TMPDIR`, `CHROME_PATH`, `GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS`, `XDG_RUNTIME_DIR`, `DBUS_SESSION_BUS_ADDRESS`, `BRAVE_API_KEY`, `CUSTOM_OPENAI_API_KEY`. Never provider keys (their models would appear in LH's menus), never `LH_DESKTOP_*` (locks the frame out of LH's API).
- **Denylist** (`InaccessiblePaths`, absent paths skipped): `~/.secrets`, `~/.ssh`, `~/.omp`, `~/.pi/agent/auth.json`, pi-host token files (`pi-host.token`, `pi-gateway.token`, or `PI_HOST_TOKEN_FILE`/`PI_GATEWAY_TOKEN_FILE`), `~/.gnupg`, `~/.config/gh`, `~/.aws`, `~/.netrc`, `~/.git-credentials`, `~/.claude`, `~/.claude.json`, `~/.codex`, opencode config and data, `~/.local/share/keyrings`, `~/.password-store`, `~/.docker`, `~/.kube`, `~/.config/gcloud`, `~/.azure`, `~/.npmrc`, `~/.pypirc`, `~/.var/app`, `~/.config/google-chrome`, `~/.mozilla`, and the `.env*` files of this and the main checkout. **Add every new credential store.**
- Demos run in flatpak Chrome via a wrapper (`<home>/chrome`) sharing `TMPDIR` with flatpak's private `/tmp`; flatpak needs the user session bus.
- `npm ci` and `next build` run outside the unit with an allowlisted environment (no provider keys).
- Stopping the launcher stops the unit (`EXIT` trap). Inspect with `nsenter -t <MainPID> -U -m --preserve-credentials` and the process environment, as the launcher tests do.
- The unit's `ExecStart` in `/run/user/<uid>/systemd/transient/` holds the gateway token: never log the launcher's command line.
- Open (undecided by the user): bus and user manager are reachable inside, so code execution in LH could escape via `systemd-run --user` (LH's agent has no shell tool); a native Chromium (`CHROME_PATH=/usr/bin/chromium`) would remove the wrapper and bus. The denylist is incomplete by nature; `ProtectHome=tmpfs` plus bind mounts is the stronger alternative.

## Configuration

| Variable | Default / purpose |
|---|---|
| `GIGACHAD_LEARNHARNESS` | `1`; `0` skips LH in development (`run.sh`) |
| `GIGACHAD_LEARNHARNESS_REPO` | `~/git/learnharness`, cloned when missing, never edited |
| `GIGACHAD_LEARNHARNESS_DIR` | `~/.local/share/gigachad-bot/learnharness-upstream`, served worktree; check checkout `<dir>-check` |
| `GIGACHAD_LEARNHARNESS_REF` | Unset: newest release passing the checks. Set: pinned; a missing tag is fetched, a remote-tracking ref (`origin/main`) is fetched by `redeploy.sh` |
| `GIGACHAD_AUTO_UPDATE` | `1`; `0` serves the last working commit without fetching or checking |
| `GIGACHAD_LEARNHARNESS_BIND` | `127.0.0.1:30181`, loopback only |
| `GIGACHAD_LEARNHARNESS_HOME` | `~/.local/share/gigachad-bot/learnharness` (mode 700): `data/` (`LH_DATA_DIR`), `auth.json` (`LH_AUTH_FILE`), `snapshots/`, `served-commit`, `update-checks/`, `home/` (private HOME), `tmp/` |
| `GIGACHAD_LEARNHARNESS_URL` | `http://127.0.0.1:30181`, LH for backend and gate |
| `GIGACHAD_LEARNHARNESS_FRAME_URL` | Defaults to the URL above; production: the gate's HTTPS origin, given to host logins in `/api/config` (`learnharness_url`) |
| `GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS` | Local `:2999` origins; production defaults to `GIGACHAD_CORS_ORIGINS`. LH's CSP `frame-ancestors` and the origins its bridge trusts; baked into the build (change rebuilds; shell origin missing → blank frame) |
| `GIGACHAD_LEARNHARNESS_GATE_PORT` | `30182`, loopback identity gate |
| `GIGACHAD_LEARNHARNESS_LOGIN` | OS user in development; required in production (`users.tailscale_login`, must have signed in once; unknown login fails generation) |
| `GIGACHAD_LEARNHARNESS_BACKUP_DIR` | `~/.local/share/gigachad-bot/backups/learnharness` |
| `GIGACHAD_LEARNHARNESS_BACKUP_KEEP` | `14` archives |

## Production

```text
private Serve :8444 -> identity gate 127.0.0.1:30182 -> LearnHarness 127.0.0.1:30181 -> pi-host gateway 127.0.0.1:8301/v1
```

Host prerequisites:

| Need | Why |
|---|---|
| Node.js >= 24 and npm on `PATH`, not mise shims (`/home/linuxbrew/.linuxbrew/bin` here) | A mise shim exits 1 under the sandbox's private `HOME` |
| `git`, `rsync`, `curl` | Worktree, overlay copy, health checks |
| `flatpak` with `com.google.Chrome` | Demo rendering |
| User session bus (`dbus.socket`), lingering (`loginctl enable-linger`) | flatpak portal; the unit sets `XDG_RUNTIME_DIR=%t`, `DBUS_SESSION_BUS_ADDRESS=unix:path=%t/bus` and orders after `dbus.socket` |
| systemd user manager with unprivileged user namespaces | `systemd-run --user` with user-mode `InaccessiblePaths` (checked on systemd 262) |
| `src/frontend/node_modules` | The launcher packs `@gigachad/chat-ui`; `redeploy.sh` installs it |
| Production PostgreSQL and pi-host (`gigachad-bot-pi-host.service`, `127.0.0.1:8301`) | Generator reads catalog and gateway models; LH calls the gateway |

Units (installed by `deploy/install-systemd-service.sh`, run through `deploy/run-production-learnharness.sh server|gate`; `models` regenerates models alone):

| Unit | Runs | Listens |
|---|---|---|
| `gigachad-bot-learnharness.service` | `run-learnharness.sh` → transient `gigachad-learnharness-30181` | `127.0.0.1:30181` |
| `gigachad-bot-learnharness-gate.service` | `uvicorn backend.learnharness_gate:app` | `127.0.0.1:30182`, Serve HTTPS `:8444` |
| `gigachad-bot-learnharness-backup.timer` | `deploy/backup-learnharness.sh` daily | - |

- The service wraps the launcher instead of carrying its own sandbox, so development and production share one sandbox definition (the denylist is computed, including both checkouts' `.env*`).
- `KillMode=mixed`: TERM goes to the launcher only; its trap stops `gigachad-learnharness-30181` and exits 143 (`SuccessExitStatus=143`). The wrapper has `NoNewPrivileges` and `UMask=0077`, no `PrivateTmp` (it would put the launcher in its own user namespace).
- Exits 78 without restart when `GIGACHAD_LEARNHARNESS_LOGIN` or `POSTGRES_PASSWORD_PROD` is missing. Generation failures keep and serve the last settings.
- `deploy/run-production-learnharness.sh` sets `GIGACHAD_DATABASE_URL`/`PGPASSWORD` like `run-backend.sh --prod` (`127.0.0.2:5433`, OS user, `POSTGRES_PASSWORD_PROD` from `~/.secrets`) and `GIGACHAD_PI_HOST_URL` (`http://127.0.0.1:8301`) unless set; only the generator sees them.
- The retired LiteLLM proxy unit (`gigachad-bot-litellm-proxy.service`) is stopped and removed by `deploy/install-systemd-service.sh`.

`$ENV`:

```dotenv
GIGACHAD_LEARNHARNESS_LOGIN=<your Tailscale login>
GIGACHAD_LEARNHARNESS_FRAME_URL=https://<tailnet-hostname>:8444
```

Start:

```bash
"$PROD/deploy/install-systemd-service.sh"
systemctl --user start gigachad-bot-learnharness.service \
  gigachad-bot-learnharness-gate.service gigachad-bot-learnharness-backup.timer
"$PROD/deploy/tailscale-serve.sh" 30182 8444
systemctl --user restart gigachad-bot.service
```

- First start runs `npm ci` and `next build` (~2 min).
- **Never publish LH directly**: no login, answers any origin. The gate forwards only logins granted `gigachad.local/cap/host` and refuses writes to LH's provider keys, custom endpoint and subscriptions.
- Development's `run.sh` reuses production's LH on the default port, same data. If a development instance holds `30181` when the unit starts, the unit exits inactive: stop the development one and start the unit again.
- Check: `curl -sf http://127.0.0.1:30181/` succeeds; `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30182/` → 403; `/api/config` lists `learnharness_url` for a host login. From a permitted device, open Learn in the sidebar and start a disposable session.
- Logs: `journalctl --user -u gigachad-bot-learnharness.service` (launcher + LH output via `systemd-run --pipe`), `-u gigachad-learnharness-30181.service` (start/stop).

### Backup and restore

- The timer (`Persistent=true`) archives `data/`, `auth.json`, `*-credentials.json*` and `served-commit` from the home to `$GIGACHAD_LEARNHARNESS_BACKUP_DIR/learnharness-<utc>.tar.gz`, keeping the newest `GIGACHAD_LEARNHARNESS_BACKUP_KEEP`. LH keeps running; files are copied as they are (chats one JSON each, graph ledgers appended JSONL). `snapshots/` is not archived.
- Archives hold chats and LH credentials (mode 600, dir 700): point the backup dir only at a place allowed to hold credentials (another disk for an off-host copy).
- Run now: `systemctl --user start gigachad-bot-learnharness-backup.service`.

```bash
systemctl --user stop gigachad-bot-learnharness-gate.service gigachad-bot-learnharness.service
H="$HOME/.local/share/gigachad-bot/learnharness"
mv "$H" "$H.before-restore-$(date -u +%Y%m%dT%H%M%SZ)"
"$PROD/deploy/backup-learnharness.sh" restore "$HOME/.local/share/gigachad-bot/backups/learnharness/learnharness-<utc>.tar.gz"
systemctl --user start gigachad-bot-learnharness.service gigachad-bot-learnharness-gate.service
```

- `restore` refuses a home holding `data/` or `auth.json`, and prints the commit that wrote the data. If the served ref differs, the launcher snapshots and serves the data with the new commit; pin `GIGACHAD_LEARNHARNESS_REF` to the writing commit first to avoid that.
- The next start rewrites the custom provider snapshot and binding from the current catalog; the learner's tier choices stay.

## Upstream facts and pitfalls

- Maintainer: owner-driven, rarely merges outside PRs. Proposals are posted by the user (issue #306); drafts (U4 "Embedding host API", U1 type/radius ladder) in `.wayfinder/learnharness-integration/spec/upstream-proposals.md`.
- No backward compatibility for stored data (an older graph format was wiped). Hence snapshot and backup.
- No X-Frame-Options, no auth, no Origin checks on most routes (`proxy.ts` is open unless `LH_DESKTOP_*` is set, which would 403 the frame). `POST /api/custom-provider` needs ETag + Origin; `PUT /api/settings/models` replaces the whole section: always send all three tiers.
- Sessions stamp the strong tier at `POST /api/chats`; `PUT /api/chats/<id>/settings` changes one session. A stamped model gone from the catalog makes turns fail with 503 until a served model is picked.
- Extraction into Knowledge runs after a turn once untransferred transcript reaches `graph.extractionChunkTokens` (default 4096), single-flight per session, extractor → auditor → merger on the cheap tier; "Extract now" pushes the remainder; Practice sessions are never extracted.
- The plan is authored once after probing (`plan_init`, validated DAG); node states derive from quiz results.
- The launcher reuses whatever listens on its port, warning only when it refuses frames from the first ancestor or serves another overlay. Restart to apply a change.
- The sandbox unit takes ~90 s to stop; an immediate restart fails with "already loaded".
- `ModelDropdown`'s "Manage models" panel must get the **full** catalog; narrowed lists go through `pickModels` (a filtered catalog once wiped `model-providers.yaml`).
