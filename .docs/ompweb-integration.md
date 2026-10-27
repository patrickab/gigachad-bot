# omp-web integration runbook

How to change the omp tabs (gigachad-bot shell + upstream omp-web + private overlay), and how several agents work on it at once. Rules and ownership: `AGENTS.md`, "omp-web integration". Contracts: `.docs/architecture.md`, "omp tabs (omp-web)".

## Map

```
~/git/gigachad-bot/                      gigachad-bot main checkout (master)
~/git/worktrees/gigachad-bot/<branch>/   gigachad-bot worktrees, one per task
  ompweb-overlay/                        private overlay, becomes custom/ in the served omp-web
    root.tsx, framing.ts, next.config.ts   entry (CustomRoot), frame-ancestors CSP, module overrides
    components/                          EmbedBridge, EmbedMode, QaTurn, turn-summary, EmbeddedMarkdown
    embed/                               postMessage channel, completions, open-file, insets
    overrides/<upstream path>            replaces that upstream module, wraps it by relative import
    theme/                               embed-only CSS (html.cf-embed), palette/tokens mapping
    seams.patch                          the 2 upstream lines that load the overlay (keep at 2)
    *.test.mjs                           overlay tests (node:test + jiti, run inside an omp-web checkout)
  src/frontend/features/omp/             shell adapter: frame, protocol, composer, tabs (only place that knows omp-web)
  src/frontend/packages/chat-ui/         @gigachad/chat-ui: protocol.ts, palette, Q&A cards, markdown, code/diagram renderers
  src/lib/omp_web.py                     backend adapter (only place that builds omp-web URLs)
  run-omp-web.sh                         builds and serves omp-web (reset to ref + seams.patch + overlay + chat-ui)

~/git/ompweb_original/                   upstream clone kahme247/ompweb (main), never edited for gigachad
~/git/worktrees/ompweb_original/<topic>/ upstream topic worktrees (seams, fixes meant as upstream PRs)
  integration_patrick_ab                 integration branch: upstream main + our pending upstream PRs, the ref we serve
~/.local/share/gigachad-bot/ompweb-upstream/   served build checkout (D), owned by run-omp-web.sh, never edit by hand
```

## Where a change belongs

| Change | Place | Ships via |
|---|---|---|
| Look/behaviour of omp tabs only when embedded | `ompweb-overlay/` (override, component, embed CSS) | gigachad-bot commit, next omp-web restart |
| Anything both chat and omp tabs render, the postMessage protocol | `src/frontend/packages/chat-ui/` | same gigachad-bot commit (the launcher packs it) |
| Shell side of omp tabs (top bar, composer, tabs, frame layout) | `src/frontend/features/omp/` | gigachad-bot commit, frontend hot-reloads |
| omp-web lacks a seam (hardcoded value, unexported setup, bug) | upstream topic branch, PR to kahme247/ompweb, then into `integration_patrick_ab` | ref the launcher serves |
| omp-web server API, sessions, RPC | never | n/a |

Upstream changes must be refactors without behaviour change or features/fixes valuable to every omp-web user (issue #314 is the umbrella for seams). Follow upstream `CONTRIBUTING.md`: `tsc`, `npm run lint` (zero new warnings), `npm test`, i18n strings in all three locales, focused PRs.

## Workflows

### gigachad-bot change (overlay, chat-ui, shell)

1. Own worktree off the integration branch of gigachad-bot (`ompweb-upstream` until merged to master):
   `git -C ~/git/gigachad-bot worktree add -b <task> ~/git/worktrees/gigachad-bot/<task> ompweb-upstream`
2. Edit. Protocol change → bump `EMBED_PROTOCOL_VERSION` in `packages/chat-ui/src/protocol.ts`, handle the message on both sides in the same commit.
3. Shell checks (`src/frontend`): `npx tsc --noEmit`, `npx eslint .`, `npx vitest run`. Backend: `pytest tests/test_omp_web.py tests/test_omp_web_launcher.py`.
4. Overlay checks need a private omp-web checkout (below). Never rsync into the served checkout D, another agent's overlay would be replaced.

### Private overlay check checkout

```bash
task=<task>; G=~/git/worktrees/gigachad-bot/$task; C=~/.local/share/gigachad-bot/ompweb-check-$task
git -C ~/git/ompweb_original worktree add --detach "$C" integration_patrick_ab   # once
cd "$C" && npm ci && git apply "$G/ompweb-overlay/seams.patch"                    # once
# every check:
rsync -a --delete "$G/ompweb-overlay/" "$C/custom/"
T=$(cd "$G/src/frontend" && node packages/chat-ui/pack.mjs --out "$C/node_modules/.gigachad-chat-ui") \
  && npm install --no-save --no-audit --no-fund --ignore-scripts "$T"
node_modules/.bin/tsc --noEmit && npx eslint custom && node --experimental-strip-types --test custom/*.test.mjs
# done: git -C ~/git/ompweb_original worktree remove --force "$C"
```

An override typechecks against the upstream module it wraps, so an upstream change that breaks it fails `tsc` here.

### Upstream change (seam or fix)

1. Topic worktree off upstream `main` (PRs target main): `git -C ~/git/ompweb_original worktree add -b <topic> ~/git/worktrees/ompweb_original/<topic> main`
2. Change, add/adjust upstream tests, run upstream checks. Baseline: `npm test` passes ~1700, lint has ~10 pre-existing warnings.
3. The integrator cherry-picks the commit into `integration_patrick_ab`. Only commits there reach the served omp-web: `run-omp-web.sh` resets D to the ref, uncommitted edits in any worktree are invisible to it.
4. PR to kahme247/ompweb only with the user's approval. Never push without it.

## Shared resources and concurrency

| Resource | Owner | Rule |
|---|---|---|
| Frontend 2999 + backend 8001 (`run.sh`, tmux `gigachad-bot`) | user | Never start a second copy, never restart without the user. Serves one gigachad-bot worktree only |
| omp-web 30177 (served checkout D) | integrator | One instance per machine, it owns the omp session processes. Others never stop or rebuild it |
| `integration_patrick_ab` branch | integrator | Agents commit to topic branches, the integrator cherry-picks |
| `ompweb-upstream` branch (gigachad-bot) | integrator | Agents commit to task branches, the integrator merges |
| omp sessions | user | Never send prompts or click Edit/Fork in existing sessions. Live tests only in a throwaway session the user approved (e.g. new session in `/tmp`) |

Parallel-safe slices: overlay theme/CSS, a single overlay override, shell `features/omp/`, `chat-ui` components, one upstream seam per topic. Serialize: `protocol.ts` (one owner per protocol version), `seams.patch`, `run-omp-web.sh`, the overlay `next.config.ts` override list.

## Restart and live smoke (integrator)

1. Merge task branches into the served gigachad-bot worktree, cherry-pick upstream commits into `integration_patrick_ab`.
2. Check omp-web is idle: `curl -s http://127.0.0.1:30177/api/sessions` → `runningSessionIds` empty.
3. Restart: stop the `node bin/omp-web.js --port 30177` process, then `agent-proc start ompweb ./run-omp-web.sh` from the served worktree. Build takes ~2 min.
4. Smoke at `http://localhost:2999` (not `127.0.0.1`: Next dev never hydrates there): open an omp session, switch omp tabs, check the changed surface. Send prompts only in an approved throwaway session.

## Pitfalls

- `run.sh` relaunches `run-omp-web.sh` when 30177 is down: stopping omp-web while `run.sh` runs races two builds ("Another next build process is already running"). Wait for the port instead of starting a second launch.
- A launch fails if `src/frontend/node_modules` is being reinstalled (`pack.mjs` needs `typescript`). Retry after the install.
- Moving or deleting frontend files under a running `next dev` can leave a stale module graph (500, "Element type is invalid"). Restarting `run.sh` fixes it, ask the user.
- The served checkout D only changes on a restart. A built overlay change is invisible until then.
- `custom/` in D is overwritten on every start, edit `ompweb-overlay/` only.
- Overlay CSS that targets omp-web markup (`[role=log]`, `.chat-input-shell`, `.activity-row*`) silently stops matching after an upstream rename: prefer an upstream class hook or CSS variable seam.
