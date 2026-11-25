# Agent factory

A factory task is a request a Pi coding session carries out on its own: code, check, review, merge. This page is the
behavior. Implementation contracts: `.docs/architecture.md` ("Agent factory" rows). Source map: `.docs/module-map.md`.

## Where the plugin lives

- **Not in this repo.** The factory is the package `agent-factory`, a repository of its own:
  [`patrickab/agent-factory`](https://github.com/patrickab/agent-factory) (locally `~/git/agent-factory`). It ships the
  core (`src/core.ts`), the `factory` CLI, the `factory-mcp` server and the Pi extension (`src/pi/`).
- **Installed** from a clone: its `factory` CLI on pi-host's PATH (the shell's under `run.sh`, the pi-host unit's in
  production) or named by env `AGENT_FACTORY_CLI` (production: `.docs/deployment.md`, "Provision the agent factory"),
  and the clone registered as a Pi package (`pi install <clone>`), so every Pi session loads the extension.
  Tasks the old plugin location (`~/.pi/agent/extensions/factory/`) started keep running.
- **Why**: it runs in any Pi session on any repo (the `factory` tool, `/factory` commands), with or without gigachad.
  It is the factory core: repos, settings, `factory.yaml`, tasks, their runners and files. gigachad is a thin client.
- **Factory tasks of this repo cannot change it.** A factory change is a commit in `agent-factory` (tests: `bun test`
  and `bun run types` there). A change to the contract below needs a commit in both repos.
- **Contract with gigachad**: the core's actions (`core.ts`: `ACTIONS`, `callAction`, `TaskView`, `TimelineEntry`,
  see the package's README), which pi-host imports from the path `factory --core-path` prints, and the `preview`
  setting the core stores for gigachad without acting on it.
- **Not installed** (no `factory` CLI, a failing one, or a core path that does not exist): pi-host answers
  `factory_plugin_missing`, the board and a failed start name the link above.

## Parts

- **Core** (the plugin's `core.ts`): every factory feature as an action over plain JSON (`repos.*`, `branches.list`,
  `settings.get|set`, `config.get|write`, `tasks.list|get|timeline|diff|start|approve|retry|cancel|remove|message`,
  host-only `tasks.import`). A detached runner (`bun runner.ts`, `pi --mode rpc`) holds every task, whoever started it.
  State: `<agent dir>/factory/` (task records, per-repo settings files, task sessions).
- **pi-host**: `POST /factory/call {action, params}` and `GET /factory/actions` run the core in-process (its path asked
  once from `factory --core-path` and kept while it imports; imported again when `core.ts`'s mtime changes; a change in
  another module of the package needs a pi-host restart). A runner it starts
  inherits pi-host's environment, as a terminal-started task inherits the shell's. It never opens or prompts a factory
  session; a change of a task session, record or runner lock under `<agent dir>/factory/` sends `sessions_changed`.
- **Runners and restarts**: under systemd the plugin starts a runner through `systemd-run --user --scope`, outside
  pi-host's unit, so a pi-host restart leaves it running. `redeploy.sh` still counts live runners as running
  sessions and does not restart pi-host while one runs.
- **Backend** (`routes/factory.py`, `lib/pi_host.py`): `POST /api/factory/call` and `GET /api/factory/actions`, a pass
  through (a refusal keeps the core's `detail`). A `tasks.start` without `tag` gets the requesting login as its tag
  (the core would fall back to the host's system user, shared by every login), each character the core refuses in a tag replaced by `-`. Its own parts: reverting a merged task's commit on `tasks.remove` (`lib/factory_revert.py`: a dry run refuses a conflicting revert first, then, once the core removed the task, `git revert` in the repo's checkout; last, because the revert reloads the dev server), previews (stopped after a successful `tasks.remove`)
  and the one-time move of the old Postgres rows (`lib/factory_migration.py`). It stores no factory state.
- **Plugin** (the package's Pi extension, see above): runs the pipeline inside the task's own Pi session. Every step is a `custom`
  entry in the session file, the task is rebuilt from them after a restart.
  - `index.ts` session hooks and commands, `pipeline.ts` round and merge control flow, `merge.ts` rebase and merge,
    `queue.ts` merge queue, `review.ts` reviewer child, `config.ts` `factory.yaml`, `state.ts` entries and feedback.
- **Where sessions live**: `<agent dir>/factory/sessions/<repo dir name>/` (agent dir: `PI_CODING_AGENT_DIR`, default
  `~/.pi/agent`), apart from the user's Pi sessions. Older tasks keep their files in Pi's normal root;
  `GET /api/pi/sessions` drops every session with a `factory` state, so the session picker and sidebar never show a task.
- **Frontend** (`src/frontend/features/factory/`): board, task view (timeline plus a message box: `tasks.message`),
  settings; all through `callFactory(action, params)`. The core takes text only: images attached to a request or
  message go to `POST /api/factory/images` (`lib/factory_images.py`, stored in `~/.cache/gigachad/factory-images/`) and
  their absolute paths are appended to the text for the coder to open (`requestImages.tsx`).
- **Repo** (`factory.yaml`, `.factory/agents/{coder,reviewer}.md`): hooks, gates, protected paths, preview, role
  prompts. Read from the task's base commit, never from the tree the coder edits.

## Lifecycle

1. **Start** (`tasks.start`): the core makes a worktree off the task's branch, which it also merges into, runs
   `hooks.after_create` and spawns the task's runner. A task never runs in the repo's main checkout.
2. **Rounds**: coder turn → `hooks.after_code` → `gates` → protected-path check → review (when on, off by default).
   A problem goes back to the coder as feedback and opens the next round, up to `maxRounds`. The tests run only after
   that, in the merge queue: a round the reviewer rejects costs no test run.
3. **Pass**: `merge: auto` merges, `merge: review` waits for approve.
4. **Merge**: the merge queue below.
5. **End**: `done` (merged, or `no_changes` when the diff is empty), else `blocked` with a reason.

- States: `running`, `waiting_approve`, `waiting_answer`, `merging`, `done`, `blocked`, `cancelled`.
- Commands: `approve` (from `waiting_approve`), `retry` (from `blocked`/`cancelled`, a fresh round budget), `cancel`.
  A plain message to a resting task opens new rounds.
- **Questions**: the coder may ask (`factory_ask_supervisor`): the task waits in `waiting_answer` with its `question`,
  and the next message answers it, continuing the same round.
- **Waves**: a supervisor declares a wave (`waves.start`): a tag, a title, the reviewer's instructions and its tasks
  with which waits for which. The factory builds them on the wave's own branch, then reviews and merges the whole
  branch. A task joins a wave only through `waves.start`/`waves.add` and carries its `wave` id; the core has no merge
  request kind any more.
- **Tags**: every task has a `tag`, a login or, in a wave, the wave's tag (`plugin/m1/w1`), and may name its plan's
  memory_store (`plan`).

## Branches

- Source: the setting, else the repo's default branch (origin's `HEAD`), else `main`, else `master`. Target: the
  setting, else the source.
- The worktree is rebased onto the target when a run opens and again at merge, so a task always builds on the latest
  target.

## Checks

- **Rounds**: `gates` = `gates.sh --changed --reuse lint types`. Seconds, only what the change reaches.
- **Merge**: `merge_gates` = `gates.sh --changed --reuse`. The tests: affected vitest, all Python, lint, types.
- `factory/gates.sh`:
  - `--changed`: steps, TypeScript projects, eslint files and vitest tests by the paths changed since the merge base
    with the target. Docs alone run nothing. A path outside the known projects runs everything.
  - `--reuse`: a step that passed on the same tree (working tree with untracked files) and merge base is skipped.
  - Steps run in parallel at `nice 10`. pytest on `cpu_count - 1` workers, each with its own throwaway database.
  - A failed test is rerun once: passing then, the step passes and says `flaky`.
  - Output: a passed step is one line, a failed one a summary of its failures (failing tests with their error, type
    and lint errors, at most 50 lines) and the path of its full log (`<git common dir>/check-logs/`, last 20 runs).
    That is all the coder gets back.
  - Exit 75: the environment is broken (missing packages or tools, an empty submodule). Before failing it waits for
    `node_modules` still being copied and refills an empty submodule from the main checkout.

## Merge queue

One per repository, shared by every task's session through ticket files in `<git common dir>/factory-merge-queue`.

1. Commit the worktree, squash to one commit, rebase onto the target. A conflict → `conflict` round, no ticket.
2. Join the queue, first come first served. Status `queued for merge, N ahead`.
3. While waiting, each merge that lands rebases the waiting tasks. A conflict leaves the queue → `conflict` round.
4. On its turn: rebase, `merge_gates`, fast-forward the target (under `flock`).
   - A gate failure → `gates` round on the rebased branch. Nothing merged.
   - The target moved anyway (a merge outside the queue): rebase and retry, 3 attempts.
5. Leave the queue. After a merge that changed a lockfile, `hooks.after_merge` run in the main checkout.

- A ticket whose Pi process ended is dropped by the next reader: a crashed session cannot block the queue.
- Rounds a merge opens count against `maxRounds` like any other.

## Failures

| Reason | Meaning | What to do |
|---|---|---|
| `max_rounds: <trigger>` | the coder did not fix it within the budget | read the timeline, retry or message |
| `infra: <reason>` | a check exited 75, the environment is broken | fix the environment, retry |
| `merge_failed` | the fast-forward was refused, usually uncommitted edits in the main checkout on a file the merge touches | commit or stash, retry |
| `rebase_failed: …` | a rebase failed without a conflict (deleted branch, untracked file in the way) | fix, retry |
| `review_<status>` | the reviewer gave no usable verdict | retry |
| `pipeline_error: …` | an unexpected error | read it, retry |

- Review: `blocker` and `major` findings go back to the coder, `minor` ones pass.
- A worktree task touching a protected path gets a round to restore it.

## Observability

- **Board**: three columns, Working, Merging, Finished. What waits on the supervisor or the user (tasks
  `waiting_answer`, `waiting_approve`, `blocked`, interrupted; waves in Waiting below; refused starts) sits in a Waiting
  block atop Working, absent while nothing waits and outside Working's count. The gap between columns is a drag handle
  (and arrow keys) for one shared gap, 0 to 64px, stored in `localStorage` (`gigachad:factory-column-gap`).
- **Board card**: state, round, blocked reason, the open question, tokens in/out and cost (`formatFlow`, coder plus
  reviews).
- **Waves on the board** (`waveTree.ts`, `WaveTree.tsx`): built from `waves.list`, read with `tasks.list` on the same
  refresh events. Nothing sits above the columns: a wave is a card-shaped container inside a column (`waveColumn`):
  `building` (once a task of it is done), `reviewing`, `merging` in Merging; `rejected`, `blocked`, `waiting_approve`
  or interrupted in the Waiting block; `merged`, `cancelled` in Finished, closed by default. Its header (a `Disclosure`):
  `done/total`, the wave's title, stage, tokens and cost. Inside: the wave's done tasks as task cards (they leave
  Finished), then a link to the declaring supervisor's Pi session (`origin`), the outcome (`WaveOutcome.tsx`): reviews
  newest first in the task timeline's review row (`ReviewStep.tsx`; model, usage, duration, verdict, findings, a link
  to the reviewer's session), failed gate runs only, the merged commit range (`from..to`, short), the remote request's
  link; and the description. Declared tasks not started yet count only in the total. A wave's other tasks are ordinary
  cards in the columns of their states with a quiet line naming the wave (its tag's last segment).
  Waves of one column sharing a tag prefix are wrapped in one container per prefix segment (track, milestone):
  `merged waves/waves`, the title (first line of the prefix's description, heading marks stripped, else the segment),
  summed tokens and cost. A one-segment tag has no outer container. A session link appears only when its id exists
  and opens the session as a Pi workspace item (`piSessionItem`). A description is the document
  `plan/<tag or prefix>.md` of the plan's memory_store (`GET /api/memory/read`); its `Description` row appears only for
  a non-empty document.
  Tasks without a wave (or of a wave the list lacks) behave as before whatever their tag. A factory without
  `waves.list` (`unknown action`) shows no waves and no error.
- **Waves in a Pi tab** (`features/pi/SessionWaves.tsx`): above the composer, beside the subagent roster and in its
  shape, the waves the tab's session declared (`origin`) that are not `merged` or `cancelled`, from `waves.list` of the
  tab's repo on the board's refresh events. A collapsed row per wave (status icon, tag, title, `done/total`, stage,
  cost; a stage that needs the user in the warning colour) opens to its tasks that are not done or cancelled, a started
  one with an "Open session" button that opens it like the board. No waves (or `unknown action`): nothing renders.
  The helpers both features share (`waveStageLabel`, `waveNeedsUser`, `isUnknownAction`, `concernsFactory`) live in
  `lib/domain/factory.ts`: `features/pi` never imports `features/factory`.
- **Timeline** (task view): per round the trigger, the coder's turn (model, tokens, cost, last message, earlier ones
  collapsed), its questions, each gate with duration and failed output, the review with findings, the merge.
- Tokens count cache reads and writes as input; cost is the provider's figure.
- Logs: the backend logs failed or slow (> 5 s) pi-host calls, pi-host slow RPCs, requests and event-loop stalls.

## Settings

Per repo in the core (`settings.get|set`, one file per repo, which also registers it), overridable per task:
`merge`, `maxRounds`, `review`, `models` (`small`, `big`, `coder`, `reviewer`), `branch`; defaults and resolution are
the core's (`settings.get` answers `{stored, resolved, defaults}`). `preview` (boolean) is stored there for gigachad only.
A task keeps the settings it started with.

- **Models**: easy tasks of a wave run on `small`, everything else and the reviewer on `big` (`settings.get` answers
  `big` equal to `coder`). The settings dialog shows two pickers, Small model and Big model; it reads big as
  `big ?? coder` and saves `models: {small, big, reviewer: <big>}`, an empty picker leaving its keys out.

- **Moved from Postgres**: at backend start, each row of `factory_repo_settings` / `factory_tasks` not yet marked in
  `migrated_at` moves: a repo's settings go to `settings.set`, starting from the old defaults (merge `review`, 3
  rounds) with its stored keys over them, only when the core stores none (else logged and kept); a task goes to
  `tasks.import` (its id as a dashed UUID, its old-id preview stopped). Each row is marked once it moved or never can
  (a task without a worktree of its own, a repo gone from disk: logged), so a task removed in the core never comes
  back. A failed row stays unmarked for the next start (retried 5 times, 30 s apart). The tables stay, unmarked rows
  readable.

## New repo

A repo needs a `factory.yaml` once, in either of two ways, both from the same generator (the plugin's, no model):

- **Plain Pi**: `/factory init` in a session of the repo (any session, no task needed) writes the draft into the repo's
  main checkout, uncommitted, and reports which file the gates were detected from. It refuses to overwrite an existing
  `factory.yaml`.
- **gigachad**: opening the board of a repo without one shows the draft (`config.get`) in an editable text area.
  Save (`config.write`) writes it uncommitted into the main checkout. New task stays disabled until the file exists.

Review and commit it: the factory reads `factory.yaml` from a task's base commit, so tasks pick it up from that commit on.

- **Draft** (the plugin's `draft.ts`, a pure function of the repo's files; the backend has no copy): fills
  the commented template `factory.template.yaml` (`hooks`, `gates`, `merge_gates`, a comment per key) with `gates`
  (lint, types, format) and `merge_gates` (everything, tests included). The highest-level source wins:
  `factory/gates.sh`, Makefile targets (`lint`, `typecheck`, `check`, `test`), `.gitlab-ci.yml` scripts, then
  `.pre-commit-config.yaml`, `package.json` scripts and `pyproject.toml` (ruff, mypy/pyright, pytest) together. From
  GitLab CI only lint-like and test-like stages count; jobs that deploy, publish, build images, are manual or need
  CI-only variables are skipped, and so are install lines. Nothing found: the template as it is.
- **Contract**: a gate is one argv list, run in the task's worktree, judged by its exit code: 0 pass, 75 infra
  (the task blocks, no round), anything else fails. On failure the command prints a summary (at most 50 lines) on
  stdout: that is all the coder gets back. GitLab CI is never executed and there is no merge-request mode.
- **CI**: call the same script from the repo's CI, so the factory and CI judge the same thing.
