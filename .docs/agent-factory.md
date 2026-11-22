# Agent factory

A factory task is a request a Pi coding session carries out on its own: code, check, review, merge. This page is the
behavior. Implementation contracts: `.docs/architecture.md` ("Agent factory" rows). Source map: `.docs/module-map.md`.

## Where the plugin lives

- **Not in this repo.** The factory plugin is a Pi user extension in the owner's dotfiles:
  [`~/.os/chezmoi/dot_pi/agent/extensions/factory/`](https://github.com/patrickab/.os/tree/master/chezmoi/dot_pi/agent/extensions/factory), deployed by chezmoi to
  `~/.pi/agent/extensions/factory/`. Edit the source in `~/.os`, never the deployed copy.
- **Why**: it runs in any Pi session on any repo (`/factory init`, `/factory start`), with or without gigachad.
  gigachad only starts and monitors tasks on top of it.
- **Factory tasks of this repo cannot change it.** A plugin change is a commit in `~/.os` (tests: `bun test` and
  `bun run types` in the plugin's directory), then `chezmoi apply`. A change to the contract below needs a commit in
  both repos.
- **Contract with gigachad**: the commands (`/factory start|approve|retry|cancel|init`), the `factory-*` session
  entries (`.docs/architecture.md`, "Agent factory") and `draft.ts`'s `initDraft(root)`, which pi-host imports from
  the installed plugin.
- **Not installed**: pi-host answers `factory_plugin_missing`, the board and a failed start name the link above.

## Parts

- **Backend** (`src/lib/factory.py`, `routes/factory.py`): repos and their settings, a task's start (workspace, session,
  `/factory start`), list, diff, commands, messages, deletion. Stores no state: a task's state is its session's last
  `factory-state` entry, which pi-host reports.
- **Plugin** (`~/.pi/agent/extensions/factory/`, see above): runs the pipeline inside the task's own Pi session. Every step is a `custom`
  entry in the session file, the task is rebuilt from them after a restart.
  - `index.ts` session hooks and commands, `pipeline.ts` round and merge control flow, `merge.ts` rebase and merge,
    `queue.ts` merge queue, `review.ts` reviewer child, `config.ts` `factory.yaml`, `state.ts` entries and feedback.
- **pi-host**: spawns and supervises the sessions. Its session list carries each task's `factory` state and `usage`,
  its entries endpoint the timeline (`?customType=factory-&assistant=1`).
- **Where sessions live**:
  - A new task's session file goes to `<agent dir>/factory/sessions/<repo dir name>/` (agent dir:
    `PI_CODING_AGENT_DIR`, default `~/.pi/agent`), one folder per repository, apart from the user's Pi sessions:
    `pi -r` never looks there. `POST /sessions` with `factoryRepo` (the repo's path) puts it there, the session still
    runs in the task's worktree.
  - pi-host lists that root only with `GET /sessions?factory=1` (the backend's task list), finds its sessions by id and
    never archives them (409 `factory`). Plain listings, `continue` and `/search` leave them out.
  - Older tasks keep their files in Pi's normal root, the `factory=1` listing covers both. `GET /api/pi/sessions` drops
    sessions with a `factory` state unless `factory=true` (the factory board's store list), so the session picker and
    the sidebar never show a task.
- **Frontend** (`src/frontend/features/factory/`): board, task view, timeline, settings.
- **Repo** (`factory.yaml`, `.factory/agents/{coder,reviewer}.md`): hooks, gates, protected paths, preview, role
  prompts. Read from the task's base commit, never from the tree the coder edits.

## Lifecycle

1. **Start**: a worktree (`agent-worktree`, branch `<title-slug>-<id8>`) off the task's branch, which it also merges
   into. The session opens there and gets `/factory start`. A task never runs in the repo's main checkout.
2. **Rounds**: coder turn → `hooks.after_code` → `gates` → protected-path check → review (when on, off by default).
   A problem goes back to the coder as feedback and opens the next round, up to `maxRounds`. The tests run only after
   that, in the merge queue: a round the reviewer rejects costs no test run.
3. **Pass**: `merge: auto` merges, `merge: review` waits for approve.
4. **Merge**: the merge queue below.
5. **End**: `done` (merged, or `no_changes` when the diff is empty), else `blocked` with a reason.

- States: `running`, `waiting_approve`, `merging`, `done`, `blocked`, `cancelled`.
- Commands: `approve` (from `waiting_approve`), `retry` (from `blocked`/`cancelled`, a fresh round budget), `cancel`.
  A plain message to a resting task opens new rounds.

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

- **Board card**: state, round, blocked reason, branch, total tokens and cost (coder plus reviews).
- **Timeline** (task view): per round the trigger, the coder's turn (model, tokens, cost, last message, earlier ones
  collapsed), each gate with duration and failed output, the review with findings, the merge.
- Tokens count cache reads and writes as input; cost is the provider's figure.
- Logs: the backend logs failed or slow (> 5 s) pi-host calls, pi-host slow RPCs, requests and event-loop stalls.

## Settings

Per repo (`factory_repo_settings`), overridable per task: `merge` (`auto`/`review`, default `review`), `max_rounds`
(default 3), `review` (default off), `preview`, `models` per role (`coder`, `reviewer`), `branch` (the one a task
starts from and merges into, unset: the repo's default branch, origin's `HEAD`, else `main`, else `master`). The
new-task dialog shows them at the repo's values: a model per role (the reviewer's only with review on), merge, review,
the branch. A task keeps the settings it started with.

## New repo

A repo needs a `factory.yaml` once, in either of two ways, both from the same generator (the plugin's, no model):

- **Plain Pi**: `/factory init` in a session of the repo (any session, no task needed) writes the draft into the repo's
  main checkout, uncommitted, and reports which file the gates were detected from. It refuses to overwrite an existing
  `factory.yaml`.
- **gigachad**: opening the board of a repo without one shows the draft in an editable text area
  (`GET /api/factory/config?repo=`, the draft comes from pi-host's `GET /factory/draft?cwd=`, which calls the plugin's
  `initDraft` directly: no session). Save (`PUT /api/factory/config?repo=`) writes it uncommitted into the main
  checkout and refuses to overwrite an existing file (409). New task stays disabled until the file exists.

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
