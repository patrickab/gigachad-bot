# Agent factory

A factory task is a request a Pi coding session carries out on its own: code, check, review, merge. This page is the
behavior. Implementation contracts: `.docs/architecture.md` ("Agent factory" rows). Source map: `.docs/module-map.md`.

## Parts

- **Backend** (`src/lib/factory.py`, `routes/factory.py`): repos and their settings, a task's start (workspace, session,
  `/factory start`), list, diff, commands, messages, deletion. Stores no state: a task's state is its session's last
  `factory-state` entry, which pi-host reports.
- **Extension** (`pi-extensions/factory/`): runs the pipeline inside the task's own Pi session. Every step is a `custom`
  entry in the session file, the task is rebuilt from them after a restart.
  - `index.ts` session hooks and commands, `pipeline.ts` round and merge control flow, `merge.ts` rebase and merge,
    `queue.ts` merge queue, `review.ts` reviewer child, `config.ts` `factory.yaml`, `state.ts` entries and feedback.
- **pi-host**: spawns and supervises the sessions. Its session list carries each task's `factory` state and `usage`,
  its entries endpoint the timeline (`?customType=factory-&assistant=1`).
- **Frontend** (`src/frontend/features/factory/`): board, task view, timeline, settings.
- **Repo** (`factory.yaml`, `.factory/agents/{coder,reviewer}.md`): hooks, gates, protected paths, preview, role
  prompts. Read from the task's base commit, never from the tree the coder edits.

## Lifecycle

1. **Start**: a worktree (`agent-worktree`, branch `<title-slug>-<id8>`) off the source branch, or the current checkout.
   The session opens there and gets `/factory start`.
2. **Rounds**: coder turn → `hooks.after_code` → `gates` → protected-path check → review (when on, off by default).
   A problem goes back to the coder as feedback and opens the next round, up to `maxRounds`. The tests run only after
   that, in the merge queue: a round the reviewer rejects costs no test run.
3. **Pass**: `merge: auto` merges, `merge: review` waits for approve.
4. **Merge** (worktree task): the merge queue below. Current-checkout task: one commit of the task's paths onto `HEAD`.
5. **End**: `done` (merged, committed, or `no_changes` when the diff is empty), else `blocked` with a reason.

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
| `protected: …` | a current-checkout task touched a protected path | restore it |
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
(default 3), `review` (default off), `worktree` (default on), `preview`, `models` per role (`coder`, `reviewer`),
`source_branch`, `target_branch`. The new-task dialog shows them all at the repo's values: a model per role (the
reviewer's only with review on), workspace, review and merge as pills, the branches. A task keeps the settings it
started with.
