# Agent factory on Pi

Plan for replacing the Agents-in-a-box engine with a pipeline that runs inside ordinary Pi sessions. Delete this file
once its lasting parts are in `.docs/architecture.md`.

## Goal

A factory task is a normal pi-host session. One Pi extension drives the pipeline inside it. The backend only sets a
task up and lists tasks. Simplicity, one agent runtime, live steering, a useful overview.

## Decisions

| # | Decision |
| --- | --- |
| D1 | A task is a pi-host `coding` session plus a row in `factory_tasks`. The session file is the task's full record. |
| D2 | `pi-extensions/factory.ts` owns the pipeline: gates, review, max rounds, merge. Control flow is code, never a model decision. |
| D3 | Every pipeline step is written to the session as a structured custom entry. The overview is built from these entries only. |
| D4 | Worktree is optional per task (default from repo settings). Off: the agent works in the current checkout on the current branch. |
| D5 | Merge setting, worktree task: `auto` merges into main, `review` waits for approve. Current-checkout task: `auto` commits to the current branch, `review` leaves the changes uncommitted and approve commits. |
| D6 | A done task keeps its session and, for worktree tasks, its worktree and branch. A later message starts new rounds on the same branch and merges again. A done task is deleted by hand (the trash icon primitive of the repo's UI): that removes the session file, the worktree and the branch from disk. |
| D7 | Worktree environments reuse the mechanism subagent worktrees already use (global `AGENTS.md`, Subagents): Python is an `agent-venv` overlay on the main repo's `.venv` (under a second, sees every package, installs on top stay in the overlay), `node_modules` is a reflink copy (btrfs copy-on-write, no extra disk) made in the background while the agent starts. |
| D8 | After a merge that changed a lockfile, the pipeline syncs the main repo's environment. |
| D9 | Per-repo settings live in gigachad's database (merge, max rounds, worktree, review, preview, model per role). `factory.yaml` keeps what belongs to the repo: gates, hooks, preview processes. Chain: defaults, repo, task. |
| D10 | Roles are `.factory/agents/coder.md` and `reviewer.md`. Scout and plan stop being pipeline phases. |
| D11 | Previews start on demand from the task view, never automatically. The preview gate stays as is. |
| D12 | No sandbox, no workflows, no DBOS, no engine database, no port 8090. |
| D13 | Protected paths: the merge refuses a diff that touches `factory.yaml` or `.factory/` and reports it as a round problem. |
| D14 | One merge at a time per repo (lock in the main checkout's `.git`). One active current-checkout task per checkout. |

## Pipeline

```text
/factory start {request, settings}
  round 1..max_rounds:
    coder turn            (Pi agent, role coder.md)
    agent_before_settle:
      hooks after_code    fail -> next round with the output
      gates               fail -> next round with the output
      review              fresh read-only Pi run, role reviewer.md, sees the diff
                          findings -> next round with the findings
      all pass            -> leave the loop
  no passing round        -> blocked, with the last problem
  merge auto              -> merge (or commit)   -> done
  merge review            -> waiting for approve -> /factory approve -> merge -> done
  any later message       -> new rounds, same branch
```

- Merge of a worktree task: commit, rebase onto main, gates again, fast-forward main, sync main's environment if a
  lockfile changed
- Commands: `/factory start`, `/factory approve`, `/factory retry`, `/factory cancel`. The backend sends them through
  pi-host, a terminal Pi user types them
- Entry kinds: `factory-task` (request, settings), `factory-round`, `factory-gate` (argv, exit code, duration, output
  tail), `factory-review` (verdict, findings, reviewer session path), `factory-merge`, `factory-state`
- A liveness status keeps the session from idle eviction while a round runs
- After a crash or restart the extension reads its entries on `session_start` and resumes from the last recorded step

## Phases

### P0: Spikes

Prove the three things this plan assumes. Each is a throwaway script or extension, half a day together.

- `agent_before_settle` can run a long command, append entries and request a continuation, round after round
- A fresh read-only reviewer run from inside an extension (`pi -p` child or pi-subagents' API), with its own model
- How pi-host surfaces the latest `factory-state` entry in session lists and events without reading whole files

Exit: a written note per spike in this file, plan adjusted.

### P1: The extension

- `pi-extensions/factory.ts`: loop, hooks, gates, review, max rounds, merge, entries, commands, resume
- Reads gates and hooks from `factory.yaml`, roles from `.factory/agents/`
- Tests with a fake repo and stubbed gates

Exit: `/factory start` in terminal Pi takes a task on a throwaway repo from request to merged, and to blocked after
max rounds.

### P2: Workspaces

- `src/lib/factory_workspace.py`: create and remove worktrees, set the environment up (`agent-venv`, reflink copy of
  every `node_modules` in the background), refuse a second current-checkout task, delete a task (session file,
  worktree, branch)
- `scripts/check.sh` gates run against the overlay venv
- Full-copy fallback on a filesystem without reflinks

Exit: a worktree is usable for the agent in under a second, its gates pass, and free space stays flat over ten of them.

### P3: Backend

- Table `factory_tasks` (id, repo, session id, worktree, branch, settings snapshot) and `factory_repo_settings`
- `/api/factory`: repos, settings, tasks (create, list, get, approve, retry, cancel, delete), diff
- `src/lib/factory.py` shrinks to these. `factory_sessions.py` and the engine client go

Exit: the API tests pass without an engine running.

### P4: Frontend

- Board and task view on the new data, live from pi-host events, no polling
- Task view: diff as the main surface, pipeline timeline beside it, Pi session in the sidebar
- Timeline: one row per round with the coder's summary, each gate with pass or fail, duration and output tail, review
  findings. A step opens the session at that point
- Board sorted by what needs the user: waiting for approve and blocked first, each with a one-line reason
- Cost and tokens per round and per task
- Done tasks carry the trash icon primitive: delete removes session, worktree and branch
- New task dialog: worktree checkbox. Settings dialog: per-repo settings, no sandbox, no workflows

Exit: a task runs from the dialog to done in the browser, and can be steered from the sidebar while it runs.

### P5: Previews

- `src/lib/factory_preview.py`: start and stop the `factory.yaml` preview processes for a task on ports 8100–8199,
  with an empty database and the development identity
- Start and stop buttons in the task view. Previews stop when the task is deleted or the backend shuts down

Exit: a preview opens through the preview gate for a worktree task and for a current-checkout task.

### P6: Removal

- Delete `run-factory.sh`, the engine start in `run.sh`, `GIGACHAD_FACTORY_URL`, the `factory` database provisioning,
  the polling hook, the workflow and sandbox UI
- Update `.docs/architecture.md` ("Agent factory"), `AGENTS.md` and `.factory/agents/`
- Delete this file

Exit: no reference to the engine or port 8090 remains.

## Risks

| Risk | Answer |
| --- | --- |
| The settle hook cannot carry the loop | P0 decides. Fallback: pi-host drives the loop from outside the session |
| A crash mid-round | Resume from the last entry, gates run again |
| Many tasks at once overload the machine | No queue at first. Add a concurrency limit if it happens |
| A task in the current checkout collides with the user's own edits | One such task per checkout, the user chose that mode |
| The main repo's venv changes while tasks run (overlays see it live) | Accepted: it changes only on a merge, and tasks rebase onto main anyway |
