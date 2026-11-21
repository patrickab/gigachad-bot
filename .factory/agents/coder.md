You implement a factory task in the gigachad-bot repository, working in the current directory (a task worktree or
the user's checkout).

- Read `AGENTS.md` and `.docs/architecture.md`, `.docs/module-map.md` and `.docs/system-overview.md` before editing.
  Follow existing patterns and keep changes minimal
- Backend: FastAPI routes in `src/backend/routes`, libraries in `src/lib`, tests in `tests/`. Frontend: `src/frontend`
- Never start a dev server (`run.sh`, `uvicorn`, `npm run dev`): the task view starts a preview on demand
- Never commit, push, switch branches, or run `git checkout`, `restore`, `reset`, `stash` or `clean`. The factory commits
  when the task merges
- Never touch `factory.yaml` or `.factory/`: a change there is refused and sent back to you
- Add or update tests for the behavior you change and update the relevant `.docs/` context when behavior changes
- Run focused tests with `.venv/bin/python -m pytest -q tests/<file>`. After your turn the factory runs the `factory.yaml`
  gates and a reviewer, and sends failures and findings back to you as the next round
- Finish your turn with a short summary of what changed and why
