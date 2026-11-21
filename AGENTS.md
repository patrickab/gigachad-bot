## Running the dev server

The backend (`uvicorn`, port 8001) and frontend (`next dev`, port 2999) start
via `run.sh`.

- Check whether the app is already up first (ports 2999 and 8001). If it is,
  use that instance. Never launch a second copy alongside it.
- Otherwise start it yourself in tmux via `agent-proc start app './run.sh'`
  (global AGENTS.md, "Long-Running Processes"), tell the user
  `tmux attach -t gigachad-bot`, and `agent-proc stop app` when done.
- Never start it with plain background jobs, `nohup` or raw tmux.
- Exception: the agent factory engine starts previews of task worktrees (backend
  and frontend on ports 8100-8199, `.docs/architecture.md`, "Agent factory").
  That is the engine's job, you never start one yourself.

## Testing

- `scripts/check.sh [lint|types|python|frontend|pi-host]` is the one pipeline: local runs, factory gates (`factory.yaml`) and before a redeploy. Run it before you call work done.
- It is hermetic (`env -i`), and gives the Postgres-backed tests a throwaway database on the dev or production PostgreSQL. Never rely on variables from your shell in tests: `tests/__init__.py` strips the host's `GIGACHAD_*`, `PI_*` and similar for plain `pytest` too.
- Tests never reach the network or a real model: stub `completions`, pi-host and the engine.
- `lint` runs ruff on Python files changed since `origin/master`: fix what ruff reports in a file you touch.

## Documentation

- Follow the global documentation standard. Keep generated context complementary: system overview for concepts/workflows, architecture for implementation contracts, module map for source ownership and navigation.
- Regenerate `.docs/system-overview.md`, `.docs/architecture.md`, and `.docs/module-map.md` through `/repomix`. Preserve operational details in maintained runbooks and load them only for relevant tasks.
- `.docs/deployment.md` — Private production deployment, identity, service configuration, verification, and rollback.
- `.docs/remote-inference.md`: Remote OCR/Ollama setup, SSH tunnels, and environment configuration.
- `.docs/integration-learnharness.md`: everything about learn mode (LearnHarness): goal, decisions, architecture, rules, operations. Read it only when working on learn mode.
- `.docs/pi-migration.md`: moving from OMP to Pi (ADR 0003): pi-host, contracts, decisions, phases. Read it for any Pi or pi-host work. Keep it current as phases land, delete it once its lasting parts are in `.docs/architecture.md`.
- `.docs/factory-on-pi.md`: plan for rebuilding the agent factory on Pi sessions (decisions, pipeline, phases). Read it for any factory work. Delete it once its lasting parts are in `.docs/architecture.md`.
- `templates/latex/AGENTS.md`: rules for the LaTeX presentation template sets (primitives, theme, re-rendering `example.tex`/`example.pdf` after every change). Read it before touching `templates/latex/`.

## UI hints

- Prefer intuitive, minimal UI over explanatory hints for every input or keybinding. Do not add routine helper text or instructional placeholders.
- Use consistent keybindings: Enter confirms, Esc cancels or closes, Shift+Enter inserts a newline in multiline inputs.
- Keep accessible labels and meaningful errors, they are not instructional hints.
- No hover tooltips, except on rendered markdown.

## UI style

Clean, calm UI that lets the user focus on the content (chat, document, canvas).

- **Content first**: the work surface gets the space. Chrome stays small and quiet.
- **Minimal**: no decoration without a function. Remove an element before adding one.
- **Low noise**: muted colors, no heavy borders, no badges or animation for their own sake.
- **Progressive disclosure**: secondary controls appear on hover or focus, and common actions stay one click away.
- **Consistent**: new controls copy the shape, size, colors and states of their neighbors.
- **Stable**: adding a control must not shift or resize existing ones.
- **Shared primitives**: reuse the components in `src/frontend/components/ui/` before writing custom markup. Extend a primitive if it falls short, rather than forking a one-off.
- **Sidebars**: every sidebar is a `SidePanel` (`components/ui/SidePanel.tsx`, floating card, eased open/close, optional drag-resize), its scrolling content a `SidePanelScroll` (edge fades, overlay thumb). No hand-rolled `<aside>` panels.
- **Accessible**: icon-only buttons have an `aria-label`. Keyboard focus is always visible.

## Jupyter notebooks

`.ipynb` files open in `@datalayer/jupyter-react`. Architecture: `.docs/architecture.md`, "Jupyter notebooks".

- **Adapter-rule exception**: `@jupyterlab/services` speaks the Jupyter protocol from our bundle, but all traffic goes through the backend proxy `/api/jupyter/server/` (same origin, the token stays server-side). The browser never reaches a Jupyter server directly
- **One place per side**:
  - Frontend: `@datalayer/*`, `@jupyterlab/*`, `@jupyter/*` and `@lumino/*` are imported only in `src/frontend/features/notebook/` (eslint enforces it)
  - Backend: Jupyter server URLs, token and response shapes are known only by `src/lib/jupyter.py`, which runs in the one Jupyter service process (`src/backend/jupyter_service.py`). Workers reach it only through `src/lib/jupyter_service.py`, never hold a `JupyterHost` (each worker would get its own server and kernels). `routes/jupyter.py` maps its errors and never builds upstream URLs
- **Keep the `package.json` overrides** `@jupyter/web-components` and `@jupyter/react-components` at `^0.17.1`. With npm's default tree three copies of `@jupyter/web-components` share one FAST foundation and its design tokens recurse until the page freezes
- **`@datalayer/jupyter-react` is a fork tarball** (`package.json` points at the `patrickab/jupyter-ui` release asset `v2.0.21-gc.1`, the lockfile pins its integrity, `npm ci --ignore-scripts` installs it). The fork adds only the `Notebook` `editorExtensions` prop (upstream PR https://github.com/datalayer/jupyter-ui/pull/515), which `features/notebook/syntaxHighlight.ts` uses for the Neovim-scheme highlighting. When the PR is released on npm, return to the npm version and drop the tarball. Release runbook: `.docs/architecture.md`, "Jupyter notebooks"
- **No patch-package**: `redeploy.sh` installs with `npm ci --ignore-scripts`, so a postinstall patch never applies. Library fixes go into `next.config.mjs` as declarative webpack config
