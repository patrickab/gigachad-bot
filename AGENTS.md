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

## Documentation

- Follow the global documentation standard. Keep generated context complementary: system overview for concepts/workflows, architecture for implementation contracts, module map for source ownership and navigation.
- Regenerate `.docs/system-overview.md`, `.docs/architecture.md`, and `.docs/module-map.md` through `/repomix`. Preserve operational details in maintained runbooks and load them only for relevant tasks.
- `.docs/deployment.md` — Private production deployment, identity, service configuration, verification, and rollback.
- `.docs/remote-inference.md`: Remote OCR/Ollama setup, SSH tunnels, and environment configuration.
- `.docs/integration-ompweb.md`: omp-web integration runbook: worktrees, where a change belongs, overlay/upstream workflows, shared services and concurrent agent work.
- `.docs/integration-learnharness.md`: everything about learn mode (LearnHarness): goal, decisions, architecture, rules, operations. Read it only when working on learn mode.

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
- **Accessible**: icon-only buttons have an `aria-label`. Keyboard focus is always visible.

## omp-web integration

omp tabs embed unmodified upstream omp-web (`kahme247/ompweb`). Everything gigachad-bot needs on its side is a private overlay in this repo, `ompweb-overlay/`: embed mode, the postMessage bridge, the composer, Q&A cards, Markdown and the embed-only theme. `run-omp-web.sh` copies it into the served worktree as `custom/` and applies `ompweb-overlay/seams.patch`, the two upstream lines that load it (`app/layout.tsx`, `next.config.ts`). gigachad-bot adapts to omp-web, never the reverse. Architecture: `.docs/architecture.md`, "omp tabs (omp-web)".

- **Minimal delta upstream**: never ask for or make changes to omp-web's server API, session handling or protocol logic. Read what it already serves. Keep `seams.patch` at its two lines. A seam the overlay needs goes upstream as a pull request (the current ones: #332, for issue #314) that is a refactor or feature valuable to every omp-web user and easy for the maintainer to review, never gigachad-specific
- **Overrides wrap upstream**: `ompweb-overlay/overrides/<path>` replaces upstream `<path>` (`ompweb-overlay/next.config.ts`). An override imports the upstream module by relative path and types against its exported props/types, so an upstream change fails `tsc`
- **Embed only**: every overlay style is scoped to `html.cf-embed`, standalone omp-web keeps upstream's look
- **Drop before you change**: omp-web UI gigachad-bot does not want is hidden in embed mode (`ompweb-overlay/theme/`), not rewritten. Prefer hiding a feature over porting or patching it
- **Adapters own all omp-web knowledge**, one place per side, so an omp-web change lands in one file:
  - Backend: `src/lib/omp_web.py` only. Routes (`routes/omp.py`) map its errors, they never build omp-web URLs. Pydantic contract models validate every response, a shape change fails loudly with the field name (502). `src/backend/omp_web_gate.py` (production Tailnet gate in front of the frame) is a plain identity-checking proxy, keep omp-web API knowledge out of it
  - Frontend: `src/frontend/features/omp/` only. Nothing else knows the iframe, the postMessage protocol (defined in the shared package, eslint lets only this feature import it) or omp-web URLs. Other features import its public files (eslint `crossFeature`)
  - The browser never calls omp-web's API, the backend does, server to server
- **Never vendor omp-web code** (components, hooks, CSS). Write new code against its API with omp-web as reference
- **Shared contracts live once, in `src/frontend/packages/chat-ui/`** (`@gigachad/chat-ui`): the postMessage protocol (`src/protocol.ts`) and the role palette (`src/palette.css`). The shell imports the source, `run-omp-web.sh` packs it into the served worktree on every start, so both sides ship in one commit
- **One shared iframe**: tabs switch it by message, never by reload or by moving it in the DOM. Do not add a second frame or a per-tab frame
- **Atomic commits**: one concern each (adapter, frame, tabs, sidebar, ...), the body names what omp-web contract it relies on
- **After an omp-web or overlay change**: run the overlay checks in a private check checkout, then the integrator restarts omp-web (it resets its worktree to `GIGACHAD_OMP_WEB_REF` and reapplies the overlay on start), runs `pytest tests/test_omp_web.py` and smokes an omp tab: open a session, switch omp tabs, send a prompt only in an approved throwaway session. Concurrent work: `.docs/integration-ompweb.md`
