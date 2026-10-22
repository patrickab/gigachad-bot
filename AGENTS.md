## Never start the dev server autonomously

The backend (`uvicorn`, port 8001) and frontend (`next dev`, port 2999) are
started via `run.sh` or manually by the user, in a terminal they control.

- Never run `run.sh`, `uvicorn`, `npm run dev`, or equivalent yourself —
  even in the background, even to verify a change.
- If you need to see the app running (UI check, manual test, screenshot),
  ask the user to start it and confirm it's up, then use their already-running
  instance. Do not launch your own copy alongside theirs.
- Exception: the agent factory engine starts previews of task worktrees (backend
  and frontend on ports 8100-8199, `.docs/architecture.md`, "Agent factory").
  That is the engine's job, you still never start one yourself.

## Documentation

- Follow the global documentation standard. Keep generated context complementary: system overview for concepts/workflows, architecture for implementation contracts, module map for source ownership and navigation.
- Regenerate `.docs/system-overview.md`, `.docs/architecture.md`, and `.docs/module-map.md` through `/repomix`. Preserve operational details in maintained runbooks and load them only for relevant tasks.
- `.docs/deployment.md` — Private production deployment, identity, service configuration, verification, and rollback.
- `.docs/remote-inference.md`: Remote OCR/Ollama setup, SSH tunnels, and environment configuration.

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

omp tabs embed omp-web from the fork `patrickab/ompweb` (branch `custom`), itself a fork of `kahme247/ompweb`. Both repos must keep rebasing onto upstream without conflicts, so gigachad-bot adapts to omp-web, never the reverse. Architecture: `.docs/architecture.md`, "omp tabs (omp-web)".

- **Minimal delta upstream**: never ask for or make changes to omp-web's server API, session handling or protocol logic. Read what it already serves. A feature omp-web lacks is built here, or added in the fork's `custom/` following its `custom/AGENTS.md`
- **Drop before you change**: omp-web UI gigachad-bot does not want is hidden in the fork's embed mode (`custom/theme/embed.css`), not rewritten. Prefer hiding a feature over porting or patching it
- **Adapters own all omp-web knowledge**, one place per side, so an omp-web change lands in one file:
  - Backend: `src/lib/omp_web.py` only. Routes (`routes/omp.py`) map its errors, they never build omp-web URLs. Pydantic contract models validate every response, a shape change fails loudly with the field name (502). `src/backend/omp_web_gate.py` (production Tailnet gate in front of the frame) is a plain identity-checking proxy, keep omp-web API knowledge out of it
  - Frontend: `src/frontend/features/omp/` only. Nothing else knows the iframe, the postMessage protocol (defined in the shared package, eslint lets only this feature import it) or omp-web URLs. Other features import its public files (eslint `crossFeature`)
  - The browser never calls omp-web's API, the backend does, server to server
- **Never vendor omp-web code** (components, hooks, CSS). Write new code against its API with omp-web as reference
- **Shared contracts live once, in `src/frontend/packages/chat-ui/`** (`@gigachad/chat-ui`): the postMessage protocol (`src/protocol.ts`, bump `EMBED_PROTOCOL_VERSION` with every message change) and the role palette (`src/palette.css`). The shell imports the source, the fork the vendored tarball. Ship a change with `npm run sync:ompweb` (bump the package version first) and commit the fork's tarball, manifest and lock
- **One shared iframe**: tabs switch it by message, never by reload or by moving it in the DOM. Do not add a second frame or a per-tab frame
- **Atomic commits**: one concern each (adapter, frame, tabs, sidebar, ...), the body names what omp-web contract it relies on
- **After an omp-web update**: commit it to the fork's `custom` branch and restart omp-web (its worktree follows `custom` on start), run `pytest tests/test_omp_web.py`, then smoke an omp tab: open a session, switch omp tabs, send a prompt

