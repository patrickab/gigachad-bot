# Pi replaces OMP and omp-web as the agent runtime

gigachad-bot runs coding agents, workbench agents, Agents-in-a-box tasks and subscription model completions on [Pi](https://github.com/earendil-works/pi) (`earendil-works/pi`), through one service of our own (`pi-host`). The agent UI is gigachad's own React, speaking Pi's documented RPC protocol (`pi --mode rpc`). OMP, `omp auth-gateway` and the embedded omp-web frame go away. Plan, contracts and phases: `.docs/pi-migration.md`.

## Why

- omp-web is customized through module overrides and a seam patch against its internals, so any upstream change can break the overlay. Pi publishes a typed, versioned protocol: a UI built on it depends on that contract only.
- Pi is the actively maintained upstream that OMP forks. Its RPC mode covers what the shell needs: prompt, steer, follow-up, abort, queue, slash commands, extension dialogs, sessions, branching, model and thinking control.
- One runtime for sessions and completions removes the second login store and model list (OMP and Pi).

## Considered options

- **Keep OMP and omp-web**: rejected. The overlay is the main maintenance cost and the main source of breakage.
- **Embed a Pi web UI (pi-web.dev, `agegr/pi-web`) like omp-web**: rejected. It moves the same overlay fragility to another upstream. pi-web.dev's plugin API is versioned, but its styling contract is experimental and it documents no embedding.
- **Pi SDK sessions in-process**: rejected for sessions. One crashing extension would take every session down, and sandboxed (container) sessions need a process boundary anyway. The SDK-level library `pi-ai` is used for the completion gateway.
- **Keep `omp auth-gateway` for chat completions**: rejected by the user to drop OMP entirely. The gateway is a thin OpenAI-compatible layer over `pi-ai`, which owns provider shaping.

## Consequences

- **`pi-host` is a separate Bun service.** It supervises `pi --mode rpc` child processes (local, or inside an Agents-in-a-box container) and serves the completion gateway. It outlives backend reloads, so gigachad-bot can be edited from a session running inside gigachad-bot. Restarting `pi-host` ends running turns, it waits for idle sessions.
- **Pi's protocol is the only agent protocol.** `pi-host` forwards Pi's records with a small supervisor envelope. Types live once in `src/frontend/features/pi/types.ts`. The postMessage embed protocol goes away.
- **Adapters stay one per side.** Backend: `src/lib/pi_host.py` only. Frontend: `src/frontend/features/pi/` only. The browser never reaches `pi-host` directly.
- **Normal chats keep the completion API** (LiteLLM, curated tools, no harness). Only their upstream moves from `omp auth-gateway` to the `pi-host` gateway.
- **Ported omp-web code**: the tool-call rows (`src/frontend/features/pi/tools/`) and file-link rules (`src/frontend/lib/fileLinks.ts`) are ported from MIT omp-web with attribution. The "never vendor omp-web code" rule ends with the omp-web integration.
- **Old OMP sessions are left in place**, reachable through the `omp` CLI. Nothing is converted.
- **Pi is pinned.** `pi-host` checks `pi --version` at start; upgrades are deliberate.
