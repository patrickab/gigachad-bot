# System Overview

## What runs where

Local-first workspace for chat, documents, architecture, canvases, and coding agents.
The browser uses server-issued paths. FastAPI owns persistence, filesystem authority,
and external-service access.

| Component | Responsibility | Development address |
| --- | --- | --- |
| Next.js / React | Workspace UI | `localhost:2999` |
| FastAPI | API, storage, identity, filesystem policy | `127.0.0.1:8001` |
| LikeC4 | Parse and edit architecture models | `127.0.0.1:8011` |
| omp-web | OMP sessions and execution | `127.0.0.1:30177` |
| Factory engine | Agent workflows in task worktrees | `127.0.0.1:8090` |

| Runtime | Contract |
| --- | --- |
| Development | Let the user start `./run.sh` or individual launchers. Agents must not start dev servers, except factory-managed previews. |
| Desktop | Tauri hosts an exported frontend and PyInstaller FastAPI sidecar. A readiness handshake supplies its ephemeral loopback API address. Tauri has no separate filesystem authority. |
| Production | Keep backend services private. Follow [deployment](deployment.md) for identity, units, ports, and limitations. |
| Remote inference | Configure external services without code changes. Follow [remote inference](remote-inference.md). |

## Source map and ownership

| Location | Responsibility |
| --- | --- |
| `src/backend/server.py`, `src/backend/routes/` | App setup, HTTP/WebSocket boundaries, dependencies |
| `src/lib/` | Storage, filesystem policy, domain services, upstream adapters |
| `src/config.py` | Backend defaults and runtime configuration |
| `src/frontend/app/`, `contexts/` | App shell, workspace tabs, shared state |
| `src/frontend/features/` | Feature UI and public cross-feature entry points |
| `src/frontend/lib/backend/`, `lib/domain/` | API access and domain contracts |
| `src/frontend/components/` | Shared UI and content rendering |
| `src/frontend/packages/chat-ui/` | Shared local/OMP chat rendering, palette, embed protocol |
| `src/c4/` | Stateless LikeC4 parsing and editing |
| `run*.sh`, `deploy/`, `factory.yaml` | Launchers, production operations, task execution |
| `tests/`, `src/frontend/**/__tests__/` | Backend/frontend contract and regression tests |

- Keep routes at the boundary. Reuse domain services.
- Import other features through public entry points, not internals.
- Preserve dependency boundaries enforced by `src/frontend/eslint.config.mjs`.
- Keep UI conventions in `AGENTS.md`. Reuse shared primitives and preserve accessibility.

## Persistence and security

| Data | Authority |
| --- | --- |
| Chats, uploads, prompts, project documents, memory, database-native artifacts | PostgreSQL through `*_store.py` / `PostgresDataStore` |
| PDFs and MinerU inputs/outputs | Configured filesystem mirror, commonly Nextcloud |
| Mounted files | Original disk files, optionally referenced by projects and chat attachments |

| Boundary | Required behavior |
| --- | --- |
| Host-access identity | Allow OMP, factory, Neovim, and files outside mounts |
| Other identities | Restrict file access to their mounted roots |
| Git/tree/worktrees | Remain mount-scoped, except explicit factory worktree admission |
| File APIs | Deny credential stores through `lib/file_roots.py`: `.ssh`, `.gnupg`, `.secrets`, `.env*`, app runtime `env`, OMP `agent.db` / `models.yml` |
| Production ingress | Trust private Tailscale Serve's `gigachad.local/cap/host`, never browser-supplied identity headers. Bind loopback and enforce exact origins. |

## External services

| Service | Use |
| --- | --- |
| LiteLLM | Common cloud/local chat and completion interface |
| Brave LLM Context | `web_search`: retrieve passages with backend-only `BRAVE_API_KEY`, then generate citation-grounded answers with the selected model |
| GPT-Researcher | `deep_research` through its DuckDuckGo retriever |
| MinerU | PDF extraction/OCR queue for document previews and Study generation. Desktop bundles only the HTTP client and requires an external server. |
| Agents-in-a-box | Sandboxed code execution and the separate factory engine |

- Disable capabilities whose optional service is unavailable. Do not replace it locally.

## Chat context

- Attach mounted files, project/global documents, architecture, or drawings through the composer.
- Toggle attachments in/out of model context. Open them to preview or edit.
- Create documents within their project. Preserve web-search source references.
- Stream through `/api/chat` SSE. Branch and merge conversations as needed.
- Keep project membership tab-local. Switching projects elsewhere must not change a chat's memory scope.

### Attachment lifecycle

Implement live references in `features/chat/attachments.ts`:

| Event / type | Contract |
| --- | --- |
| Active document reference | Store `Attachment.vaultPath`. Re-read on send and chat open, deduplicating reads per source. Skip inactive references until enabled. |
| Preview opening | Refresh the source before a text editor can modify it. Retry previously missing files. |
| Moved file | Use `FileRoots.relocate` only when the original mount contains exactly one matching filename. |
| Missing / ambiguous file | Keep last text as a read-only snapshot. Avoid repeated missing-file reads on send. Preserve state on other read failures. |
| Text / task-list edit | Save through the source's normal write path before updating chat state. Never text-save PDFs or images. |
| Referenced PDF | Sync into the shared library. Changed bytes invalidate OCR, concurrent attachments share extraction, obsolete results are discarded. Keep the previous extraction until replacement completes, so the first send may use old text. Resolve legacy `PDFs/<name>` references. |
| Pasted image, drawing, canvas image, architecture | Keep a snapshot. Reattach to include later changes. |

## Shared frontend contracts

| Concern | Owner / behavior |
| --- | --- |
| Chat rendering | `packages/chat-ui`: Q&A turns, Markdown/LaTeX, source-preserving copy, shared local/OMP role palette |
| Theme | Shell and fork map the palette to their tokens. Shell persists the preference and propagates it over the OMP bridge. |
| Resource deletion | `UndoDeleteContext` defers the request during undo, commits the previous deletion when another is scheduled, and restores on refusal. |
| Chat deletion | Remove pairs/attachments from model context immediately, but allow restoration. |
| Canvas undo | Keep independent editing history. |

## Architecture workspaces

### Use and storage

- Open a project's architecture. Edit the diagram or `.c4` sources.
- Create systems, modules, and saved views. Distinguish view-only removal from model deletion.
- Use source text for unsupported visual edits and compound descriptions.
- Keep filter/style-based views read-only on the canvas.

| Concern | Contract |
| --- | --- |
| Workspace | One merged LikeC4 model per project |
| Source location | Use linked mount `docs/architecture/` when present, otherwise PostgreSQL `graph/<slug>/`. Mount folder name matches project slug. Worktree switching does not change the linked clone's source. |
| File discovery | `lib/architecture_workspace.py` abstracts disk/database. Git directories include tracked and nonignored untracked files, nongit directories include all files. Never follow symlinks. |
| Source organization | Merge every `.c4` file. Classify by declarations, not filenames. Systems declare elements/views, modules extend systems, saved views use `views/<id>.c4`. |
| Relations | Store once with the source element. Display in every view containing both endpoints. |
| Layout | Persist `.likec4/<viewId>.likec4.snap`. Fall back to automatic layout for invalid snapshots. Preserve positions on rename/addition and explicit pins when moving parents. |
| Chat | Attach path-labelled source snapshots, or individual sources through the document picker. |

### Edit pipeline

```text
diagram / source edit
  -> features/editor/architecture/ (serialize reads and writes)
  -> routes/architecture.py (resolve storage)
  -> lib/c4_service.py -> src/c4/ POST /model or /apply
  -> syntax-tree text edits -> reparse -> revision check -> transactional write
```

- Reject invalid results with file/line diagnostics.
- Support element, relation, file, view, layout, and dangling-reference operations.
- Reread and revalidate on revision conflict. New files and deletions are not revision-checked.
- Prune references to removed elements after deletes/hand edits. Reject remaining invalid references.
- Prevent stale frontend reads from overwriting edits.
- Render with React Flow, route with libavoid, and fall back when WebAssembly is unavailable.
- Highlight sources with LikeC4's TextMate grammar.
- Keep layout fitting display-only until a move or resize persists it.

## Provider usage

| Concern | Contract |
| --- | --- |
| API | `GET /api/provider-usage` through `lib/provider_usage.py` |
| Source / cache | Read `omp usage --json`, cache account limits/reset times for five minutes |
| Forced refresh | `?refresh=true` invalidates OMP's cache first |
| Failure | Return 502 when OMP is missing or fails |
| Display | Summarize five-hour usage, expose other provider windows. Never substitute weekly values for missing five-hour limits. |

## File mounts

- Mount backend directories through **Add file mount**.
- Apply content policy even when the directory browser can list host directory names.
- Keep policy in `routes/file_roots.py` and `lib/file_roots.py`.

| Operation | Behavior |
| --- | --- |
| Browse tree | Skip `.gitignore` matches and internal directories such as `.git`, `node_modules`, `.obsidian`, `.trash` |
| Open file | Use specialized PDF/canvas/LikeC4/plot/markmap viewers, otherwise the text editor |
| Non-UTF-8 text request | Return 415 rather than risk lossy binary overwrite |
| Project file reference | Write edits to disk. Removing the reference leaves the source intact. Sync PDFs into the extraction library. |

## Diff viewer

- Open a mount's diff to compare working-tree changes with `HEAD`.
- Navigate files and choose unified or split layout.

| Action | Behavior |
| --- | --- |
| Comment | Select code within one file and split-view side, then use the comment action. Save selected text/context per worktree across reloads, attached after the last selected line. |
| Copy review | Export ordinary diff notes without clearing drafts |
| Send review | Submit factory notes in `human_review` or `blocked`. Include selected code and current locations/context. Keep original locations and mark outdated when anchors are missing/ambiguous. |
| Send result | Preserve notes on failure. Clear only submitted, unchanged notes on success. Show sent reviews in task messages, without inline reply threads. |
| Edit diff | Open read-only `HEAD` beside the working file in Neovim. Save with `:w`, return with `:qa`. Refuse deleted, binary, oversized, or symlinked files. |

Implementation: `features/editor/diff/` calls `/api/fileroots/diff/files`, `/diff`,
and `/diff/content`, backed by `routes/file_roots.py` / `lib/git_diff.py`.
Polling reloads changed patches by stamp.

## Worktrees

Use `/api/fileroots/worktrees` (`lib/git_worktree.py`) for mounted git clones:

| Action | Contract |
| --- | --- |
| Switch | Change mount tree, diff, and editor without renaming the mount. Persist selection per mount. |
| Find default | Infer `origin/HEAD`, otherwise main checkout's branch. Reuse/create its checkout without pulling or switching the main checkout. |
| Create | Default base to `dev`, otherwise remote default. Fetch and fast-forward the checked-out base, then create `~/git/worktrees/<repo>/<name>`. Refuse existing folders/branches. |
| Remove | Require explicit force for uncommitted changes. Delete branches separately, confirm unmerged deletion, never delete the remote branch. |

## omp tabs (omp-web)

Embed `patrickab/ompweb` branch `custom`. `run-omp-web.sh` serves a detached
worktree following the configured ref and reuses existing listeners.
Restart after updates and allow the frontend's frame origin.
Use [deployment](deployment.md#update-omp-web) for operations.

### Ownership and boundaries

| Owner | Contract |
| --- | --- |
| `lib/omp_web.py` | Own all upstream API URLs and Pydantic response contracts |
| `/api/omp/*` | Map adapter errors: 502 with field name for changed shapes, 503 for unavailable upstream. Never let the browser call upstream APIs directly. |
| Identity | Return frame URL and allow OMP routes only for host-access logins |
| `backend/omp_web_gate.py` | Production identity-checking proxy, with no omp-web API knowledge. Configure origin permissions separately. |
| `features/omp/` | Own frontend URLs and bridge knowledge |
| `OmpProvider` | Keep one iframe. Deep-link the initial session, switch by postMessage, never reload or move it. |
| Shell composer | Bridge prompts, images, documents, aborts, model/thinking changes, `/` and `@` completion. Merge returned drafts. |
| omp-web | Own execution, queueing, shell commands, session creation |

### Sessions and compatibility

- Group project-linked mount sessions under code, other workspaces under OMP.
- Navigate forks/in-session branches, restore archives, and open file links under shell file policy.
- Keep `src/frontend/packages/chat-ui/` canonical. Install its versioned tarball in the fork, never a sibling checkout.
- Bump package version, run `npm run sync:ompweb`, and commit fork tarball/manifest/lock together.
- Check synchronization with `npm run check:ompweb`.
- Bump `EMBED_PROTOCOL_VERSION` on every message change. Deploy matching sides.
- Preserve native standalone omp-web UI.

| Limitation | Effect |
| --- | --- |
| Shell composer | No word prediction or busy-run steering/queueing |
| Persisted OMP messages | Deletion unsupported |
| Frame DOCX/PDF previews | Blocked by `SAMEORIGIN` |

## Agent factory

The independent Agents-in-a-box engine runs each task in a git worktree and
supplies built-in agents/workflows. Gigachad provides boards, messages, diffs,
and previews.

### Run a task

- Open the repository's board and register it if needed.
- Choose a workflow and submit a request.
- Review phases, findings, gates, messages, and changes.
- Approve, retry, cancel, or send feedback as the state permits.
- Use native OMP `provider/model` overrides, not `litellm_proxy/` names.

### Configuration and repository contract

```text
built-in defaults -> user defaults -> repo factory.yaml -> workflow -> request
```

| Concern | Contract |
| --- | --- |
| Agents | Load built-ins, user defaults, or `.omp/agents/<name>.md`. Defaults affect new tasks only. |
| `factory.yaml` | Define sandbox profile, hooks, gates, and preview processes |
| `{repo_path}` | Resolve registered checkout. Reuse its `.venv` for gates/preview backend. Install frontend packages in task worktrees. |
| `.omp/agents/coder.md` | Supply repository conventions and scenarios |
| Protected files | Engine reverts task-agent changes to `factory.yaml`, `.factory/**`, `.omp/agents/**` |
| Task output | Keep phase JSON envelopes and scenario scripts in ignored `.factory/out/`, never commits |
| Startup | `run-factory.sh` starts/reuses engine, provisions missing checkout/config and `factory` database in selected dev/prod PostgreSQL, preserving existing state |
| Development opt-out | Set `GIGACHAD_FACTORY=0` |

### API and access

- Keep engine URLs/contracts in `lib/factory.py`.
- Configure the upstream with `GIGACHAD_FACTORY_URL`. See [factory deployment](deployment.md#provision-the-agent-factory) for the default.
- Host-gate `/api/factory`: repositories, defaults, workflows, tasks, actions, messages, events, preview restarts.
- Preserve engine refusals. Return 503 for unavailable upstream, 502 for invalid shapes.
- Share `lib/upstream.py` and route error mapping with OMP.
- Poll active boards only. Start at `/events/latest`, filter by repo, advance the engine-wide cursor only after successful refresh.
- Allow host logins to diff linked git worktrees without a mount. Retain file-viewer secret protection.

### Previews and scenarios

| Concern | Contract |
| --- | --- |
| Service startup | Engine alone starts ports 8100–8199 |
| Backend | Empty database and development identity |
| Frontend | `NEXT_PUBLIC_API_BASE` points to preview API |
| Scenario API | Read `preview.api_url` |
| Scenario result | Print final `{"open":"<path and query>"}` to select initial view |
| Remote production | Disable previews: remote browsers cannot reach host-loopback ports. Follow the deployment runbook. |

Encode one of these payloads as `?open=<base64url JSON>`:

```json
{"kind":"chat","history_file":"..."}
```

```json
{"kind":"document","path":"...","project_slug":"..."}
```

- Substitute `canvas` or `architecture` for `document` as needed. Project slug is optional.
- Validate and consume once in `app/_components/deepLink.ts`, removing the URL parameter.

## Neovim editor

- Enable per device in Settings when `/api/config` returns `neovim_available: true`: host access, `nvim` on backend `PATH`, and an existing `$XDG_CONFIG_HOME/nvim` directory (default `~/.config/nvim`).
- Use the built-in editor otherwise.

| Event / boundary | Contract |
| --- | --- |
| Session | `WS /api/neovim/session` through `routes/neovim.py` / `lib/neovim_session.py`: one `nvim -n` PTY per document, private temporary copy, xterm.js rendering |
| Working directory | Innermost mounted root, or home for database documents |
| `:w` | Send buffer over RPC through browser's normal write path, including architecture validation. Replace buffer with stored text if rewritten. Leave modified on failure. Never save directly to the temp copy as authoritative storage. |
| `:q`, document close, disconnect | Terminate process and remove copy. Unsaved edits are unrecoverable. |
| Preview / fullscreen switch | Preserve session |
| Terminal focus | Own all keys, including Escape |
| Security | Treat as host execution. Check WebSocket origin against `GIGACHAD_CORS_ORIGINS`, or loopback for dev `*`. CORS alone does not protect WebSockets. |

## Memory system

Document-backed, LLM-mediated memory, not a vector database.
Owners: `lib/memory_store.py`, `routes/memory.py`.
Global scope stores user facts, project scope stores project facts, each with configurable categories.

| Stage | Contract |
| --- | --- |
| Extract | `MEMORY_MODEL` proposes new/corrected facts since each chat's per-scope watermark. Run global/project extraction independently. |
| Review | Accept, edit, or reject buffered candidates. Do not change canonical memory yet. |
| Reconcile | Merge accepted candidates per affected category. Preserve untouched categories and preview changes. |
| Commit | Save canonical JSON/rendered Markdown, then advance `(chat_id, scope)` watermark. Cancel without advancing it. |
| Apply | `MemoryStore.augment_system_prompt` appends global and containing-project Markdown under `Persistent Memory Context`. |

- Edit categories, move memories between scopes, and remap orphaned entries through the app.
