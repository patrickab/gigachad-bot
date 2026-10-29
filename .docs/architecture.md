Generated from 9f773d5 on 2026-10-27

# Architecture

Use [system overview](system-overview.md) for concepts/workflows and
[module map](module-map.md) for source ownership. Find task-specific operational
guides in the [repository documentation index](../AGENTS.md#documentation).

## Runtime boundaries

```text
Next.js / React workspace
  | HTTP + SSE + WebSocket
  v
FastAPI: identity + scoped domain services
  |              |                |
PostgreSQL   mounted disk   upstream adapters
  |                         /      |       \
change log + SSE        LikeC4  omp-web   factory
```

| Component | Authority / lifecycle |
| --- | --- |
| Frontend | Workspace tabs, feature state, API clients. Use server-issued paths, never browser filesystem authority. |
| FastAPI | Identity, per-user storage, filesystem access, integrations. Keep routes at HTTP/WebSocket boundaries and reuse domain services. Store and file IO blocks, so a handler that calls it is a plain `def` (FastAPI's threadpool) or moves the call into `asyncio.to_thread`; a blocking call inside `async def` stalls every request and SSE stream (`tests/test_route_concurrency.py`). |
| Startup | Configure the OMP LiteLLM proxy before model calls, apply database migrations, start extraction and change brokers. Shutdown closes clients/pool and stops workers/MinerU children. |
| LikeC4 | Stateless model parsing and syntax-tree operations. Backend launcher starts/stops it and waits for health before FastAPI. Architecture routes return 503 if unavailable, but managed launch fails when startup health fails and tears down/restarts both processes if LikeC4 exits. |
| Development | User-controlled launchers. Agents must not start servers, except through engine-managed factory previews. |

| Listener | Development | Production |
| --- | --- | --- |
| Frontend | `localhost:2999` | Private frontend origin or existing Vercel deployment |
| FastAPI | `127.0.0.1:8001` | `127.0.0.1:8002` |
| LikeC4 | `127.0.0.1:8011` | `127.0.0.1:8012` |
| omp-web | `127.0.0.1:30177` | Same upstream behind identity gate |
| Factory | `127.0.0.1:8090` | Same default, separate production configuration/database |

## Frontend read recovery

- `contexts/StartupReadContext.tsx` bounds initial settings, models, prompts, projects, histories, sidebar listings, reasoning levels, and OMP overview reads to 15 seconds.
- Report failures in a shared banner. Retry only failed reads without remounting the workspace or clearing drafts and loaded data.
- Abort timed-out or disposed requests and ignore obsolete responses. Do not add automatic retries to these startup reads.
- Keep streams, uploads, writes, and explicit post-mutation refreshes unchanged. OMP retains focus and interval polling through the bounded read, only for host-access logins.
- `ProjectProvider` keeps the project list's identity when a refresh returns it unchanged, so startup refreshes (change-feed replay, first branch metadata) do not re-run reads keyed on it, such as the sidebar's per-project document lists.

## Identity and filesystem policy

- Keep production listeners on loopback behind trusted private Tailscale Serve. Prefer one private HTTPS origin for frontend and `/api`.
- Resolve `Tailscale-User-Login` to a database user. Serve strips client-supplied identity/capability headers. Directly exposing the backend would break this trust boundary.
- Grant host execution through `gigachad.local/cap/host`, carried in `Tailscale-App-Capabilities`. Missing/malformed grants do not grant host access.
- Development identity requires `GIGACHAD_ENV=development` and `GIGACHAD_DEV_USER`. That local login has host access.
- Treat `X-Device-Id` as a UUID for write attribution/echo suppression, never authorization. SSE may omit it.
- Resolve identity with one read per request. `resolve_user` writes only for a new login, a new or reassigned device, or a device last seen over `DEVICE_SEEN_INTERVAL` (one minute) ago, so concurrent requests never queue on the users row lock or a commit.
- Reject wildcard/empty production `GIGACHAD_CORS_ORIGINS`. Explicitly validate WebSocket origins too: CORS does not protect sockets.

| Operation | Policy |
| --- | --- |
| OMP, factory, Neovim, mounting/cloning roots, directory browser, redeploy | Require host access. A mounted root is readable in full, so mounting is a host action. |
| Ordinary file access | Restrict to that user's registered mounted roots and admitted linked worktrees |
| Host-access file content | Allow outside mounts only under the service user's home directory, retaining credential-store protection. Unreadable files return 403, not 500. |
| Git/tree/worktree endpoints | Require a mounted/admitted root. Host-gated factory diff explicitly admits linked git worktrees without mounting the repo. |
| Credential files | Outside mounted roots, host access rejects `.ssh`, `.gnupg`, `.secrets`, `.aws`, `.azure`, `.kube`, `.docker`, `.password-store`, `.pki`, `.mozilla`, `.config/{gh,gcloud,google-chrome,chromium,BraveSoftware,op,rclone}`, `.local/share/keyrings`, token/history files (`.netrc`, `.git-credentials`, `.pgpass`, `.npmrc`, `.pypirc`, shell/psql/python history), `.env`, `.env.*`, app runtime `gigachad-bot/env`, and OMP `agent.db` / `models.yml`. Explicitly mounted roots remain admitted, so this is not a blanket denylist inside mounts. |
| Directory browser | Lists host directory names for host-access logins. Listing is not permission to read/write content. |
| File trees | Honor layered `.gitignore` rules and skip `.git`, `node_modules`, `.obsidian`, `.trash`. Explicit content paths still use access validation, not tree visibility. |
| Text decoding | Return 415 for non-UTF-8 content instead of lossy binary overwrite |

## Persistence and synchronization

| Data | Authoritative representation |
| --- | --- |
| Chats, prompts, project documents, memory, model configuration, sandbox records | User-scoped PostgreSQL logical namespaces through domain stores / `PostgresDataStore` |
| Uploads, PDF library, extracted Markdown/images | PostgreSQL asset records. PDF/MinerU files also have a convenience filesystem mirror, never read back as canonical data. |
| Mounted file contents | Original disk files. Projects/attachments reference them rather than creating independently editable copies. |
| Architecture | Linked clone `docs/architecture/` when present, otherwise PostgreSQL `graph/<slug>/` |
| Diff notes/layout, per-device preferences, archived Code directories | Browser-local state, not shared database artifacts |
| PDF annotations | Canvas document `pdf-annotation/<sha256>.canvas`, keyed by the PDF bytes' SHA-256. `POST /api/fileviewer/annotations` creates it empty on first use. Every copy of the same bytes shares annotations, renames keep them, changed bytes start a fresh canvas. Strokes sync through the collaborative canvas path. |

- Keep logical namespaces independent of the physical Documents layout. Validate storage keys against absolute paths and traversal.
- Scope `PostgresDataStore`, assets, projects, file-root registry, and other domain dependencies to authenticated users.
- Check expected revisions on conditional writes. PostgreSQL `apply(writes, deletes)` is atomic: one stale revision aborts the whole batch, deletes included.
- Read several documents with `DataStore.read_many` (one query) instead of a read per key: project listings and prompt libraries depend on it. Content columns (`documents`, `assets`) are fetched in binary result format, half the bytes of hex text, and stored uncompressed out of line (`STORAGE EXTERNAL`, schema step `0011`), so a byte range fetches only the TOAST chunks it covers. Values written before `0011` keep their compression until rewritten.
- `documents.meta` (jsonb) holds chat tree metadata (`chat_id`, `parent_id`, `branch_message_idx`, `children`, `qa_count`). `ChatStore` writes it in the same statement as the content, stamped with the row version, every other write clears it. Branch metadata, the chat ID index and project cleanup read it with one `DataStore.list_meta` query, never chat bodies. Meta that is missing or stamped with an older version (code predating `0011`, project rename) is derived from the body and stored back by `write_meta`, guarded by the content hash, without a new revision or change row. Meta jsonb would not return unchanged (NUL, floats, unsorted object keys) stays NULL and is derived on every read.
- `documents.sha256` is a stored generated column (`sha256(content)`), so document hashes (PDF annotation keys) never read the content.
- Meta also carries `refs`, the keys an item references, for delete cleanup: a chat's attachments (each `vaultPath`, and `PDFs/<name>` for a PDF) written by `ChatStore`, a canvas's frame and attachment paths written by `CanvasMutationStore` with each accepted batch (other canvas writes leave it to be derived), a `project.json`'s `files` derived on read. Meta without `refs` (stored before it existed, or by older code after a rollback) counts as missing and is derived again, so no schema step backfills it. An annotation canvas's meta also keeps `hosts`, the host files it was opened with, which cannot be derived from content (see Delete cleanup).
- Keep development/production PostgreSQL containers and volumes separate. Their default Documents/PDF mirror is intentionally shared. Back up both database and mirror, not only code worktrees.
- Load user model defaults from persistent provider configuration. Seed shipped prompts only for an uninitialized collection, never overwrite an existing collection or restore deliberately deleted defaults. `get_prompt_store` checks each user's collection once per process, later prompt requests skip the check.
- Apply numbered `db/schema/` steps under a PostgreSQL advisory transaction lock. Record checksums in `applied_schema_steps`, reject modified applied steps, and apply pending steps in one transaction. Add migrations rather than rewriting applied history.

### Project rename and archive

- `ProjectStore.update_project` slugifies a new name. Same slug only renames, another slug runs `PostgresProjectRenamer` in one transaction; an existing destination raises `StorageConflictError`.
- The rename moves document and asset keys under `chat/`, `project/`, `graph/`, `memory/project/`, `attachment/project/`, rewrites slug/history fields and app-relative links inside JSON/`.md`/`.txt`/`.c4`/`.canvas`/`.snap` documents of app namespaces, and rewrites canvas mutation values. Prose, absolute disk paths and similarly named projects stay unchanged. Every move records change rows.
- A disk folder linked by the old slug's folder name stays linked through `linked_root` in `project.json`.
- `archived` is a catalog flag (`PATCH /api/projects/{slug}`) that hides the project from the sidebar. Archiving Code directories is per device (`localStorage` `archived_code_directories`). OMP session archive/restore goes through omp-web.

### Project file names

- `project.json` `files` lists each referenced key as a plain string, or as `{"path", "name"}` when the project shows it under a display name. `ProjectStore` reads both forms and writes the object form only while a display name is set.
- Renaming a project file (`POST /api/documents/rename` with a slug) only sets that project's display name, keeping the file's suffixes. Stored key, chat and canvas attachments, library PDF, mirror and annotation key stay untouched, and the same file can carry another name in another project. Renaming back to the file name drops the display name. A note (empty slug) has no catalog entry, so its key moves to the new name.
- `GET /api/documents` and `/all` return the display name as `name`, the project list returns each project's `file_names` (path to display name). The frontend resolves tabs, editor headers, canvas window titles and chat attachment chips through `useDisplayName` in `ProjectContext`, which every rename refreshes.
- Moving a canvas keeps the name it is shown under: into a project as display name, into the notes as its key.

### Change delivery and canvases

```text
committed user write -> durable changes row -> PostgreSQL NOTIFY
  -> process ChangeBroker -> user-scoped SSE -> resource-specific refetch
```

- Resume `/api/sync/stream?since=<seq>` from the applied sequence. Subscribe before replay and suppress overlap to avoid missing concurrent commits. Replay (`changes.latest_changes`) sends only the latest change per resource, since clients reload by resource.
- A fresh page connects without `since`: the stream subscribes, then sends `event: cursor` with the user's head sequence (`changes.head_cursor`) and only later changes, no replay. `syncStream.syncCursorReady()` resolves on the first event, and startup reads (`useStartupRead`), file editors and the architecture view await it before reading, so a change racing a read still arrives as an event. A first connection that fails or sends nothing within 5 seconds releases the reads and reconnects with `since=0`.
- `UserScope.record` deletes the rows its new row supersedes (same user, kind and key) in the same statement, skipping rows another transaction holds (`FOR UPDATE SKIP LOCKED`), so the table keeps about one row per resource, deletes included. Schema step 0013 pruned the backlog once and added `changes_resource_idx`. Replay from any cursor is unchanged, because it only ever returned each resource's latest row.
- Send metadata, not document contents. Bound subscriber queues, close lagging streams, and recover through durable replay.
- Keep `.canvas` snapshots in documents and accepted mutation batches in `canvas_mutations`.
- Serialize canvas writers with a document row lock. Apply snapshot/log/change-row updates transactionally, deduplicate mutation IDs, and replay accepted batches by revision.
- Keep canvas editing undo separate from general resource deletion.

## Chat, models, and tools

| Boundary | Contract |
| --- | --- |
| Chat stream | `/api/chat` SSE carries tokens, usage/tool events, completion, and failures. Keep proxy buffering disabled for streaming. |
| Project context | Bind memory scope to the chat's containing project/tab, not the project selected elsewhere. Support branch/merge histories: branch, merge, move and (cascade) delete commit all their chat and `project.json` writes and deletes in one unguarded `DataStore.apply` transaction, upload cleanup runs after it. |
| Providers | LiteLLM routes cloud/local calls, including the configured OMP model source. Use stored model defaults, not hard-coded production models. `lib/llm_resilience.py` retries a call once with `allowed_openai_params` when stale LiteLLM metadata rejects parameters. |
| Search | Brave LLM Context retrieves passages with backend-only `BRAVE_API_KEY`, then the selected model generates cited answers. |
| Research | GPT-Researcher uses its DuckDuckGo retriever for multi-step research. Serialize runs with `RESEARCH_ENV_LOCK` because model/reasoning configuration temporarily changes process-wide environment, restoring it and deleting temporary config afterward. |
| Sandboxed tools | Agents-in-a-box executes workspace code and image-based plot tasks. Distinguish this execution service from the external task factory. |
| Optional service failure | Report dependency unavailability for affected capabilities, do not silently substitute a local service. |

### Collapsed chat rendering

- `features/chat/ChatPair.tsx` defers collapsed Markdown and equation rendering until within 400px of the viewport or focused. Hidden tabs stay deferred until shown, while headers, tool labels, and screen-reader text remain available.
- Reserve the 12.5vh preview cap offscreen, allowing short answers to shrink when activated. Expanding a turn renders its full content and plots.
- Chats and omp sessions open at their end and hold it through late layout until reader input (`features/chat/useStickToBottom.ts`, overlay `embed/insets.ts`). Streaming follows while pinned.

### Sandbox checkpoint boundary

- Keep workspace-agent and plot slots scoped by user, chat, and sandbox scope. `SandboxRuntimeState` survives request-scoped service facades within one backend process and serializes each slot, not all chats. Uncheckpointed state is not shared across worker processes or recovered after process loss. Checkpoints and `delete_chat_state` run their store IO in a worker thread while holding the slot lock and `SandboxRuntimeState.assets_lock`, so asset garbage collection never sees a half-written checkpoint. Its asset deletes stay on the event loop, atomic with runs staging outputs there.
- Persist rich output assets immediately, but stage the latest workspace archive/manifest in memory. `routes/histories.py` checkpoints that workspace only after saving the corresponding chat successfully. Do not treat a finished tool call as a durable workspace checkpoint. `SandboxStore.checkpoint` writes the slot, manifest, active pointer and every run record in one `DataStore.apply` transaction; `recover_completed_run` only heals records written before that.
- Reuse completed `tool_call_id` results. Refuse a still-running record as busy, allow abandoned records older than five minutes to run again, and release native run/restore resources even on cancellation.

### Live attachments and extraction

- Store source references in `Attachment.vaultPath`. Re-read active references on send/chat open and refresh previews before editor writes. Deduplicate reads per source, defer inactive references.
- Relocate moved files only when the original mount contains exactly one matching filename. Missing/ambiguous references keep read-only last text.
- Skip repeated missing-file reads on send, retry on preview opening, and preserve state for other read failures.
- Save text/task-list edits through the original write path before updating chat state. Never text-save PDF/image attachments.
- Sync changed referenced PDF bytes into the shared library and invalidate cached extraction. Deduplicate queued/extracting content and discard superseded results.
- Keep previous extracted text while replacement runs. The first send after a file change may use old text. Continue resolving legacy `PDFs/<name>` references.
- Run one extraction worker per backend process to avoid competing parses within that process. The queue/pending set is process-local, not a cross-worker lock. Persist results to PostgreSQL and mirror PDF/Markdown/images as convenience copies. Mirror failures do not invalidate authoritative database writes.
- Key the per-user PDF library by filename (`PDFs/<name>`) and extraction by filename stem (`Mineru/<stem>.md`, `Mineru/images/<stem>/`). Reusing a filename replaces that library entry, not a second independent document.
- Keep pasted images, drawings, canvas images, and architecture source attachments as snapshots. Reattach for later changes.
- Use `MINERU_SERVER_URL` for external OCR. Without it, the backend can spawn local MinerU per parse.

### PDF viewer, annotations and assistant

- `features/editor/pdf/AnnotatedPdfViewer.tsx` draws pen/highlighter strokes anchored to pages (page units, margins allowed), reusing canvas ink, history and collaboration. Save failures show as unsaved. The PDF starts loading alongside the annotation key request, ink joins the same viewer once the key arrives.
- `GET /api/fileviewer/raw` sends `ETag` with `Cache-Control: private, no-cache`: a reopened, unchanged file answers 304 from metadata (asset `sha256`, document version and `updated_at`, host file stat). Stored files serve single byte ranges sliced in Postgres, so pdf.js fetches the pages it renders first. The CORS middleware exposes `Accept-Ranges` and `Content-Range` for split-origin deployments. `POST /api/fileviewer/annotations` hashes without loading the PDF: assets and documents both keep a stored `sha256` column.
- `PdfAssistant.tsx` chats about a PDF in the tab's file panel. Its system prompt embeds the MinerU Markdown, only for mounted (`/…`) and library (`PDFs/…`) PDFs, polled every 5 seconds while extraction runs. Selection questions add the selected passage and page. History lives only as long as the component, it is not a saved chat.

## Architecture workspaces

| Representation | Contract |
| --- | --- |
| Workspace | Merge all `.c4` files. Match linked mount folder name to project slug. Switching worktrees does not move the linked clone's architecture source. |
| Files | Classify declarations, not filenames. Systems declare elements/views, modules extend systems, saved views use `views/<id>.c4`. |
| Discovery | Git workspaces include tracked and nonignored untracked files, nongit directories include all files. Do not list symlinks. |
| Relations | Store once with the source element. Show in any view containing both endpoints. |
| Layout | Persist `.likec4/<viewId>.likec4.snap`. Invalid snapshots fall back to auto-layout. Preserve positions on rename/addition and explicit pins when moving parents. |
| Rendering | React Flow, libavoid with fallback if WebAssembly is unavailable, LikeC4 TextMate source grammar. Fit-to-view stays display-only until move/resize persistence. |

```text
diagram / source edit -> serialized frontend reads/writes -> architecture route
  -> workspace snapshot -> LikeC4 POST /model or /apply -> reparse
  -> revision-checked storage writes -> returned model
```

- Apply element/relation/file/view/layout/pruning operations as syntax-tree text edits, not a second model representation.
- Hand edits must parse cleanly. Operations refuse newly introduced errors. Prune references to removed elements, reject remaining invalid references with file/line diagnostics.
- Reread and revalidate operations on revision conflict. Return 412 when retries cannot resolve it.
- Revision-check changed, previously existing files only. New files, deletions, and unchanged reads are not protected against races.
- Keep PostgreSQL batches transactional. Disk-backed batches check all revisions first, then replace each file separately, not as one multi-file transaction. Delete removed sources/layouts afterward without revision checks.
- Serialize frontend reads/writes to prevent stale reads overwriting edits.
- Keep filter/style-based views canvas-read-only. Distinguish removal from one view from deletion from the model. Source editing covers unsupported visual operations and compound descriptions.
- Attach path-labelled `.c4` source snapshots to chat, or individual sources through the document picker.

## Git diffs and worktrees

- Manage mounted git clones through `/api/fileroots/worktrees` and its operation endpoints.
- `POST /api/fileroots/clone` clones an https/ssh link into `~/git/<repo>` (`git_worktree.clone_repository`, link validated by regex, `--` before the URL) and mounts it. It is not gated to mounted roots, since it creates one.
- Remote git operations run non-interactively (`GIT_TERMINAL_PROMPT=0`) with `ssh -F ~/.ssh/config` (or `/dev/null`) to skip `/etc/ssh`, which the service sandbox's user namespace makes unreadable. An explicit `GIT_SSH_COMMAND` wins.

| Concern | Contract |
| --- | --- |
| Diff API | `/api/fileroots/diff/files`, `/diff`, `/diff/content`: compare working tree to `HEAD`, poll changed patches by stamp |
| Layout | Unified/split/auto presentation over parsed patch lines |
| Review anchors | Same file and split side. Save selected text, original line identity, and nearby diff context per worktree in localStorage. Resolve by text and adjacent same-side context, not line number alone. |
| Export | Use current location/context when resolved. Missing/ambiguous matches stay outdated with original context. Copy retains notes. Failed sends retain notes, successful sends clear only submitted unchanged notes. |
| Factory review | Send only while task accepts feedback (`human_review` / `blocked`). Sent review is a task message, not an inline reply thread. |
| Native edit | Read-only `HEAD` plus editable working buffer in Neovim. Reject deleted, binary, non-UTF-8, oversized, symlinked, or escaping paths. |
| Mount selection | Persist selected worktree per mount. Switch tree/diff/editor without renaming the mount. |
| Default checkout | Infer `origin/HEAD`, otherwise main checkout's branch. Reuse/create checkout without pulling or switching the main checkout. |
| New worktree | Default base `dev`, otherwise remote default/main checkout fallback. Fetch and fast-forward checked-out base. Create `~/git/worktrees/<repo>/<name>`, refuse existing branch/folder. |
| Removal | Require explicit force for dirty/untracked files. Delete branches separately, confirm unmerged deletion, never delete remote branches. |

## omp tabs (omp-web)

- Serve unmodified upstream `kahme247/ompweb` plus the private overlay `ompweb-overlay/` (embed mode, bridge, composer, Q&A cards, Markdown, embed-only theme). Never vendor its components/hooks/CSS or change its server/session/protocol logic to suit the shell.
- Let `run-omp-web.sh` serve a detached worktree at `GIGACHAD_OMP_WEB_REF`, with the overlay as `custom/` and the two-line `seams.patch` that loads it. Overrides wrap upstream modules by relative path, overlay CSS is scoped to `html.cf-embed`. It reuses existing listeners, so updates require restart. Allow the frontend's frame origin.

| Owner | Contract |
| --- | --- |
| Backend adapter | Only `src/lib/omp_web.py` knows upstream URLs/response shapes. Pydantic validates responses, `/api/omp/*` maps errors. Browser-side shell code never calls upstream APIs directly. |
| Gate | `src/backend/omp_web_gate.py` checks host identity before proxying the production frame. Keep upstream API knowledge out. |
| Frontend adapter | Only `features/omp/` owns iframe URLs and postMessage details. `OmpProvider` keeps one frame, deep-links the initial session, switches by message without reload/DOM relocation. |
| Shell composer | Send prompts/images/documents, abort, model/thinking changes, `/` and `@` completion. Merge returned drafts. Lies over the full-height frame with the top bar like a chat, the `insets` message tells the overlay how far to pad the transcript (`embed/insets.ts`, `theme/embed.css`). |
| Frame dialogs | The overlay reports an open modal dialog (`modal`, `embed/modal.ts`). `OmpProvider` lays `FloatingWindow`'s backdrop over the whole shell and lifts the frame above it, omp-web's backdrop (the shell's scrim, `theme/tokens.css`) covers the frame. A click on it or Escape in the shell sends `close-modal`. |
| Extension select | An extension's `select` (e.g. `/files`) is reported as `select` (state message, request id keyed) and never reaches upstream's panel (overlay `embed/select.ts`). The active tab's `OmpSelect` picks in a top-anchored fuzzy finder (`lib/fuzzy.ts`), `select-answer` replies with the option or null for cancel. Closing it refocuses the composer, whose send had disabled the input. |
| Engine | omp-web owns execution, queueing, shell commands, session creation. Shell does not expose word prediction or busy-run steering/queueing. |
| Navigation | Group project-linked mount sessions under code, others under OMP. Navigate forks/branches, restore archives, route file links through shell file policy. An extension's `Open file: /abs/path` notify (e.g. `/files`, `~/.omp/agent/extensions/files.ts`) opens the path in the file panel instead of showing, overlay `overrides/hooks/useAgentSession.ts`. |

- Keep shared chat rendering/protocol/palette canonical in `packages/chat-ui`. Local and OMP chat share Q&A turns, Markdown/LaTeX, fenced code with Mermaid and Markmap diagrams (each host passes its own `light`/`dark` scheme), source-preserving copy, and role colors. An omp compaction anchors a Q&A card in active-context order: the compaction (what it saved, summary folded beneath, overlay `overrides/components/MessageView.tsx`) asks, the kept messages and the continued run answer.
- Map the palette to each side's tokens. Shell persists its theme preference and propagates it through the bridge.
- `run-omp-web.sh` packs `packages/chat-ui` and installs it into the served worktree on every start, so overlay and shell always share one commit's package. A changed package also drops webpack's build cache, which takes `node_modules` packages as immutable per version.
- Bump `EMBED_PROTOCOL_VERSION` on every message change. No runtime handshake detects mismatches, a running omp-web keeps the old bundle until restarted.
- Return 503 for unavailable upstream, 502 naming invalid response fields, and preserve explicit upstream refusals. Share HTTP/error plumbing with factory through `lib/upstream.py`.
- Gate frame URL/API availability by host access. Persisted OMP-message deletion is unsupported. Frame DOCX/PDF previews are blocked by `SAMEORIGIN`. Standalone omp-web keeps its native UI.

### OMP model source is a separate integration

- Route local model-chat OMP selections through LiteLLM's `litellm_proxy/` prefix to the auth gateway, not through the omp-web frame. Configure the proxy before first model use, preserving explicitly supplied LiteLLM environment values.
- Use `GIGACHAD_OMP_GATEWAY_URL` (default `http://127.0.0.1:4000/v1`). Resolve the bearer from `GIGACHAD_OMP_GATEWAY_TOKEN`, then OMP's token file, then the no-auth fallback. `PI_CONFIG_DIR` overrides OMP's normal config directory.
- Return an offline catalog with a reason when the gateway is unreachable. That catalog response is distinct from a failed generation and from omp-web's 503. Cache the catalog for 60 seconds and OMP CLI reasoning levels for five minutes.

## Agent factory

- Keep Agents-in-a-box independent: engine owns tasks, phases, worktrees, agents, and previews. Shell displays boards, messages, diffs, and preview access.
- Keep engine URLs/contracts in `src/lib/factory.py`. Configure `GIGACHAD_FACTORY_URL`, default `http://127.0.0.1:8090`.
- Host-gate `/api/factory` repository registration, defaults, workflows, tasks, commands, messages, events, and preview restarts. Preserve engine refusals, map unavailable to 503 and invalid shapes to 502.
- Poll active boards only. Start at `/events/latest`, filter by repo, and advance the engine-wide cursor only after successful refresh.

```text
configuration: built-ins -> user defaults -> repo factory.yaml -> workflow -> request
agents:        built-ins / user defaults / .omp/agents/<name>.md
```

| Concern | Contract |
| --- | --- |
| Defaults | Affect future tasks only. Model overrides use native OMP `provider/model`, not `litellm_proxy/` names. |
| Repository config | `factory.yaml` defines sandbox profile, hooks, gates, preview processes. `{repo_path}` resolves registered main checkout. Reuse its `.venv` for gates/backend, install frontend packages in worktrees. |
| Agent instructions | `.omp/agents/coder.md` supplies conventions and scenarios |
| Protected paths | Engine reverts task-agent changes to `factory.yaml`, `.factory/**`, `.omp/agents/**`. Put phase JSON/scenarios in ignored `.factory/out/`, never commits. |
| Startup | `run-factory.sh` provisions missing checkout/dependencies/config and `factory` database in selected dev/prod PostgreSQL, preserving state. `GIGACHAD_FACTORY=0` skips development startup. |
| Previews | Engine alone starts ports 8100–8199. Use empty database/development identity, frontend `NEXT_PUBLIC_API_BASE`, scenario API `preview.api_url`. Disable remote-production previews because host-loopback URLs are unreachable. |
| Scenario output | Final stdout `{"open":"<path and query>"}` selects initial view |

Encode one payload as `?open=<base64url JSON>`:

```json
{"kind":"chat","history_file":"..."}
```

```json
{"kind":"document","path":"...","project_slug":"..."}
```

- Substitute `canvas` or `architecture` for `document`. Project slug is optional.
- Validate and consume once in `app/_components/deepLink.ts`, removing the query parameter.

## Neovim sessions

- Consume `neovim_available` from `/api/config`, not a frontend reconstruction: require host access, `nvim` on backend `PATH`, and `$XDG_CONFIG_HOME/nvim` (default `~/.config/nvim`) directory. Settings opt-in is per device.
- Run one `nvim -n` PTY per document through `WS /api/neovim/session`, rendered by xterm.js. Edit a private temporary copy, start in the innermost containing mount or home for database documents.
- Route `:w` through RPC to the browser's normal document API, including architecture validation. Replace the buffer with stored text if rewritten. Leave modified on failure. The temporary copy is not canonical storage.
- Terminate/remove the copy on `:q`, document close, or disconnect. Unsaved edits are unrecoverable. Preserve the session across preview/fullscreen switches. Terminal focus owns all keys, including Escape.
- Yanks reach the browser clipboard: a `TextYankPost` autocmd notifies over RPC, the socket sends `{"type":"yank"}`, the browser writes it best effort.
- Treat as host code execution. Match WebSocket origins to `GIGACHAD_CORS_ORIGINS`, or loopback for development `*`.

## Memory and deletion semantics

| Memory stage | Contract |
| --- | --- |
| Scope | Document-backed JSON, not vector search. Atomic memories (`{id, text, topic, scope, ...}`) belong to topics (`{name, summary}`), the complete memories. A `MemoryScope(kind, slug)` names the global profile or one project. Global: `memory/global-topics.json` + `global-profile.json`, seeded with `profile` and `preferences`. Project: `memory/project/<slug>/memory/topics.json` + `memory.json`, no seed topics. Schema step `0010_memory_topics.sql` migrated the former category files and fields. Persistence lives in `memory_store.py`, LLM extraction/regroup in `memory_extract.py` (`MemoryExtractor`), prompt injection in `memory_context.py`. |
| Extract | Manual (`/memorize*`), using the stored `memory_model`. One call per scope after its `(chat_id, scope)` watermark returns operations (`add`, `update`/`delete` by short fact id) plus summaries for new or outgrown topics. `build_proposal` drops unknown ids, repeated targets, duplicates and no-ops, and gives every new topic a proposal. |
| Review | One screen (`MemoryPanel`): accept, edit, move or deny each operation, rename or summarize new topics (renaming onto an existing topic merges), accept or deny summary revisions. |
| Commit / cancel | `POST /api/memory/commit` applies accepted operations deterministically (stale targets skipped), upserts topics, then advances that scope's watermark, also when every operation was denied. Topics, memories, watermark and pending buffer are written in one `DataStore.apply` transaction. Cancellation discards the review without advancing. |
| Injection | `memory_context.augment_system_prompt` renders topics with summaries under `Persistent Memory Context`. Pinned global topics always render in full, other topics fill `PROMPT_CHAR_BUDGET` newest first, the rest appear as a summary index. |
| Maintenance | One memory window (sidebar brain button, `features/memory/MemoryViewer.tsx`) lists global and per-project topics. Every edit writes immediately (`PUT /api/memory/topics`, then `PUT /api/memory/memories`), queued per scope. Facts move between topics, a title naming another topic merges into it after confirmation. `POST /api/memory/regroup` proposes subject topics for a scope, applied only after review. Deleting a topic deletes its memories, undoable via `UndoDeleteContext`. |

- Use `UndoDeleteContext` for deferred resource deletions. Scheduling the next deletion commits the previous one, refusal restores it.
- Remove chat pairs/attachments from model context immediately while allowing undo. Undo restores by message identity: a pair goes back before the message that followed it (at the end once that is gone), an attachment only into the message it was removed from. Keep canvas history independent.

### Delete cleanup

`lib/references.py` is the one reference model. Every delete path records what it let go of in a `Released` (`owned` keys it deletes itself, released `refs`, deleted `chats`), commits its own delete, and only then calls `References.release`, so a failed delete never loses shared data.

| Delete | Removes | Unless |
| --- | --- | --- |
| Chat, cascade, merge child (`ChatCleanup.after`) | Its uploads and sidecars, sandbox state, then the library PDF and annotation canvas of each PDF it referenced | Another chat, project, held canvas or upload still references the PDF name (library) or the bytes (annotation) |
| Project (`ProjectStore.delete_project`) | Its chats as above, its stored documents `project/<slug>/document/...` (canvases release what they show) and what its files derived | Another project lists the document, a chat or held canvas references it |
| Project file (`DELETE /api/documents`) | The reference, then the file and its derived data | As for a project. A note (empty slug) is deleted outright, its derived data released |
| One chat upload (`DELETE /api/files/chat/...`) | The upload and its sidecar, then its library copy and annotations | As for a chat |
| Canvas | Its mutation log and share rows (foreign keys `ON DELETE CASCADE`), annotation canvases included | |

- Held canvases: every `note/` canvas, and project canvases a chat, a project's `files` or another held canvas reaches. Annotation canvases count PDF bytes: stored uploads, library PDFs and document PDFs, plus host PDFs a chat, project or held canvas references, hashed from disk.
- Annotation canvases of host PDFs opened from the file tree: `POST /api/fileviewer/annotations` records each resolved host path in the canvas's meta `hosts` (`annotation_canvas`, kept by `CanvasMutationStore` across batches). The canvas stays while any recorded file still exists with its hash. Cleanup stats a recorded file and hashes it only when nothing stored keeps the canvas.
- MinerU output (`Mineru/<stem>.md`, `Mineru/images/<stem>/...`) is never deleted by a release nor listed by the dry run.
- Host files are references only: hashed, never written or deleted. The PDF/MinerU convenience mirror on disk is left alone.
- Cleanup failures after the commit are logged, not raised. Nothing purges older orphans automatically: `python -m lib.references --dry-run [--database-url URL]` (from `src/`) lists them per user with sizes, plus uploads of chats that are not stored. Its reads may store missing meta, never content.
- omp sessions: gigachad-bot stores nothing per session. The omp composer's uploads are keyed by the shell tab's chat ID and show up in the dry-run report once no stored chat has that ID.

## Provider usage

- Serve `GET /api/provider-usage` from `omp usage --json` through `lib/provider_usage.py`. Cache for five minutes, invalidate OMP's cache first for `?refresh=true`.
- Return 502 for missing/failing OMP. Reuse last reports for dropped providers for up to an hour, adjusting reset times. Report unavailable accounts without retained reports explicitly.
- Summarize five-hour limits and expose other provider windows. Never substitute weekly values for missing five-hour limits.

## Production redeploy

- `POST /api/deploy/redeploy` (host access, settings menu) starts `redeploy.sh` as transient user unit `gigachad-bot-redeploy` via `systemd-run`, so restarting the backend does not kill it. Returns 202, or 409 with stderr while one runs or when the start fails.
- The script rebases the production worktree onto `origin/master`, refusing tracked changes, and restarts `gigachad-bot.service` from a clean environment. It then restarts omp-web when its ref, `ompweb-overlay/` or `packages/chat-ui` changed, never while an OMP session runs, and restarts its gate. Steps and rollback: [deployment](deployment.md).
