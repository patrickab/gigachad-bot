Generated from bd63559a on 2026-11-25. Corrected by hand on 2026-11-27 after the complexity cleanup (removed files and names dropped, new files added; no `/repomix` run). Edited by hand on 2026-11-27 after the reuse pass (no `/repomix` run).
Reduced by hand on 2026-11-25. Conversations section, the Pi tool rows and the backlog edited by hand from 78140998 on 2026-11-26 (no `/repomix` run).

# Architecture

Implementation contracts. Concepts and workflows: [system overview](system-overview.md). Source ownership: [module map](module-map.md). Task guides: [documentation index](../AGENTS.md#documentation).

## Runtime boundaries

```text
Next.js / React workspace
  | HTTP + SSE + WebSocket
  v
FastAPI: identity + scoped domain services
  |              |                |
PostgreSQL   mounted disk   upstream adapters
  |                         /      \
change log + SSE        LikeC4   pi-host
```

| Component | Authority / lifecycle |
| --- | --- |
| Frontend | Workspace tabs, feature state, API clients. Uses server-issued paths, never browser filesystem authority. |
| FastAPI | Identity, per-user storage, filesystem access, integrations. Routes stay at the HTTP/WebSocket boundary and reuse domain services. Store and file IO blocks: a handler calling it is a plain `def` or uses `asyncio.to_thread`; a blocking call in `async def` stalls every request and SSE stream (`tests/test_route_concurrency.py`). |
| Startup | Reap orphaned host processes, sync Pi's `ollama` provider in an executor thread, apply migrations, start extraction and change brokers. `openai` and GPT-Researcher load lazily on first use; a module-level import moves their cost into startup (`python -X importtime -c "import backend.server"` shows it). Shutdown closes clients/pool, stops workers, MinerU children, the Jupyter Server and owned host processes. |
| LikeC4 | Stateless model parsing and syntax-tree operations. The launcher starts it and waits for health before FastAPI; failed startup health fails the launch, a LikeC4 exit tears down/restarts both. Architecture routes return 503 while unavailable. |

| Listener | Development | Production |
| --- | --- | --- |
| Frontend | `localhost:2999` | Private frontend origin or Vercel |
| FastAPI | `127.0.0.1:8001` | `127.0.0.1:8002` |
| LikeC4 | `127.0.0.1:8011` | `127.0.0.1:8012` |
| pi-host | `127.0.0.1:8300` | `127.0.0.1:8301`, own unit |
| Preview gate | `127.0.0.1:30185` | Same, own unit, behind Tailscale Serve `:8445` |

- `TransferMiddleware` (`backend/http_plumbing.py`), for whole `200` JSON/text bodies without their own `ETag` or encoding:
  - GET gets a weak body-hash `ETag` with `private, no-cache`; an unchanged reread is `304` (the handler still runs).
  - Bodies ≥ 1 KiB are gzipped for accepting clients; > 256 KiB hash/compress in a thread.
  - SSE, streamed proxies, files, ranges and routes with their own validator (`/fileviewer/raw`) pass through.
  - Request bodies with `Content-Encoding: gzip` are inflated first (400 corrupt, 413 past 512 MiB). `lib/backend/api/http.ts` `jsonInit` gzips JSON bodies ≥ 64 KiB with `CompressionStream` when the browser has it.

### Host processes

Every long-lived child ends with the backend, also when it dies without shutdown:

| Process | Ends through |
| --- | --- |
| Jupyter Server | Bootstrap thread watching its parent (the Jupyter service); stops like SIGTERM, exits after 10 s if that hangs |
| Kernels | Linux parent-death signal (SIGKILL when the server ends) |
| Neovim | PTY hangup when the backend side closes |
| MinerU `LocalAPIServer` | stdin EOF watcher (`MINERU_API_SHUTDOWN_ON_STDIN_EOF`) |

- Safety net `lib/process_owner.py`: the Jupyter Server and Neovim get `GIGACHAD_PROCESS_OWNER=<pid>:<start time>` (`owned_environment`), inherited by their children. Startup `reap_orphans` SIGKILLs the user's processes whose owner no longer runs (a reused pid counts as ended); shutdown `stop_owned` SIGKILLs what carries its own mark. Neither touches the caller or its ancestors. Linux `/proc` only.
- Shell terminals (`routes/terminal.py`) are not marked: hangup ends the shell and its plain jobs, detached programs survive.
- A tab's working directory (`contexts/tabWorkingDirectory.ts`, claimed with `useTabWorkingDirectory`, read with `tabWorkingDirectory(tabId)`) is sent as the shell's `cwd`; the backend uses it when it is an existing absolute directory, else home. A running shell keeps its start directory.

## Frontend reads and loading

- `contexts/StartupReadContext.tsx` bounds startup reads (settings, models, prompts, projects, histories, sidebar listings, reasoning levels) to 15 s, reports failures in one banner and retries only failed reads, without remount or losing drafts. Timed-out/disposed requests abort, obsolete responses are ignored. No automatic retries here; streams, uploads, writes and post-mutation refreshes are not part of it.
- Every tab stays mounted with its own sidebar and chat, so user-global values are shared resources: `lib/resource.ts` (one cached value, one request), instances in `lib/backend/resources.ts`, hooks in `hooks/useResource.ts`. Models, prompts, reasoning levels, roots-enabled, provider usage, notes, project documents, all documents, root tree, worktrees, template sets live there. A mutation `set`s its response or refreshes the resource, never a local copy.
- Freshness: lists with change rows (notes, project documents, all documents) are invalidated by `watchDocumentLists` (mounted once in `AppShell`) from the change feed, own device included, and file events. Values without change rows revalidate on tab activation (`revalidateOnActivation`). Provider usage polls once in `AppShell`.
- `ProjectProvider` keeps the project list's identity when a refresh is unchanged. `TabSidebar` is memoized beside the chat state owner so streamed tokens never re-render navigation: keep its props referentially stable, and keep `memo` on the wrapper (React 19.2 kept stale `useEffectEvent` closures in a directly memoized `Sidebar`).
- Startup reads only what the first view needs: `GET /api/fileroots/enabled` (no walk) at startup, `/fileroots/files` only when the picker opens.
- Measure first-load JS with `next build --webpack`: the `<script>` chunks in `.next/server/app/index.html`, except `polyfills-*.js`.
- Heavy surfaces load on first use through a `Lazy*` module (own file, `next/dynamic`, `ssr: false`); callers and test mocks use the `Lazy*` module, since a static import of the target defeats the split:

| Module | Loaded through |
| --- | --- |
| pdf.js / react-pdf | `components/content/LazyPdfViewer.tsx`, `features/editor/pdf/LazyAnnotatedPdfViewer.ts` |
| Canvas editor | `features/editor/canvas/LazyCanvas.ts` |
| React Flow, libavoid, LikeC4 diagram | `features/editor/architecture/LazyArchitectureWindow.ts` |
| Shiki | `packages/chat-ui/src/shikiBundle.ts` on first highlight |
| Mermaid, Markmap, pdf-lib, notebook | `import()` at the point of use |

- Deep link `?open=<base64url JSON>`: `{"kind":"chat","history_file":"..."}` or `{"kind":"document"|"canvas"|"architecture","path":"...","project_slug":"..."}` (slug optional), validated and removed by `app/_components/deepLink.ts`.

## Identity and filesystem policy

- Production listeners stay on loopback behind private Tailscale Serve, which strips client-supplied identity/capability headers. `Tailscale-User-Login` resolves to a database user; exposing the backend directly breaks this boundary.
- Host execution needs capability `gigachad.local/cap/host` in `Tailscale-App-Capabilities`; missing/malformed grants grant nothing.
- Development identity needs `GIGACHAD_ENV=development` and `GIGACHAD_DEV_USER`; that login has host access.
- `X-Device-Id` (UUID) is write attribution and echo suppression, never authorization; SSE may omit it.
- `resolve_user` reads once per request and writes only for a new login, a new/reassigned device, or a device last seen over `DEVICE_SEEN_INTERVAL` (1 min) ago.
- Per-user display preferences are `users` columns (`lib/user_preferences.py`), e.g. `animated_background` in `GET /api/config`, set by `PUT /api/config/animated-background`.
- Production `GIGACHAD_CORS_ORIGINS` must not be wildcard/empty. WebSocket origins are validated explicitly; CORS does not protect sockets.

| Operation | Policy |
| --- | --- |
| Pi sessions, factory, Neovim, workbench, LaTeX builds, learn mode, mounting/cloning roots, directory browser, redeploy | Host access. A mounted root is readable in full, so mounting is a host action. |
| Ordinary file access | The user's registered mounted roots and admitted linked worktrees |
| Host-access file content | Outside mounts only under the service user's home; unreadable is 403 |
| Git/tree/worktree endpoints | A mounted/admitted root. Exception: the host-gated factory core (`tasks.diff`) reads a task worktree without a mount. |
| Credential files | Outside mounted roots, host access rejects `.ssh`, `.gnupg`, `.secrets`, `.aws`, `.azure`, `.kube`, `.docker`, `.password-store`, `.pki`, `.mozilla`, `.config/{gh,gcloud,google-chrome,chromium,BraveSoftware,op,rclone}`, `.local/share/keyrings`, `.netrc`, `.git-credentials`, `.pgpass`, `.npmrc`, `.pypirc`, shell/psql/python history, `.env`, `.env.*`, `gigachad-bot/env`, OMP `agent.db`/`models.yml`. Explicit mounts stay admitted. |
| Directory browser | Lists host directory names for host logins; not read/write permission |
| File trees | Honor layered `.gitignore`; skip `.git`, `node_modules`, `.obsidian`, `.trash`. Content paths still go through access validation. |
| Text decoding | 415 for non-UTF-8 instead of lossy overwrite |
| Share links (`/api/shared/{secret}`) | The secret is the grant to one resource, acting as its owner without a device; still behind the private ingress. Unknown, rotated or revoked secret: same 404. |

## Persistence and synchronization

| Data | Authority |
| --- | --- |
| Chats, prompts, project documents, model configuration, sandbox records | User-scoped PostgreSQL namespaces via domain stores / `PostgresDataStore` |
| Memory | memory-store files (package `memory-store`), not PostgreSQL. Root per login (`memory_context.memory_root`): host access `store.default_root()` (`~/.pi/memory_store`, shared with Pi), others `GIGACHAD_MEMORY_DIR` or `<repo>/.memory_store/<user id>` |
| Share links | `share_links` rows: secret SHA-256 only, one link per resource, FK to `documents`/`assets` with `ON UPDATE/DELETE CASCADE` (follows renames, dies with the resource) |
| Uploads, PDF library, extracted Markdown/images | PostgreSQL assets, with a convenience filesystem mirror. Only read back: `GET /api/assets/Mineru/images/<stem>/<name>` for a host login restores a missing image from `Documents/Mineru/images/<stem>/<name>` or flat `Documents/Mineru/images/<name>` (`attachment_materialize.restore_library_image`) |
| Mounted file contents | The disk files; projects/attachments reference them, no editable copies |
| Architecture | Linked clone `docs/architecture/` when present, else PostgreSQL `graph/<slug>/` |
| Diff notes/layout, per-device preferences, archived Code directories | Browser-local |
| Workbench documents | Disk files under `config.WORKBENCH` (`GIGACHAD_BASE_DIR/workbench`), referenced by absolute path from `project.json`, or a LaTeX folder as one entry. Chat-to-Pi-session links: per-user `workbench/chats.json`, only via `lib/workbench/links.py`. PDFs are never copied. |
| PDF annotations | Canvas `pdf-annotation/<sha256>.canvas` keyed by PDF bytes; created empty by `POST /api/fileviewer/annotations`. Same bytes share annotations, renames keep them, changed bytes start fresh. |

- Logical namespaces are independent of the Documents layout; storage keys are validated against absolute paths and traversal. Stores and registries are scoped to the authenticated user.
- Conditional writes check expected revisions. `apply(writes, deletes)` is atomic: one stale revision aborts the batch.
- Read several documents with `DataStore.read_many` (one query). Content columns are fetched in binary format and stored `STORAGE EXTERNAL`, so a byte range fetches only its TOAST chunks.
- `documents.meta` (jsonb) holds chat tree metadata (`chat_id`, `parent_id`, `branch_message_idx`, `children`, `qa_count`) and `refs` (referenced keys, for delete cleanup). `ChatStore`/`CanvasMutationStore` write it with the content stamped with the row version; other writes clear it. Branch metadata, chat index and cleanup read it via `DataStore.list_meta`, never bodies. Missing, stale or `refs`-less meta is derived from the body and stored back (`write_meta`, hash-guarded, no revision or change row); meta jsonb cannot round-trip (NUL, floats, unsorted keys) stays NULL and is derived each read. Annotation canvases also keep `hosts` (see Delete cleanup).
- `documents.sha256` is a generated column, so hashes never read content.
- Dev and production PostgreSQL stay separate; their default Documents/PDF mirror is shared. Back up database and mirror.
- Model defaults come from stored provider configuration. Shipped prompts seed only an uninitialized collection (checked once per user per process), never overwriting or restoring deleted defaults.
- Numbered `db/schema/` steps apply under an advisory lock in one transaction, checksums in `applied_schema_steps`; modified applied steps are rejected. Add steps, never rewrite history.

### Project rename and archive

- `ProjectStore.update_project` slugifies; a new slug runs `PostgresProjectRenamer` in one transaction (existing destination: `StorageConflictError`). It moves keys under `chat/`, `project/`, `graph/`, `memory/project/`, `attachment/project/`, rewrites slug/history fields and app-relative links in JSON/`.md`/`.txt`/`.c4`/`.canvas`/`.snap` documents and canvas mutations, recording change rows. Prose and absolute paths stay.
- A disk folder linked by the old folder name stays linked via `linked_root` in `project.json`.
- Workbench folders: `workbench.move_project_directories` moves `WORKBENCH/<kind>/<old>` to `<new>` before the transaction (non-empty destination: conflict), the transaction rewrites absolute path prefixes (file lists, `chats.json` keys, canvases, mutation log, notes); a failed transaction moves folders back. A crash between move and commit is not recovered.
- Memory: `project_rename.py` moves an unlinked project's memory_store after commit (skipped when none or the name is taken).
- `archived` is a catalog flag (`PATCH /api/projects/{slug}`). Code directory archive is per device (`localStorage` `archived_code_directories`). Pi session archive goes through pi-host.

### Project files, rename and move

- `project.json` `files` entries are a key string or `{"path", "name"}` while a display name is set. Every read-modify-write of `project.json` goes through `ProjectStore._update_project`: reread and reapply on conflict, up to 10 attempts, then `StorageConflictError`.
- `POST /api/documents/rename {slug, path, name}` renames the file itself, keeping suffixes unless the name already ends with the last one. Empty, separator, `.`, `..`: 400. Name in use in its folder or shown by another project file (case-insensitive): 409. Same name: no-op. Legacy display names of the key are dropped (`ProjectStore.forget_file_names`).
  - Stored file or note (`document_service.rename`): only notes and `project/<slug>/document/…` keys; `PDFs/…` is 400 (canvases and folders are not offered). The library copy `PDFs/<old>` follows (`follow_library_pdf`), MinerU re-extracts.
  - Disk file (`document_service.rename_on_disk`): `FileRoots.rename` within its folder under write policy (403 outside, 404 missing/unreferenced, 409 target exists); a failed reference rewrite moves the file back. A `.tex` drops `.latex-build/<stem>.*` of both names (`latex.discard_build`).
  - LaTeX references (`lib/latex_references.py` `rewrite_references`): repoints `\includegraphics`, `\input`, `\include`, `\includeonly`, `\bibliography`, `\addbibresource` (resolved against the project folder) and `% !TEX root` (against its own file) in the project's `.tex`/`.bib`; comments untouched, extensionless references stay extensionless when unambiguous, `./` kept. Dot folders, `node_modules` and build output are skipped. Each rewritten file is a `document` change and listed as `changed` in the response; the frontend (`useFileRename`) announces each. A cascade failure is logged and never undoes the rename.
- `GET /api/documents` and `/all` return the display name as `name`; the project list returns `file_names`. The frontend resolves names through `useDisplayName` (`ProjectContext`).
- `PostgresKeyRelocator.relocate` (`project_rename.py`) runs moves and renames in one transaction: rewrites every exact reference (project file lists, chat attachments, canvases and mutation log, note links, `chats.json`), then rekeys the row (an absolute `old` only rewrites references). Change rows carry `moved_to` for renames and `operation` (`write`, `move`, `delete`), also in NOTIFY and SSE, so devices move tabs and tell deletes from rewrites. Moving a stored file to disk records `move`, not `delete`.
- `POST /api/documents/move` between projects or notes keeps the shown name.
- Upload `POST /api/documents/write-binary` stores `project/<slug>/document/<name>`; a PDF is also synced to `PDFs/<name>` and queued for MinerU (`attachments.sync_pdf`).
- `POST /api/documents/move-to-folder` writes a stored file into a file-root folder (never overwriting, 409), relocates references to the absolute path, deletes the row; a failed relocation removes the written file. Canvases cannot move.
- `POST /api/fileroots/add-to-project` references dropped mounted files/folders by absolute path. A folder holding a root `.tex` (`latex.holds_root_document`) stays one `inode/directory` entry; other folders expand to PDF and UTF-8 text files. Removing the entry only unlinks.
- Sidebar drags carry a kind (`FileTreeItem.dragKind`/`accepts`); chats and canvases never drop on Code folders.

### Change delivery and canvases

```text
committed user write -> durable changes row -> PostgreSQL NOTIFY
  -> process ChangeBroker -> user-scoped SSE -> resource-specific refetch
```

- `/api/sync/stream?since=<seq>` resumes from the applied sequence: subscribe before replay, suppress overlap. Replay (`changes.latest_changes`) sends the latest change per resource.
- Without `since`, the stream sends `event: cursor` with the head (`changes.head_cursor`) and only later changes. Startup reads, file editors and the architecture view await `syncStream.syncCursorReady()` so a racing change arrives as an event; no event within 5 s releases the reads and reconnects with `since=0`.
- `UserScope.record` deletes superseded rows (same user, kind, key; `FOR UPDATE SKIP LOCKED`) in the same statement, keeping about one row per resource.
- Events carry metadata, never content. Subscriber queues are bounded; lagging streams close and recover through replay.
- `syncStream.subscribeToChanges` drops this device's own changes unless `{ ownDevice: true }` (only `watchDocumentLists`).
- Canvases: snapshots in documents, accepted mutation batches in `canvas_mutations`. Writers serialize on a document row lock; snapshot/log/change rows update in one transaction, mutation IDs deduplicate, batches replay by revision. Ink points are stored at 0.01 x/y and 0.001 pressure (`inkPoint`, `lib/drawing.ts`). Canvas undo is separate from resource deletion.
- Canvas chat (host access): an open canvas has a Pi chat (`FileChat`). `POST /api/workbench/prepare` returns a stored `.canvas` key unchanged (404 missing) and an absolute path in place; sessions run in `WORKBENCH/canvas` with the canvas prompt, linked under the canvas key in `chats.json`, and see the canvas only through screenshots (`useCanvasScreenshot`, attached via `CanvasCaptureContext`).

## Chat, models, and tools

| Boundary | Contract |
| --- | --- |
| Chat stream | `/api/chat` SSE: tokens, usage/tool events, completion, failures. Proxy buffering off. The POST uses `retryOffline` (a `TypeError` retries on a fresh connection, so a reply may generate twice); transport failures are written into the reply. `useChatStream` paces tokens via `lib/backend/tokenPacer.ts`. |
| Project context | Memory scope binds to the chat's containing project/tab. Branch, merge, move and cascade delete commit all chat and `project.json` writes in one `DataStore.apply`; upload cleanup runs after. |
| Providers | Every backend model call goes through `lib/completions.py`: `openai` SDK on pi-host's gateway (`pi_host.gateway_url()`, bearer `pi_host.token()`), model `pi_models.gateway_model_id(selector)`. Stored selectors keep their spelling, translated only here (`litellm_proxy/x` -> `x`, `gemini/m` -> `google/m`). An unserved model fails with `ModelNotServed`. The gateway clamps `reasoning_effort` and ignores `response_format`. Use stored defaults, not hard-coded models. |
| Search | Brave LLM Context with backend-only `BRAVE_API_KEY`, then the selected model answers with citations. |
| Research | GPT-Researcher with DuckDuckGo (`ddgs`), chat models `vllm_openai:<gateway id>`, embeddings Ollama `nomic-embed-text`. Effort in `LLM_KWARGS` (not passed on deep research's plan/learnings calls). Runs serialize on `RESEARCH_ENV_LOCK` because gateway URL, token, Ollama URL and tiktoken cache (`~/.cache/gigachad-bot/tiktoken`) are process environment, restored afterward. |
| Sandboxed tools | Agents-in-a-box runs workspace code and plots. A workspace-agent run is a `PiTask` with stored `sandbox_model` (`<provider>/<id>`, `--provider gateway`); the container reaches the gateway with the gateway-only token on `PI_GATEWAY_PORT` (dev 8300, prod 8301). |
| Optional services | Report unavailability; never silently substitute a local service. |
| Memory tool | Toggle `memory` (default on) adds `list_memories`, `read_memories`, `apply_memory_ops` (`lib/tools.py`, threaded, on `ToolContext.memory_root`). `memory_context.memory_prompt` creates `user` and the project's memory_store if missing and appends `# Memory` (`user` profile/preferences, indexes, rules). Project store name: `linked_root` folder name, else slug, `project-user` for `user`. Any failure drops the block, never the chat (`routes/chat.py` `_memory_block`). |

- The chat's turns render in the conversation column shared with Pi: [conversations](#conversations-chat-and-pi).

### Sandbox checkpoints

- Slots are scoped by user, chat and sandbox scope. `SandboxRuntimeState` lives per backend process and serializes per slot (two workers can still run one slot at once). Storing a run and `delete_chat_state` do store IO in a thread holding the slot lock and `SandboxRuntimeState.assets_lock`; asset deletes stay on the event loop.
- A run's workspace archive, rich output assets and run record are stored when the run returns, assets before the manifest that names them; the chat save does not checkpoint. `SandboxStore.checkpoint` writes slot, manifest, pointer and run records in one transaction. A failed or cancelled run records itself and leaves the active pointer where it was. Runs of a chat that is never saved are not collected (see "Delete cleanup").
- Completed `tool_call_id` results are reused; a running record is busy, abandoned ones (> 5 min) may rerun; native resources are released on cancellation.
- Only a manifest with `pi_sessions` resumes. A schema-1 manifest (`omp_sessions`) restores its workspace and starts a fresh Pi session.

### Live attachments and extraction

- Sources live in `Attachment.vaultPath`, re-read on send/open (deduplicated per source, inactive ones deferred). A moved file is relocated only when the original mount has exactly one match; missing/ambiguous keeps read-only last text. Repeated missing reads are skipped on send, retried on preview.
- Text edits save through the original write path before chat state; PDF/image attachments are never text-saved.
- Changed referenced PDF bytes sync into the library and invalidate extraction. Old text stays while re-extraction runs (the first send may use it). Legacy `PDFs/<name>` references still resolve.
- One extraction worker per backend process; its queue is process-local, not a cross-worker lock. `mineru.parse_pdf` runs in a temp dir; the per-user `Mineru/<stem>.md` asset is the only cache. Results persist to PostgreSQL; mirror failures do not invalidate them. `MINERU_SERVER_URL` selects external OCR, else local MinerU per parse ([remote inference](remote-inference.md)).
- Library keyed by filename (`PDFs/<name>`), extraction by stem (`Mineru/<stem>.md`, `Mineru/images/<stem>/`): reusing a filename replaces the entry.
- Project PDFs (upload, add-to-project) queue with an `on_markdown` hook calling `document_service.add_extraction`, which stores `project/<slug>/document/<stem>.md` and lists it (an existing one is kept). It writes as no device, so the uploader's sidebar also reloads on the `chat/<slug>/project.json` change event. Deduplication by name and content: the same PDF queued twice at once lists Markdown only for the first project. The sidebar hides that Markdown (stored, or moved to `…/workbench/markdown/<slug>/<stem>/<stem>.md`) and deletes it with its PDF (`workspaceTree.extractionsOf`).
- Pasted images, drawings, canvas images and architecture source attachments are snapshots.

### PDF viewer and assistant

- `GET /api/fileviewer/raw` sends `ETag` + `private, no-cache` (304 from metadata: asset `sha256`, document version/`updated_at`, host stat) and serves single byte ranges sliced in Postgres. CORS exposes `Accept-Ranges`/`Content-Range`. Annotation key hashing never loads the PDF.
- `features/editor/pdf/AnnotatedPdfViewer.tsx` anchors strokes to pages, reusing canvas ink, history and collaboration; save failures show as unsaved.
- `PdfAssistant.tsx`: with host access a Pi session per PDF in `WORKBENCH/pdf` (`FileChat`); otherwise a local chat whose system prompt embeds the MinerU Markdown (mounted, `PDFs/…`, or `project/…` via `PDFs/<name>`), polled every 5 s during extraction. History is not saved. Once Markdown is ready a PDF/Text switch shows it with `images/` links rewritten to `/api/assets/Mineru/images/<stem>/`.

## Conversations (chat and Pi)

One column and one turn card for chats and Pi sessions; what only one side has stays on its side (chat: web search results, sources, research trace, project documents; Pi: session search, queue, held messages, roster, compaction, live tool output, attachment sending). Rendering parts shared with LearnHarness stay in `packages/chat-ui` (`Turn.tsx`, `turn.css`, `previewHead.ts`): `[data-chat-column]`, `.gc-turn-preview-scroll` and the package's exports keep their names.

| Part | Contract |
| --- | --- |
| Column | `features/conversation/ConversationColumn.tsx` takes `turns` (`{key, label, icon?, live}`), `renderPreview`, `previewText`, `renderBody`, `renderAfter`, `header`, `footer`, `reveal` (`{key, seq, scroll?}`), `followSeq`, `paddingBottom`, `scrollRef`, `wide`/`onWideChange`, `readOnly`. Opens at the end and follows it; until the reader scrolls, clicks or types, late layout keeps it there. Mounts the newest `TURN_PAGE_SIZE` (20) turns, earlier pages on scrolling up; away from the end the window's top stays put. Collapsed previews render near the viewport only (an empty preview reserves no space). Ctrl/Cmd+Up/Down focus turns, Enter toggles, Esc drops the focus; a key reaches the innermost column of its surface, never a `readOnly` one (the session picker's preview). Each turn sits in `div[data-turn-key]`; `ColumnTurn` is memoized on its content. Width and wide mode (`WideModeToggle.tsx`) are shared app-wide. |
| Turn keys | Chat: `keyPairs` (`lib/domain/pairs.ts`), a pair keeps its key across an earlier pair's delete, a streamed answer, a toggled attachment. Pi: the prompt's `entryId` (`pending-<start>` until the host confirms it). Open state and edit drafts live by key. |
| Stable props | Nothing handed to a settled turn changes identity because the message list changed; a streamed frame re-renders the live turn only. `useChatTurns` reads messages and `isStreaming` through refs, `laterTurns` is a getter.|
| Turn card | `features/conversation/TurnQuestion.tsx`: `TurnQuestion` (copy, edit, delete) and `TurnAnswer` (copy, branch) on chat-ui's `TurnRegion`. No actions on a live turn; inside `TurnEditsPaused` (a reply streams: `ChatContainer`, `PiTranscript`) no edit or delete, without props changing. Edit turns the question into a textarea: Enter confirms, Esc cancels, Shift+Enter is a newline; unchanged or empty cancels. Delete always asks, an edit asks when later turns go (naming how many); the confirmation is the shared `Dialog` in a portal. |
| Delete and edit | Delete removes one turn entirely, edit removes the turn and every later one, then sends the edited text with the turn's attachments; neither creates a session or history file (only branch does). Chat (`useChatTurns`): `removePairs(user, "pair" \| "end")`, saved first through the history file; a failed save keeps the pair and shows `turnError`; a failed send puts the text into the composer. No undo for pairs. Pi (`PiTab`): `deletePiTurn(sessionId, entryId, "turn" \| "end")` -> `POST /api/pi/sessions/{id}/turns/{entryId}/delete` -> pi-host rewrites the session file and restarts the child ([Pi migration](pi-migration.md) D18, F13); the transcript follows the host's `resync` (`mode: replace`). Refusals (409 `running`, `compacted`, `locked`, `changed`; 400 `not_prompt`, `off_branch`, `factory`, ...) show as notices; files the cleanup could not remove (`notRemoved`) show once as a warning. |
| Chat list and search | `GET /api/chat-histories/all` -> `{chats: ChatListing[]}` (`historyFile`, `projectSlug`, `title`, `updatedAt`, newest first); `GET /api/chat-histories/search?q=&limit=` (`q` at most 200 characters, else 422; blank 400; `limit` 1-100, default 20) -> `{hits}`, each a listing plus `matches` (`qaIndex`, `role`, `ordinal`, `snippet`). `ChatStore.search` scans question and answer text case-insensitively, newest first, in batches of 50, nothing indexed: at most 20 matches per chat, 60 characters of snippet each side. Client: `listAllChats`, `searchChats` in `lib/backend/api/chat.ts`. |
| Chat picker | `features/chat/ChatPicker.tsx` (Alt+C in the active tab, mounted by `TabContent`), `useChatPicker.ts` (200 ms debounce, one abortable request per query). Without a query the chats grouped by project, newest group first; with one the title matches, then a row per content hit. A pick opens the chat in the active tab (`onOpenChat(historyFile, qaIndex?, hit?)` -> `ChatReveal`); `ChatContainer` settles the reveal once its own file loaded and paints the hit with `features/conversation/useTextHighlights.ts` (highlights `chat-search`, `chat-search-current`; text tagged `data-chat-text="<index>:<role>"`). Pi's search jump uses the same painter (`features/pi/useSearchHighlights.ts`). |
| Tool rows | `features/conversation/tools/`: `ToolCallRow` (optional `display` label/icon and `stages`), `ProcessDetails`, `DiffView`, `patch`, `format`, `tools.css` and the registry `renderers.ts` (`registerToolRenderer`, `toolPlacement`, `renderToolCall`). `features/pi/tools/DiffView.tsx` and `format.ts` re-export two of them. The chat adapts a `ToolCallRecord` to `{call, run}` (`features/chat/toolRun.tsx`, one stable run per record) and renders it with the chat's label and the result of the tool family under the row (`features/chat/toolRenderers.tsx`: web search, deep research, plots, diagrams, mind maps, workspace agent, learn session). `app/_components/TabContent.tsx` loads Pi's memory row for the chat without a chat -> Pi import. |
| Factory cards | A `factory-notice` or `factory` tool result with a card renders it (`features/pi/factoryCard.ts`, `FactoryCard.tsx`, `factoryView.tsx`); behaviour: [agent factory](agent-factory.md#notices). |

### Backlog: cleanliness refactors

Not done, carried over from the conversation unification plan:

- **Split `PiTranscript.tsx`**: piece derivation (`answerPieces`, `assistantPieces`, `headPieces`) beside `transcript.ts`, pure and cacheable per message; rendering, search jump and held messages as small files.
- **Explicit renderer registration**: one index listing every tool and message renderer instead of the side-effect imports in `PiTranscript.tsx`, heavy renderers (video preview, diff view) loaded lazily.
- **`PiTab.tsx` as a state machine** (no session, starting, attached, closed) instead of the refs guarding effects (`firstSent`); roster, dock and header as components reading the state.
- **Structured subagent data**: pi-host forwards the roster as a typed event instead of parsing the `PI_SUBAGENT_ASYNC_JSON:` widget text (`features/pi/subagents.ts`); touches the contract with the pi-subagents extension.
- **Drop the session-list poll**: `sessionStore.ts` refreshes every 30 s (`POLL_MS`) besides `sessions_changed`, socket reopen, focus and visibility; drop it once pi-host tests prove every list change emits the event.
- **Ambient pause** (`deepfocus-ambient`, own repository): a `paused` flag in `NightOceanFrame` so the shell can pause an unfocused window without the pop `still` causes.

## Architecture workspaces

| Concern | Contract |
| --- | --- |
| Workspace | Merge all `.c4` files. Linked mount folder name matches the project slug. Switching worktrees does not move the linked clone's source. |
| Files | Classify by declarations: systems declare elements/views, modules extend systems, saved views are `views/<id>.c4`. |
| Discovery | Git: tracked plus non-ignored untracked; non-git: all files. No symlinks. |
| Relations | Stored once with the source element; shown in any view holding both endpoints. |
| Layout | `.likec4/<viewId>.likec4.snap`; invalid snapshots fall back to auto-layout. Positions survive rename/addition, explicit pins survive parent moves. |
| Rendering | React Flow, libavoid with fallback when WebAssembly is unavailable. Fit-to-view is display-only. |

```text
diagram / source edit -> serialized frontend reads/writes -> architecture route
  -> workspace snapshot -> LikeC4 POST /model or /apply -> reparse
  -> revision-checked storage writes -> returned model
```

- Operations are syntax-tree text edits, not a second model. Hand edits must parse; operations refuse new errors, prune references to removed elements and reject remaining invalid ones with file/line diagnostics.
- Revision conflict: reread and revalidate; 412 when retries fail. Only changed pre-existing files are revision-checked (new files, deletions, reads are not).
- PostgreSQL batches are transactional. Disk batches check all revisions first, then replace files one by one; removed sources/layouts are deleted afterward without checks.
- Frontend reads/writes are serialized. Filter/style-based views are read-only on the canvas. Removing from a view differs from deleting from the model.

## Git diffs and worktrees

- Mounted clones are managed through `/api/fileroots/worktrees` and its operation endpoints. `POST /api/fileroots/clone` clones https/ssh links (regex-validated, `--` before the URL, `git_worktree.clone_repository`) into `~/git/<repo>`, mounts it and adds it to Pi's start folders.
- Remote git runs with `GIT_TERMINAL_PROMPT=0` and `ssh -F ~/.ssh/config` (or `/dev/null`) because the service sandbox cannot read `/etc/ssh`; an explicit `GIT_SSH_COMMAND` wins.

| Concern | Contract |
| --- | --- |
| Diff API | `/api/fileroots/diff/files`, `/diff`, `/diff/content`: working tree vs `HEAD`, poll by stamp |
| Review anchors | Per worktree in localStorage: selected text, original line identity, nearby same-side context. Resolved by text and context, not line number. Unresolved notes stay outdated. Failed sends keep notes; successful sends clear only submitted unchanged notes. |
| Native edit | Read-only `HEAD` plus editable working buffer in Neovim. Rejects deleted, binary, non-UTF-8, oversized, symlinked or escaping paths. |
| Mount selection | Selected worktree persisted per mount |
| Default checkout | `origin/HEAD`, else the main checkout's branch; never pulls or switches the main checkout |
| New worktree | Base `dev`, else remote default/main fallback; fetch and fast-forward the base; `~/git/worktrees/<repo>/<name>`; refuses existing branch/folder |
| Removal | Dirty/untracked needs explicit force. Branch delete is separate, unmerged needs confirmation, remote branches never |

## Pi tabs

Coding tabs, the session picker and workbench file chats render Pi sessions natively (no iframe). Plan and phases: [Pi migration](pi-migration.md); decision: [ADR 0003](adr/0003-pi-as-agent-runtime.md).

### pi-host

Bun service (`pi-host/`, `run-pi-host.sh`, `gigachad-bot-pi-host.service`): one `pi --mode rpc` child per open session, snapshot then live events to all attached clients, idle eviction, session lists and full-text search from Pi's session files, the completion gateway. Runs continue without a client and across backend reloads.

- Factory task sessions (`<agent dir>/factory/sessions/`) belong to the factory runners: never listed, archived or searched; resuming one with a `factory-state` entry is 409 `factory`.
- `autoResume` (default off, `GET/PUT /api/pi/settings`): on start, resumes sessions a dead pi-host left mid-run (`active-runs-<pid>.json`, owner checked by pid and `/proc` start time).

| Profile | Children |
| --- | --- |
| `coding` | The user's Pi plus `-e <abs path>` per top-level `pi-extensions/` entry (`*.ts` except tests/`.d.ts`, or a dir with `index.ts`; `PI_HOST_EXTENSIONS_DIR`) |
| `document` | Workbench chats; extensions as `coding`, none when a `tools` list is passed |
| `sandboxed` | `coding` arguments without extensions, via `agent-sandbox pi-rpc --profile <PI_HOST_SANDBOX_PROFILE> --cwd <cwd> --session-dir <dir> --gateway-port <PI_HOST_PORT> -- <pi args> --session-dir <dir>` (`PI_HOST_AGENT_SANDBOX_BIN`); session file stays host-side; a requested model runs as provider `gateway` |

- Local `coding`/`document` children always load `pi-host/extensions/gateway-provider.ts` (even under `--no-extensions`) with `PI_GATEWAY_URL` (`http://127.0.0.1:<PI_HOST_PORT>/v1`) and `PI_GATEWAY_TOKEN`: it registers provider `gateway` with the gateway's `/v1/models` minus providers the child runs itself (Anthropic login, `ollama`), so children reach other providers without their keys and `~/.pi/agent` stays untouched. A start model is tried as `<provider>/<id>`, then `gateway` + `<provider>/<id>` (`childModels`).
- Tokens: the full token (`PI_HOST_TOKEN_FILE`) opens every route; the gateway token (`PI_GATEWAY_TOKEN_FILE`, default `pi-gateway.token` beside it, 0600) only `/v1/*`. Children get an environment allowlist plus the gateway token/URL, never the full token or provider keys. A factory runner started by a core action inherits pi-host's whole environment.
- `pi-extensions/exclude.json` lists extensions local `coding`/`document` children skip (package source, extension name or path). With entries, pi-host resolves Pi's extension set per spawn and passes `--no-extensions` plus `-e` for kept user-scope and built-in extensions; project-scope ones are dropped (an RPC child cannot grant project trust).
- Tool-result images never travel as base64: `pi-host/src/media.ts` replaces them in frames, replays, snapshots and `/sessions/{id}/entries` with `{type:"image", mimeType, hash}` (PNG/JPEG/GIF/WebP cached in `$XDG_CACHE_HOME/gigachad-bot/pi-media`, swept after 30 days; others stay inline; user prompt images untouched). `GET /api/pi/media/<sha256>` (pi-host `/media/<hash>`) and `/thumb` (480px WebP, `lib/pi_media.py`) are immutable-cached, `If-None-Match` answers 304 without pi-host.

### Backend adapter

- Only `src/lib/pi_host.py` knows pi-host URLs, shapes and token. `/api/pi/*` (`routes/pi.py`, host access) lists repositories and sessions and relays `/api/pi/ws`. Unavailable 503, unknown shape 502, refusals pass through.
- Start folders (under `~/git`, `~/dev`, Git checkouts within 4 levels of `~`) come from the `repositories` table (`lib/repository_index.py`): rescanned in the background at startup (inline when empty), a missing name rescans once before 404, clones are added at once. Any absolute folder may hold sessions and is stored when used.
- `POST /api/pi/pastes?repo=|cwd=` (multipart `file`, `lib/pi_pastes.py`): mp4/webm/mov/mkv, 413 above 200 MB, folder inside the file roots else 404, no symlinks. Stored as `<cwd>/.pi/pastes/<YYYYMMDD-HHMMSS>-<name>.<ext>` beside a `.gitignore` of `*`; the draft gets `@<relative path> `. The previewing extension is the user's (`~/.pi/agent/extensions/video-paste.ts`), emitting a `video-preview` message (`details.videos`: path, duration, size, fps, `frames` with `seconds` and image) rendered by `features/pi/VideoPreview.tsx`; the card still renders the single 4x3 grid image older sessions stored; sessions without it keep the mention as text.
- `/api/pi/files?cwd=` serves `@` completion for a cwd inside the file roots or `WORKBENCH`.

### Frontend

- Only `features/pi/` knows Pi's protocol (`sessionStore`/`usePiSession`, `PiTab`, `PiComposer`, `PiSessionPicker`, `PiTranscript`); `features/workbench/` reuses it. The transcript renders on the shared column, turn card and tool rows ([conversations](#conversations-chat-and-pi)). `features/pi` never imports `features/factory`; shared shapes live in `lib/domain/factory.ts`.
- Search: `GET /api/pi/search` returns `(entryId, ordinal)` hits, `GET /api/pi/sessions/{id}/matches` a session's ranked hits, `GET /api/pi/sessions/{id}/entries` a preview without attaching. `searchMatch.ts` is the browser copy of pi-host's span matching, kept equal by a test; the ordinal counts spans per rendered text node, so markup the host counts as text can shift it.
- `renderers.ts` registers native UI per `custom` message type (e.g. `factory-notice` in `FactoryNotices.tsx`, `video-preview`) and re-exports the tool registry of `features/conversation/tools/renderers.ts` (e.g. `factory` in `FactoryTool.tsx`).
- `pi-extensions/files.ts` sends the RPC notice `Open file: <abs path>`; `usePiSession` routes it to the file panel through `OPEN_FILE_NOTICE_PREFIX` instead of a toast.
- `/btw` is the user's Pi extension (`~/.pi/agent/extensions/btw/`): `usePiSession` parses `setWidget` record `btw_state` (version 1, whole state, unknown versions ignored), sends `/btw-list` after attach when `get_commands` lists it; `/btw*` prompts bypass the held/steer/follow-up queue and go out without the lead.
- Model selector: offers catalog models Pi offers the session (`get_available_models`, by gateway id: `gateway` model `openrouter/x` is catalog `openrouter/x`); a pick sends `set_model` for the session's own provider, else `gateway` (`piModelFor`). An empty tab offers `/api/models/pi`.
- During a run Enter steers, Alt+Enter queues a follow-up, Esc/Stop aborts and returns queued messages to the draft. Messages sent during a run are held in memory (`outbox.ts`) until the queue snapshot or user message confirms them; flushed in order after re-attach; a delivery cut by a drop shows as not sent and is never resent automatically.
- Drafts persist in `localStorage` `gigachad:draft:<key>` (`chat:<slug>:<historyFile>`, `pi:<session>`) via `useComposerDraft`; removed on send.
- Shared chat rendering lives in `packages/chat-ui` (also installed by LearnHarness).
- Old omp sessions are not converted; they stay in `~/.omp/agent/sessions` for the `omp` CLI.
- Theme switch: one shell bloom (`lib/themeTransition.ts`); `LearnProvider` takes its theme from `useFrameThemeName`, which flips only once the bloom shows, because Chrome still paints a cross-origin frame's updates under the snapshot.
- Click hints (`packages/chat-ui/src/hints.ts`, app-wide via `AppShell`): `f` outside text entry labels every visible clickable; typing a label clicks or focuses it.

### Pi model source

- Model-chat Pi selections are ordinary gateway completions. `GET /api/models/pi` returns the gateway catalog (`GIGACHAD_PI_HOST_URL`, default `http://127.0.0.1:8300`, plus `/v1`); picks land in one catalog row `Pi` with `source: "pi"`, selectors stay `litellm_proxy/<provider>/<model>`.
- `ModelProviderStore` rewrites pre-Pi rows (`source: "omp"`) and a tab-order `OMP` into `Pi` on load.
- An unreachable gateway returns an offline catalog with a reason (distinct from a failed generation). Catalog and effort ladders cache 60 s.
- Ollama is Pi's custom provider `ollama` in `models.json` (`$PI_CODING_AGENT_DIR`, default `~/.pi/agent`). `lib/ollama_models.py` reads `OLLAMA_BASE_URL` (`/api/tags`, `/api/show` cached per digest; embedding models excluded) and replaces only that entry via `lib/pi_models_json.py` (locked, atomic, unreadable file never overwritten) at startup and on `GET /api/models`; the only write into Pi's agent dir. A hand-entered model Pi's registry lacks runs only when the user's own `models.json` declares it.

## Agent factory

The factory is the agent-factory package; behaviour, states, checks and failures: [agent factory](agent-factory.md). gigachad keeps only these contracts:

| Owner | Contract |
| --- | --- |
| pi-host | `pi-host/src/factory.ts`: `POST /factory/call {action, params}` -> 200 `{result}` or `{error, message, detail?}` (`FactoryError`: 400 `invalid`, 404 `not_found`, 409 `busy`/`refused`, 500 `failed`); `GET /factory/actions`. Imports the core from `factory --core-path` (PATH or `AGENT_FACTORY_CLI`, 10 s timeout, reloaded on mtime change); missing CLI: 503 `factory_plugin_missing`. Changes under `<agent dir>/factory/` send `sessions_changed`; `SessionInfo.factory` hides task sessions. |
| Backend | `routes/factory.py` (host-gated) `POST /api/factory/call` passes results through; refusals are `{"detail": {"code", "message", "detail"?}}`, pi-host down 503. `tasks.start` without `tag` gets the login (`login_tag`). `tasks.remove` of a `done` task dry-runs the revert (`lib/factory_revert.py`, `git merge-tree`, 409 `refused` on conflict), stops its preview (`PreviewError`: 502 `failed`), then `git revert --no-edit` last (skipped when nothing merged, off `HEAD` or reverted; failure is 409, task gone). `POST /api/factory/images` stores png/jpg/gif/webp ≤ 20 MB in `~/.cache/gigachad/factory-images/` and returns the absolute path. `lib/factory_migration.py` imports old `factory_*` rows into the core once at startup, marking `migrated_at`. |
| Frontend | `features/factory/` via `callFactory(action, params)` (`lib/backend/api/factory.ts`), types in `lib/domain/factory.ts`. No `waves.list` means no waves and no error. Settings save `models: {small, big, reviewer}`, reading big as `big ?? coder`. A repo without `factory.yaml` shows `config.get`'s draft, saved uncommitted with `config.write`. |
| Previews | `src/lib/factory_preview.py`, `routes/factory_preview.py` (`GET /tasks/{id}/preview`, `POST …/start`, `…/stop`, host-gated), on demand. Task from the core's `tasks.get` (id or unique prefix; state keyed by the full id). The worktree's `factory.yaml` `preview` section (placeholders `{port.<name>}`, `{repo_path}`, `{database_url}`, `{data_dir}`) starts process groups in ports 8100–8199 with a fresh database `gigachad_preview_<task id>` (backend role needs `CREATEDB`) and env `PATH`, `HOME`, `USER`, `LANG` plus config `env`. State per task in `$XDG_STATE_HOME/gigachad-bot/factory-previews/` (`state.json`, logs, `data/`), serialized by `flock` on `factory-previews/lock`, survives reloads. `ready` when every `ready_url` answers < 500; `failed` with a log tail when a process is gone (start time and boot id checked) or after `ready_timeout_s`. Stop: SIGTERM, SIGKILL after 5 s, drop database, remove state. Launch failure: 502, no state. |
| Preview gate | `src/backend/preview_gate.py` (`run-preview-gate.sh`, started by `run.sh` unless `GIGACHAD_FACTORY=0`) is the only browser path to a preview: `/preview/<port>/…` to the frontend (path kept, `NEXT_PUBLIC_BASE_PATH`), `/preview-api/<port>/…` to the backend (prefix stripped), HTTP and WebSocket. Only the dev login or a host-capability login passes; its own origin keeps agent code off the real API origin. |
| Preview URLs | Preview frontends get a path `NEXT_PUBLIC_API_BASE` resolved against the page origin (`lib/backend/config.ts`); `through_gate` rewrites loopback URLs onto `GIGACHAD_PREVIEW_URL` (default `http://127.0.0.1:30185`). Frontend code must use `appPath` (`lib/paths.ts`) and `apiRoot` (`lib/backend/urls.ts`) or break under the base path. |
| Gates | `factory/gates.sh` (`scripts/check.sh` forwards to it); rules in `AGENTS.md`, "Testing". |

## Neovim sessions

- `neovim_available` from `/api/config` requires host access, `nvim` on the backend `PATH` and `$XDG_CONFIG_HOME/nvim` (default `~/.config/nvim`). Opt-in is per device. Treat as host code execution; WebSocket origins must match `GIGACHAD_CORS_ORIGINS` (loopback for dev `*`).
- One `nvim -n` PTY per `WS /api/neovim/session`, rendered by xterm.js. Each document is a private temporary copy in its own directory with its own buffer; switching never saves. Cwd: innermost containing mount, home for database documents. Copies go on `:q`, close or disconnect; unsaved edits are lost.
- `:w` goes over RPC to the browser's normal document API (architecture validation included): per-buffer `BufWriteCmd`, frame `save`. Rewritten text replaces the buffer; failure leaves it modified.
- The session keeps `base` per document (text Neovim and storage last agreed on). `set_content` replaces an unmodified buffer and merges into a modified one with `text_merge.merge_theirs` (`git merge-file -p --theirs`), deliberately without conflict markers or notice, rewriting only changed hunks so the cursor stays. `:w` of a disk file whose disk text differs from `base` saves the merge; unreadable/non-UTF-8 disk text or a refused merge fails the save. `set_text` waits for a running `:w` of its document, `set_content` frames run as ordered tasks so a save's reply is never blocked.
- The clipboard read runs beside other requests, so a waiting paste holds up no `set_content`, `:w` or `insert_lines`; a denied permission answers `error: "Clipboard access denied"`.
- After the user's config: `more` off and `messagesopt=wait:2000,history:500` (a hit-enter prompt would block RPC); a global normal-mode `<Esc>` runs the user's mapping then sends `leave`; `vim.g.clipboard` becomes the gigachad provider for `+`/`*` (the browser is the clipboard, never the server).
- Keymaps notifying the browser are one list (`neovim_session._KEYMAPS`, `{mode, lhs, event, suffix?}`) installed per tracked buffer by `install_keymaps`; `suffix` limits to matching paths. Visual `<leader>a` reports `selection`; `<leader>v` in `.tex` reports `forward_search`.
- Incoming control frames are pydantic `_Control` models in `routes/neovim.py`, dispatched by one `handlers` dict; malformed frames are ignored, failing session work is logged (`_guarded`). Outgoing events use an ordered notice queue (`_BrowserSaves.notify`). Frontend types: `features/editor/neovimProtocol.ts`; hook `useNeovimTerminal` (`features/editor/NeovimEditor.tsx`). While connecting, `open_file` and `renamed` queue (`QUEUES_WHILE_CONNECTING`), `set_content` keeps the latest per path, others are refused.

| Frame (browser -> backend) | Effect |
| --- | --- |
| `open_file {path, line?, column?}` | Load from disk into a new copy or show the open buffer; cursor to line/byte column |
| `set_content {content, path}` | Replace or merge as above; ignored when `path` is not open |
| `renamed {from, path}` | Rename the buffer in place, keeping text, cursor, modified flag; no-op when `from` not open, refused when `path` open |
| `insert_lines {text}` | Insert below the cursor of the shown buffer |
| `insert_frame`, `insert_image` | See Presentations |
| `background {light}` | Sets `'background'` so the user's colorscheme reloads |
| `clipboard {id, text\|error}` | Answer to `clipboard_read` |

| Event (backend -> browser) | Meaning |
| --- | --- |
| `save {id, path, content}` | `:w` of a buffer |
| `switched {path}` | Neovim itself changed the shown buffer (not after `open_file`) |
| `open_failed {path, message}` | File unreadable |
| `selection {path, start_line, end_line, text}`, `forward_search {path, line}`, `run_cell {path, line}` | Keymaps |
| `leave`, `yank` | Esc in normal mode; yank/delete for the browser clipboard (best effort) |
| `clipboard_read {id, path}` | Paste: answered from `navigator.clipboard.read()` within 15 s, else Neovim pastes its own last yank. `text/plain` pastes as text; text ending in newline pastes linewise. A PNG without text is stored as an image (see Workbench). |
| `message {text}` | Error without a request |

- Notebook text (`open` with `"notebook": true`, path `<notebook>.ipynb<ext>`, `ext` `\.[A-Za-z0-9]{1,8}`, notebook readable under the file roots, else close 1003): the buffer gets the virtual name `<resolved notebook><ext>` so the user's LSP finds the project and filetype, but nothing is read from or written to it. `nvim` starts on an empty placeholder, never without a file argument, so session-restore and dashboard plugins stay idle. With a stale file at the virtual name a plain `:w` fails with E13, `:w!` still saves. `BufReadCmd` loads `base` via RPC; `BufWriteCmd` sends `:w`/`:w!`/`:up` as `save`; partial writes, appends and `:r` are refused; no swap/undo file, no disk merge. `:sav`/`:file` restore the virtual name. `buftype` stays empty on purpose (with `acwrite` `vim.fs.root` ignores the name and LSP loses the project). Backings: `_PrivateCopy` vs `_NotebookText` (`neovim_session.py`); only private copies are renamed or unlinked.
- Notebook keymaps (normal and insert): `<C-CR>` saves and, only when stored, sends `run_cell` with the cursor line; `<S-CR>` also moves to the next `# %%` cell, appending one if none. xterm cannot tell these from Enter, so `NeovimEditor` sends CSI-u (`ESC[13;2u`, `ESC[13;5u`) for notebook documents only.

## Shell terminal

- `t` outside text entry opens the active tab's login shell (`features/editor/ShellTerminal.tsx`); host access only. One shell per tab id, kept while the tab exists (hidden when inactive), hung up when the tab closes.
- `WS /api/terminal/session`: first frame `{"type":"open","rows","cols","cwd"?}`, then binary input, `resize`, `escape`. Output binary, `leave` answers Esc at the prompt, `exit {code}` at shell end. Runs `$SHELL -l` with the backend's environment (`pty_process.terminal_environment`). A backend restart ends every shell.
- Esc: the backend decides (`PtyProcess.is_in_foreground`): at the prompt it answers `leave`, otherwise the program gets it; a second Esc within 400 ms hides anyway.
- Disconnect: SIGHUP to the process group, SIGKILL after 2 s (`PtyProcess.close`). Gate: host access via `is_allowed_websocket`.

## Jupyter notebooks

Rules and the adapter exception: `AGENTS.md`, "Jupyter notebooks". All routes need host access. `routes/jupyter.py` only maps errors (503 unavailable, 502 invalid shape; Pydantic models validate what `lib/jupyter.py` reads).

- **Frontend shell**: `.ipynb` opens in `features/notebook/` (`NotebookWorkbench`, dynamic, keyed by tab and `useEditorPath` generation so a rename never remounts). Its Pi chat is a `document` session linked to the notebook's absolute path, running in the mount root holding it (else its folder), like its kernel.
- **Jupyter service** (`src/backend/jupyter_service.py`): one process started by `run-backend.sh` before the workers on Unix socket `GIGACHAD_JUPYTER_SOCKET` (`~/.local/share/gigachad-bot/run/jupyter-<profile>.sock`, 0700 folder). It alone holds `JupyterHost` (server, token, kernel map) so all workers see the same kernels. Workers keep the browser checks and call it through `lib/jupyter_service.py` `RemoteJupyterHost` (down: 503). Kernels survive backend reloads; the script stops all when backend, C4 or the service exits.
- **Server**: `JupyterHost` starts `python -m jupyter_server` from the app venv on a free loopback port at first use, restarts it after death. Config files ignored; runtime files in a temp folder; kernelspecs and `server.log` in private `jupyter_home()`.
- **Token**: generated by the service, set in server memory by the bootstrap, never in env, argv or a kernel-readable file. The proxy drops the browser's `token`, `_xsrf`, cookies, `Authorization`, `Origin`, `Referer` and identity headers (`browser_request_headers`, in worker and service) and injects its own.
- **Proxy** `/api/jupyter/server/{path}` (HTTP via `reverse_proxy.streaming_response`, WebSocket for kernel channels):
  - Allowlist `is_proxied_path`: `api/kernels`, `api/kernelspecs`, `api/sessions`, `api/events`, `api/status`, `api/me`, kernelspec resources; else 404 (no `api/contents`, no terminals).
  - Reads `GET`/`HEAD`; writes only kernel `DELETE`/`restart`/`interrupt` and session `POST`/`PATCH`/`DELETE`; else 405.
  - Sessions must name a kernel this host started (`check_session`); kernels start only via `POST /api/jupyter/kernel`.
  - HTTP routes depend on `require_trusted_request` (`websocket_origin.py`): writes need a trusted `Origin`, reads without `Origin` must not be cross-site per `Sec-Fetch-Site`. WebSocket: `is_allowed_websocket`.
  - Shared plumbing: `src/backend/reverse_proxy.py` (`streaming_response`, `forward`, also used by the LearnHarness gate), errors from `lib/upstream.py`.
- **Kernels**:
  - `POST /api/jupyter/kernel {path, name}` -> `{kernel_id, name}`. The path is resolved through `FileRoots` once (`notebook_path`) and keys the kernel map. Cwd: the mount holding the notebook (`FileRoots.root_of`), else its folder. Same path and name reuse; another name replaces. `DELETE /api/jupyter/kernel?path=` ends it (204, idempotent).
  - `GET /api/jupyter/kernelspecs?path=`: discovered specs plus a generated `venv-<hash>` per `.venv` in the notebook's folder or ancestors up to `$HOME` (written under `jupyter_home()`). Default: `metadata.kernelspec.name` if listed, else nearest venv, else app `python3`. A venv without `ipykernel` (detected by file, never running its python) is `needs_ipykernel`; first start runs `uv pip install ipykernel` (180 s). The pick is saved as `metadata.kernelspec`.
  - Restart of a `dead` kernel or a failed restart starts a new kernel of the same spec via `POST /api/jupyter/kernel`.
  - Document routes call `move_kernel(old, new)` after rename/move and `shutdown_kernels_under(path)` after delete; an unreachable service is logged and ignored.
  - `TabManager` announces `announceFileClosed(path)` when no tab holds a file; `useKernelShutdown` deletes `.ipynb` kernels. Tab switch or reload never shuts down; a remount reattaches by path and name. The server culls unconnected kernels after 2 h idle (checked every 60 s).
- **File sync** (the file API owns load and save; the library only edits the shared model):
  - `useNotebookFile(path)` = `useRootFileDocument` with a notebook codec, saving 800 ms after the last change (`save()` for Ctrl+S, pending changes saved on teardown).
  - `serializeNotebook` keeps the file's style (1-space indent, key order, trailing newline, `orig_nbformat`), strips injected `metadata.editable` unless present in the source, and skips writes equal to disk (also ignoring its own watch echo).
  - Disk changes go through `applyIncomingNotebook(model, base, incoming)` / `diffNotebook`. Policy "merge theirs" per field: editor-only changes stay, file or both changes take the file. Cells match by id (by position when ids are missing or repeated); file deletions delete, editor-only additions stay, the file's order wins; a type or attachments change replaces the cell.
- **Neovim beside the notebook** (per device): `WorkbenchFrame` with Neovim and the notebook, both kept mounted. The Yjs model in the browser is the single source of truth; Neovim holds one projection.
  - Percent text (`features/notebook/notebookText.ts`): marker lines `# %%`, `# %% [markdown]` (or `[md]`), `# %% [raw]`; cells joined by one blank line, trailing `\n`, empty notebook `""`, no outputs; markdown/raw lines prefixed `# ` (`#` when empty). Marker regex `^# %%(\s.*)?$` (Lua `is_marker` agrees). Text before the first marker is a code cell; a code line matching the marker splits its cell; trailing blank lines of code cells are lost on round trip. `cellAtLine(text, line)` maps a 1-based line to a cell.
  - Text to model (`applyNotebookText`) matches cells (exact type+source, then similar lines, then position; model cells by synthetic `cell-<index>` keys because `toJSON` drops ids below nbformat 4.5) so matched cells keep id, metadata and outputs; applied through `diffNotebook` ops (`notebookSync.ts`).
  - Text store (`notebookTextStore.ts`, one per Neovim session key, keyed `<tab>:<generation>`, path `<notebook path><extension>`; `state()` null while unbound, else `{extension, text}`, extension from `metadata.language_info.file_extension` when it matches `^\.[A-Za-z0-9]{1,8}$`, else `.py`): pushes model changes to Neovim 150 ms (`NOTEBOOK_TEXT_DELAY_MS`) after the last ydoc update. `save(text)` rejects while unbound and when the model moved since the last handout ("The notebook changed, write again"), so `:w` never reverts an unseen change; else applies and persists immediately. `runAt(line)` runs the cell at that line of the last saved text.
- **Cell highlighting fork**: `features/notebook/syntaxHighlight.ts` passes one factory through the `Notebook` `editorExtensions` prop, kept in a module-level array (the library rebuilds when array identity changes; factory names must be unique). The prop exists only in the fork tarball (upstream PR https://github.com/datalayer/jupyter-ui/pull/515). Once released on npm: set the npm version, drop the URL, refresh the lockfile.
  - Fork release runbook (repo `~/git/jupyter-ui`, remotes `origin` upstream and `fork` patrickab, Node >= 20):
    ```bash
    git fetch origin && git checkout feat/notebook-editor-extensions && git rebase origin/main   # skip once the PR is merged
    git push --force-with-lease fork feat/notebook-editor-extensions
    V=$(node -p "require('./packages/react/package.json').version"); N=1                         # N = fork counter
    git checkout -B release/$V-gc.$N feat/notebook-editor-extensions && npm install              # slow, run as a supervised process
    cd packages/react && npm version $V-gc.$N --no-git-tag-version && npm run build:lib           # slow
    mkdir -p /tmp/fork-pack && npm pack --pack-destination /tmp/fork-pack && cd ../..
    git add packages/react/package.json && git commit -m "chore(release): $V-gc.$N"
    git push -u fork release/$V-gc.$N && git tag v$V-gc.$N && git push fork v$V-gc.$N
    gh release create v$V-gc.$N /tmp/fork-pack/datalayer-jupyter-react-$V-gc.$N.tgz -R patrickab/jupyter-ui --title "$V-gc.$N" --prerelease --notes "@datalayer/jupyter-react $V plus the editorExtensions Notebook prop. Tarball only."
    ```
    Then `cd src/frontend && npm install https://github.com/patrickab/jupyter-ui/releases/download/v$V-gc.$N/datalayer-jupyter-react-$V-gc.$N.tgz` and restart the dev app (`run.sh` runs `npm ci`).

## Workbench (Markdown and LaTeX)

```text
open .md/.tex -> POST /api/workbench/prepare -> {path on disk, cwd = WORKBENCH/<kind> (mounted LaTeX: its project folder), kind}
  -> Workbench (WorkbenchFrame): Neovim (useWorkbenchEditor) | renderPane | Pi drawer (WorkbenchChat, cwd)
LaTeX: GET /api/latex/project?path= -> {folder, root, candidates} -> build root, chat keyed by folder
any writer: WS /api/files/watch -> changed path -> onFileChange -> useFollowFiles -> setContent(text, path) -> Neovim merge
other device: POST /api/fileroots/file -> change event (key = absolute path) -> same onFileChange listeners
project folder: the same watch on the folder -> LaTeX build
```

- **Routing**: `.md` project and mounted files and `.tex .bib .sty .cls` (`workbenchKindOf`) open as `workbench` items when Neovim is enabled and the login has host access (`opensInWorkbench`), else `TextDocumentEditor`. Notes never (prepare: 400). A LaTeX file inside an open workbench's `texFolder` opens as another buffer of that tab. Images open as `document` items via `RawFileView` (`GET /api/fileviewer/raw`), in a new tab when opened from a LaTeX workbench.
- **Prepare** (`routes/workbench.py`): a stored document moves to `WORKBENCH/<kind>/<slug>/<stem>/<name>` via `document_service.move_to_folder` (references rewritten, row deleted, 409 on conflict), only after `roots.contains` confirms readability (else 403, nothing moved). Absolute paths are checked with `resolve_allowed`. Not idempotent for keys: the frontend dedupes per path (`usePreparedWorkbench`) and takes the returned path. `POST /create` writes a new `.tex` (`latex.NEW_DOCUMENT`) or empty `.md` there and lists it; an existing folder is 409.
- **Prompts**: `templates/workbench/{latex,markdown,pdf,canvas,notebook}.md`, rendered by `lib/workbench/prompts.py` (`{{TEMPLATES}}` -> absolute template folder), appended to Pi's system prompt on every create and resume. The chat runs the user's full Pi unless a tool allowlist is passed. A new session's first message carries a lead: `File: <path>`, `Project: <folder>, root: <file>`, or `Canvas: <name>` (relative to cwd when inside). The LaTeX prompt tells the agent that the root builds, other `.tex` start with `% !TEX root = …`, images go in `imgs/` without `\graphicspath`, and the PDF rebuilds on any change so it never compiles.
- **Pi chats**: `POST /api/workbench/chat {path, session_id?, new?}` opens a `document` session for an absolute file, LaTeX folder, notebook or canvas key. Cwd: kind directory (Markdown, PDF, canvas), project folder (LaTeX), mount root else folder (notebook). A linked session pi-host lost (404) is replaced. `GET /chats?path=` lists links with `SessionInfo`s; `POST /chat/informed` clears `moved`. PDF chats answer `pdf_text`: `{state:"ready", path}` (Markdown mirrored to `Documents/Mineru/<stem>.md`) or `{state:"pending"}`.
- **Links** (`lib/workbench/links.py`, `SessionLinks`): each key (absolute Markdown/notebook path, LaTeX project folder, canvas key) maps to `{active, sessions}`; a bare stored id reads as one session. Opening an unlinked folder adopts links of LaTeX files inside it (root file's session active). A relocation changing a key's path marks its sessions `moved` (`workbench.links.owe_new_paths`), so the next message carries the lead again. Changes are read-modify-write under a process lock with up to 10 retries; `POST /chat` holds a per-key lock. Two first opens on different workers can still start two sessions (later one active).
- **LaTeX projects**: a root `.tex` has `\documentclass` before `\begin{document}` (comments ignored). `latex.project_of(path, ceiling=roots.ceiling_of(path))` never reads above the ceiling:
  1. `% !TEX root = <path>` in the first 5 lines (case-insensitive), resolved against the file's folder; must exist, be `.tex`, inside the ceiling.
  2. A root `.tex` is its own root.
  3. `main.tex` of its folder.
  4. The only root document of its folder.
  5. Else no root; candidates are root documents of its folder and the 2 above (not above the ceiling).

  Other files skip 1–2 and take 3–4 from the nearest folder with a root; no candidates. The project folder is the root's folder, else the file's. Routes (`routes/latex.py`, host access, 403 outside roots):
  - `GET /api/latex/project?path=` -> `{folder, root, candidates}`.
  - `POST /api/latex/root {path, root}` writes `% !TEX root = <relative>` as first line of `path` (replacing its old one), records the change and answers the project (`{folder, root, candidates}`); 400 unless both are `.tex` and `root` is a root inside the ceiling.
  - `POST /api/latex/file {folder, name}` creates `.tex .bib .sty .cls` via `DiskFiles` (one segment without a leading dot, else 400; 409 if exists); a `.tex` starts with `% !TEX root` when the folder has a root by rules 3–4, else `NEW_DOCUMENT`.
  - `POST /api/latex/build`: latexmk in the root's folder, output in `.latex-build/`, last good PDF published atomically as `<stem>.<content hash>.view.pdf` (newest three kept, so the range-loading viewer never reads a replaced file); 503 without latexmk. Builds queue per project (never overlap, bursts collapse); a build error names its file (`LatexIssue.file`).
  - Build output (`latex.is_build_output`: `.latex-build/`, `*.view.pdf`, aux files of a sibling `<stem>.tex`) is hidden from trees, lists and the folder stamp (`file_walk`).
- **Promote**: `POST /api/workbench/promote {slug, path}` turns a listed `.tex` directly in `WORKBENCH/latex/<slug>/<folder>/` into its folder entry (`ProjectStore.replace_file`, display name `X.tex` -> `X`); 404 not listed, 400 other shape, 409 already listed; irreversible. `GET /api/documents` lists such a directory as `{mime: "inode/directory", children}`.
- **Rename**: see Project files. `chats.json` keys follow; `useFileRename` calls `announceFileMove({from, to})`, other devices get `moved_to`. Tabs move, and `useWorkbenchEditor` calls `renameFile`, so `NeovimSession.rename` keeps unsaved text. The tab keeps its original path as `WorkspaceItem.sessionKey`, so the workbench and its Neovim session stay mounted.
- **Buffers**: one tab shows every file of a LaTeX project (`WorkspaceItem.texFolder`); a path change under the same `sessionKey` switches buffers. `useWorkbenchEditor` loads each new file from disk once and sends `open_file` (retried every 200 ms until the session exists), keeps a `synced` map per file and serializes `:w` writes. A restart keeps only the first and the shown file.
- **Following**: `useWorkbenchEditor` (`useFollowFiles`) follows every open file and pushes differing text with `setContent`; pushes during session opening wait for the open frame. `useLatexBuild` follows the project folder (excluding build output, `.git`, `node_modules`, scratch files) and queues a build on any change.
- **SyncTeX** (`features/latex/useSynctex.ts`, `lib/synctex.py`; errors: 404 no match, 403 outside roots, 503 without `synctex`; only the latest lookup acts, `hooks/useLatestRequest.ts`):
  - Inverse: `POST /api/latex/synctex {path: root, page, x, y, word?, offset?, occurrence?}` runs `synctex edit` on `.latex-build/<stem>.pdf` (10 s) -> `{path, line, column}`. Beamer reports `\end{frame}`, so `word` refines to the first line of the frame containing it (`synctex.refine_line`, `lib/tex_source.py`); `synctex.word_column` turns `offset`/`occurrence` into a 0-based UTF-8 byte column (0 without a match).
  - Forward: `<leader>v` -> `POST /api/latex/synctex/view {path: root, file, line}` -> `{page, x, y, words}` (inside a frame, asks for `\end{frame}`; `words` from `synctex.line_words`).
  - Slide gap: `POST /api/latex/synctex/slide {path: root, page}` (see Presentations).
- **Present**: `/present?pdf=&role=presenter&channel=` (`features/latex/present/`, `app/present/page.tsx`), audience window on the same `BroadcastChannel`; notes are the PDF's `/Text` annotations; the presenter keeps the PDF it opened with, so a rebuild needs a reopen. Embeds: links with `embed=1` load as iframes (`slideEmbeds.ts`, HTTPS only); frames get `postMessage({type: "deck-shown", shown})` and return unused keys via `postMessage({type: "deck-key", key})`; mirroring state between windows is the framed page's job.
- **Image paste**: on any Neovim surface showing an absolute `.tex`/`.md`, `storeClipboardImage` (`features/editor/clipboardPaste.ts`) posts to `POST /api/workbench/image?path=<shown file>` (multipart `file`): stores `imgs/img-<YYYYMMDD-HHMMSS>.png` via `DiskFiles.create` (400 non-PNG, 413 > 20 MB, 409 exists) -> `{path, reference, changed}`. LaTeX: project folder, `\includegraphics[width=\linewidth]{…}`, adds `\usepackage{graphicx}` to a root lacking it (`latex.ensure_package`). Markdown: next to the file, `![](imgs/…)`. Other paths answer `error: "Images paste only into LaTeX or Markdown files on disk"`.
- **Layout**: `WorkbenchFrame` (`features/workbench/WorkbenchFrame.tsx`) is the one layout for `Workbench` and `NotebookWorkbench` (editor, `renderPane(layout)`, chat drawer; hidden panes stay mounted; `editor: null` means no editor). `renderPane` gets `WorkbenchPane {layout, text, loadError, editor}`, `editor` = `{reveal, send, focus}` or null. A new kind is a `Workbench` wrapper, a prompt file and a `Kind` entry; `pdf` has none (prepare resolves in place, create refuses).
- **Primitives**: `DiskFiles` (`lib/disk_files.py`, `get_disk_files`) creates (exclusive, `FileExistsError` -> 409) and overwrites under the roots and records `document` changes keyed by resolved path; `record` for writers that already wrote; `record_removed(paths)` records deletes. `FileRoots.ceiling_of(path)`: the holding root, else home for host-only paths, else `ValueError`. `naming.segment(name, hidden=False)` is the single-segment name check. Frontend: `lib/paths.ts`, `lib/backend/fileEvents.ts` (`announceFileMove`/`onFileMove`, `announceFileChange`/`onFileChange`, `announceFileRemoval`, `announceFileClosed`), `lib/domain/files.ts` `isNewFileIn`.
- **Follow primitive**: `WS /api/files/watch` (`routes/file_watch.py`, trusted origin plus login): `{type: "watch", files, folders}` replaces the watch set; replies `{type: "changed", path}` or `{type: "refused", path, message}`. `lib/file_watch.py` watches files via their parent folder (atomic saves) and folders recursively (`watchfiles`, 100 ms debounce), skipping build output, `.git`, `node_modules`, editor scratch (`4913`, `*.swp`, `*.swo`, `*~`). Each watch uses its own thread, never anyio's shared pool. `lib/backend/fileWatch.ts`: one ref-counted socket per tab, backoff reconnect, catch-up report after reconnect; `hooks/useFollowFiles.ts` registers paths. Stored keys keep their sync change events.
- **Limits**: writes while the watch socket is down are picked up on reconnect. Users sharing a project slug share its workbench folders. Moving hosts needs the same absolute `GIGACHAD_BASE_DIR` and copies of `~/.pi/agent/sessions` and `~/.pi/agent/factory/sessions`.

## Presentations

LaTeX decks from versioned template sets; editing rules: `templates/latex/AGENTS.md`.

```
templates/latex/
├── render.py                         primitives -> example.tex -> example.pdf -> previews
└── <set>/
    ├── template.toml                 label, files copied into a new document, image macro, primitive groups
    ├── beamertheme<set>.sty, skeleton.tex, STYLE.md
    ├── primitives/<group>/<NN>-<name>.tex   one frame each, line 1 `% <Label>: <purpose>`
    ├── example.tex, example.pdf      generated, committed
    └── previews/<group>/<NN>-<name>.png   generated, committed
```

- **Sets** (`lib/latex_templates.py`, `config.TEMPLATES / "latex"`): folders with `template.toml` and `skeleton.tex` (cached per manifest mtime; malformed manifests logged and skipped). With `primitives/` a presentation set, else a document set (e.g. `letter/`, KOMA `scrlttr2`). `GET /api/latex/template-sets` -> `[{name, label}]`.
- **Style**: shared slide rules in [`templates/workbench/latex.md`](../templates/workbench/latex.md); per set [`presentation`](../templates/latex/presentation/STYLE.md) and [`draft`](../templates/latex/draft/STYLE.md). The theme is the agent's mode: switching changes the root's `\usetheme` and copies the set's `files`. `draft` has no primitives, so the app treats it as plain LaTeX.
- **Detection**: a root is a presentation when its comment-stripped text has `\usetheme[opts]{X}` naming a presentation set (`latex_templates.presentation_theme`). Nothing is stored.
- **Primitives**: `GET /api/latex/templates?root=` -> `{theme, primitives: [{id, group, group_label, label, description, summary, text, preview}]}` (`theme` null for plain roots; `preview` is the PNG mtime or null). `GET /api/latex/templates/{theme}/previews/{group}/{name}.png` serves only listed primitives of known sets (no request path reaches the disk), immutable-cached (URL carries `?v=`).
- **Create**: `POST /api/workbench/create {slug, name, template}` (unknown set: 400, also for Markdown) writes the folder via `TemplateSet.populate` (`main.tex`, the set's `files`, empty `imgs/`), lists it and answers the `main.tex` path; failure removes the new tree. Each project owns its copies.
- **Add slide**: `AddSlide` (`components/ui/CommandPicker.tsx`) sends `{type: "insert_frame", text}`. `NeovimSession.insert_frame` reads `_G.gigachad_cursor_state`; `tex_source.frame_insertion` places the frame after the enclosing `\end{frame}`, below the cursor outside a frame, before `\end{document}` from the preamble or after it; applied only when buffer and `changedtick` are unchanged (one retry, then logged). Templates load once per root; a hand-added `\usetheme` needs a reopen.
- **Slide gap**: with Neovim, clicking between PDF pages calls `POST /api/latex/synctex/slide {path: root, page}`: `synctex.slide_end` takes the most frequent `\end{frame}` among the page's records in the root folder's `.tex` files, or for a chapter slide its `\section` line (moved past directly following deeper headings); 404 none, 403 outside mounts. The line goes to `editor.reveal` without focus and is the next insertion point.
- **Images**: primitive image slots hold mwe placeholders. For presentation roots, `/api/workbench/image` uses `TemplateSet.image_reference` (e.g. `\roundimage{imgs/<file>}`), skips `graphicx`, and Ctrl+Shift+V sends `{type: "insert_image", path, reference}` (`path` = `StoredImage.placeholder_path`, `imgs/<file>`): `tex_source.fill_placeholder` fills the cursor line's first placeholder, else inserts `reference` below (same `changedtick` check).

## Memory, sharing and deletion

| Concern | Contract |
| --- | --- |
| Memory store | The `memory_store` package owns format, paths and ops; gigachad never writes memory files itself. `lib/memory_context.py` picks the root and renders the prompt block. |
| Memory routes | `routes/memory.py` (plain `def`, login's root): `GET /api/memory/list?memory_store=&path=`, `GET /api/memory/read` (`{kind: "memory", description, facts}` or `{kind: "document", text}`; symlinks out refused), `POST /api/memory/apply {memory_store, ops}`. `MemoryStoreError`: 404 for `not_found`/`memory_store_missing`, else 400, detail `{code, message}`. UI: `features/memory/MemoryViewer.tsx`. |
| Memory migration | `scripts/migrate_memory.py --host-login <login>` (`--dry-run`, `--delete-old`) moves old Postgres `memory/*` documents into each root; until run, they are not shown. |
| Share links | `lib/share_links.py`, `routes/share_links.py`. Owner `GET\|POST\|DELETE /api/shares` (`{kind: "document"\|"asset", key}`); `POST` returns the secret once, rotating any previous. Guest `GET /api/shared/{secret}` (`{name, view}`, key private), `/content` (`nosniff`, `no-store`, CSP `sandbox` except PDFs), canvases also `/canvas`, `/canvas/mutations`, `/canvas/stream` on the owner's engine and scope without device. Canvases are `edit`, all else `view` (`ShareTarget.view`/`.access`). Revoking publishes no change; an idle guest stream ends at the next canvas change, before delivering it. New kind: a `Kind`, a `_KINDS` entry, its own referencing column. Frontend: `features/sharing/ShareButton.tsx`, `app/share/page.tsx` (`/share#<secret>`, the secret never in the page request). |

- Deferred deletions use `UndoDeleteContext`: scheduling the next commits the previous, refusal restores it; `pagehide` or hiding commits the pending one, and `DELETE` uses `keepalive`.
- `DELETE /api/documents` answers `removed` (keys and absolute paths gone; unlinked mounts excluded) and records a `delete` change per absolute path. The frontend announces each (`announceFileRemoval`); other devices get the `delete` event. Shared lists drop entries at or inside the path; `TabManager` closes tabs whose item or `texFolder` lies inside (`tabsAfterRemoval`). Tabs are not stored.
- A tab's crumb is `tab.projectSlug`; disk files resolve their project via `owningProject` (`workspaceTree.ts`): listing project, else linked-folder project, else none.
- Removed chat attachments leave model context immediately; undo restores by message identity (an attachment only into its message). A removed pair is final (no undo, [conversations](#conversations-chat-and-pi)).

### Delete cleanup

`lib/references.py` is the one reference model. Each delete records a `Released` (`owned`, released `refs`, deleted `chats`), commits, then calls `References.release`, so a failed delete never loses shared data.

| Delete | Removes | Unless |
| --- | --- | --- |
| Chat, cascade, merge child (`ChatCleanup.after`) | Uploads and sidecars, sandbox state, then library PDF and annotation canvas of each referenced PDF | Another chat, project, held canvas or upload references the PDF name (library) or bytes (annotation) |
| Project (`ProjectStore.delete_project`) | Its chats, stored documents `project/<slug>/document/...`, derived data | Another project lists it, or a chat/held canvas references it |
| Project file (`DELETE /api/documents`) | Reference, file, derived data. A workbench folder `WORKBENCH/<kind>/<slug>/<folder>` goes whole with its chat links (irreversible on disk), the slug folder too when last | As for a project. Notes are deleted outright. Mounted files/folders are only unlinked |
| One chat upload (`DELETE /api/files/chat/...`) | Upload, sidecar, library copy, annotations | As for a chat |
| Canvas | Mutation log and share rows (`ON DELETE CASCADE`) | |

- Held canvases: every `note/` canvas and project canvases reachable from a chat, project `files` or another held canvas. Annotation canvases count PDF bytes of stored uploads, library and document PDFs, and referenced host PDFs (hashed from disk).
- Host-PDF annotation canvases record opened host paths in meta `hosts`; the canvas stays while any recorded file exists with its hash.
- MinerU output is never deleted by a release nor listed by the dry run. Host files and the disk mirror are never written or deleted.
- Post-commit cleanup failures are logged. No automatic orphan purge: `python -m lib.references --dry-run [--database-url URL]` (from `src/`) lists orphans per user, plus uploads of unstored chats (Pi composer uploads keyed `pi:<session>`).
- gigachad stores nothing per Pi session: Pi keeps them in `~/.pi/agent/sessions`, factory tasks in `~/.pi/agent/factory/sessions`.

## Provider usage

- `GET /api/provider-usage` reads pi-host `GET /usage` via `lib/provider_usage.py`: Anthropic subscription quota of Pi's login (`pi-host/src/gateway/usage.ts`), cached 60 s by pi-host, `?refresh=true` bypasses.
- Returns the worst limit per window (`5h`, `7d`). On failure the last good report stands in for up to an hour (reset times aged), else 502. Weekly values never replace missing five-hour limits.

## Production redeploy

- `POST /api/deploy/redeploy` (host access) starts `redeploy.sh` as transient user unit `gigachad-bot-redeploy` via `systemd-run`, surviving the backend restart. 202, or 409 with stderr while one runs or when the start fails.
- The script rebases onto `origin/master` (refusing tracked changes), restarts pi-host when `pi-host/` or Pi changed (never mid-turn), then the backend, learn mode and preview gate. Fetches over HTTPS (no SSH agent). Steps and rollback: [deployment](deployment.md).
