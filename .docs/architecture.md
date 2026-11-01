Generated from 2bfaf7e on 2026-11-01

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
| Startup | Configure the OMP LiteLLM proxy before model calls, apply database migrations, start extraction and change brokers, mirror the workbench omp prompts (`workbench.mirror_omp`, failures logged, never fatal). Shutdown closes clients/pool and stops workers/MinerU children. |
| LikeC4 | Stateless model parsing and syntax-tree operations. Backend launcher starts/stops it and waits for health before FastAPI. Architecture routes return 503 if unavailable, but managed launch fails when startup health fails and tears down/restarts both processes if LikeC4 exits. |
| Development | User-controlled launchers. Agents must not start servers, except through engine-managed factory previews. |

| Listener | Development | Production |
| --- | --- | --- |
| Frontend | `localhost:2999` | Private frontend origin or existing Vercel deployment |
| FastAPI | `127.0.0.1:8001` | `127.0.0.1:8002` |
| LikeC4 | `127.0.0.1:8011` | `127.0.0.1:8012` |
| omp-web | `127.0.0.1:30177` | Same upstream behind identity gate |
| Factory | `127.0.0.1:8090` | Same default, separate production configuration/database |

- `TransferMiddleware` (`backend/http_plumbing.py`) handles every whole `200` JSON/text body without its own `ETag` or encoding: GET gets a weak body-hash `ETag` with `private, no-cache`, so an unchanged reread answers `304` without the body; bodies of 1 KiB and up are gzipped (level 1) for clients that accept it. Bodies over 256 KiB hash/compress in a thread. SSE, streamed proxies, files, ranges and routes with their own validator (`/fileviewer/raw`) pass through. A 304 still runs the handler, it saves transfer, not server work.

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
| OMP, factory, Neovim, workbench, LaTeX builds, learn mode, mounting/cloning roots, directory browser, redeploy | Require host access. A mounted root is readable in full, so mounting is a host action. |
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
| Workbench documents | Disk files under `config.WORKBENCH` (`GIGACHAD_BASE_DIR/workbench`), referenced by absolute path from `project.json`, or a whole LaTeX project folder listed as one entry. Repo files stay where they are. Chat-to-omp-session links: per-user JSON `workbench/sessions.json` (absolute key -> `{active, sessions}`, the key is a Markdown file or a LaTeX project folder). PDFs are never copied: a mounted PDF is read in place, a stored one from its library mirror in Documents. |
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
- Workbench folders cascade: `workbench.move_project_directories` moves each `WORKBENCH/<kind>/<old>` to `<new>` before the transaction (an existing non-empty destination refuses the rename as a conflict, an empty leftover is removed), and the transaction rewrites those absolute path prefixes in project file lists, `workbench/sessions.json` (keys included), canvases and their mutation log and notes. A failed transaction moves the folders back. A crash between move and commit is not recovered.
- `archived` is a catalog flag (`PATCH /api/projects/{slug}`) that hides the project from the sidebar. Archiving Code directories is per device (`localStorage` `archived_code_directories`). OMP session archive/restore goes through omp-web.

### Project file names

- `project.json` `files` lists each referenced key as a plain string, or as `{"path", "name"}` when the project shows it under a display name. `ProjectStore` reads both forms and writes the object form only while a display name is set.
- Every read-modify-write of `project.json` (files, cards, tabs, learn sessions) goes through `ProjectStore._update_project`: it writes against the revision it read and rereads and reapplies on a conflict, up to 10 attempts (parallel uploads), then raises `StorageConflictError`.
- Renaming a file (`POST /api/documents/rename {slug, path, name}`, sidebar double-click renames the row in place: `FileTree` `rename`, `useInlineRename`) renames the file itself, keeping its suffixes unless the typed name already ends with the last one. An empty name, a separator, `.` and `..` are 400, a name in use is 409 (in its folder, or shown by another file of the same project, ignoring case), the same name is a no-op. The file name and every reference always agree, there is no display-name-only rename any more (`ProjectStore.forget_file_names` drops the legacy display names of a renamed key).
  - Stored project file or note (`document_service.rename`): its key moves to the new name inside its own directory through the relocator below. Only notes and `project/<slug>/document/…` keys rename: library `PDFs/…` keys are 400 and the sidebar offers no rename for them, canvases and folders. A library copy `PDFs/<old>` of a stored PDF follows to `PDFs/<new>` (`follow_library_pdf`), MinerU re-extracts under the new name.
  - File on disk (`document_service.rename_on_disk`, an absolute path: workbench file or mounted file): `FileRoots.rename` renames it inside its folder under the same root and identity policy as a write (403 outside, 404 when missing or, in a project, not referenced, 409 when the target exists), then the relocator rewrites every absolute path reference and a failed rewrite moves the file back. The workbench `<stem>` folder keeps its name, it holds the document's folder. A `.tex` drops `.latex-build/<stem>.*` of both names (`latex.discard_build`), so a build never shows the old file's PDF.
  - LaTeX reference cascade (`_follow_in_tex_project`, `lib/latex_references.py`): after a disk rename inside a tex project, `rewrite_references(folder, old, new)` repoints the references to the file in the project's `.tex` and `.bib` files. The folder is the project folder, or the folder of the first root candidate when no root exists. Dot folders, `node_modules` and build output are not searched, and only a `.tex`, `.bib` or graphic can be referenced.
    - Commands: `\includegraphics`, `\input`, `\include`, `\includeonly`, `\bibliography`, `\addbibresource`, resolved against the project folder (not the referencing file's folder). The `% !TEX root` line resolves against its own file's folder. Commented-out text and every other byte stay.
    - A reference without extension stays without one while the extension is unchanged and no other format shadows it, else it names the new file in full. A `./` prefix stays.
    - Each rewritten file is recorded as a `document` change (`DiskFiles.record`) and returned as `changed` in the `POST /api/documents/rename` response. The frontend (`useFileRename`) calls `announceFileChange` for each, so open buffers merge the new text.
    - A failure in the cascade is logged and never undoes the rename, and a file that cannot be read, decoded or written is skipped. A rename stays inside one folder, so only references change, no file moves.
  - Open Neovim sessions, tabs and the omp chat follow, see Workbench.
- `GET /api/documents` and `/all` return the display name as `name`, the project list returns each project's `file_names` (path to display name). The frontend resolves tabs, editor headers, canvas window titles and chat attachment chips through `useDisplayName` in `ProjectContext`, which every rename refreshes.
- Moving a canvas or stored file between projects or notes (`POST /api/documents/move`) keeps the name it is shown under: into a project as display name, into the notes as its key.
- Moves and renames run `PostgresKeyRelocator.relocate` (`project_rename.py`) in one transaction: every exact reference to the key (project file lists, chat attachments, canvases and their mutation log, app links in notes, `workbench/sessions.json` keys and values) is rewritten, then the row is rekeyed. An absolute `old` (a disk rename) only rewrites references, no row exists to rekey. Change rows are recorded, a rename (same-directory move or disk rename) puts the new key in `changes.moved_to` of the old key's row (schema step 0014, also in the NOTIFY payload and the SSE event) so other devices move their tabs from the old to the new key.
- Uploading a project file (`POST /api/documents/write-binary`, sidebar upload, canvas PDF export) stores it as `project/<slug>/document/<name>`. A PDF is also synced into the library (`PDFs/<name>`, Nextcloud mirror) and queued for MinerU, as `POST /api/fileroots/add-to-project` does for mounted PDFs (`attachments.sync_pdf`).
- Moving a stored project file to disk (`POST /api/documents/move-to-folder`, drag onto a Code folder) writes it into a file-root folder under the name it is shown under, never overwriting (409), then relocates every reference to the absolute path and deletes the stored row. From then on it is a mounted file: read from disk, never deleted by cleanup. A failed relocation removes the written file. A PDF is synced and queued as above. Canvases cannot move.
- Sidebar drags carry a kind (`FileTreeItem.dragKind`/`accepts`): stored project files drop only on project roots, `files` folders and Code folders, chats and canvases never on Code folders.

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
- New and transformed ink points are stored at 0.01 units (x/y) and 0.001 pressure (`inkPoint`, `lib/drawing.ts`): capture, selection move/scale, duplicate and cross-canvas drop. Existing strokes keep their digits until transformed.
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
- Reserve the 15vh preview cap offscreen, allowing short answers to shrink when activated. Expanding a turn renders its full content and plots.
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
- List a finished extraction in the project's files: `POST /api/documents/write-binary` (project upload) and `POST /api/fileroots/add-to-project` (mounted PDF) queue the PDF with an `on_markdown` hook (`materialize`, `sync_pdf`), which calls `document_service.add_extraction` once the Markdown is cached (at once on a cache hit). It stores `project/<slug>/document/<stem>.md` and adds it to `project.json`, keeping an existing document of that name, so an edited copy survives a re-extraction. Images stay in the library, the Markdown keeps its relative `images/` links. The hook writes as no device (`deps.get_extraction_publisher`), so the sidebar of the uploading device reloads the project's files on the `chat/<slug>/project.json` change event too. The queue deduplicates by name and content: the same PDF queued twice at once lists its Markdown only for the first project.
- Keep pasted images, drawings, canvas images, and architecture source attachments as snapshots. Reattach for later changes.
- Use `MINERU_SERVER_URL` for external OCR. Without it, the backend can spawn local MinerU per parse.

### PDF viewer, annotations and assistant

- `features/editor/pdf/AnnotatedPdfViewer.tsx` draws pen/highlighter strokes anchored to pages (page units, margins allowed), reusing canvas ink, history and collaboration. Save failures show as unsaved. The PDF starts loading alongside the annotation key request, ink joins the same viewer once the key arrives.
- `GET /api/fileviewer/raw` sends `ETag` with `Cache-Control: private, no-cache`: a reopened, unchanged file answers 304 from metadata (asset `sha256`, document version and `updated_at`, host file stat). Stored files serve single byte ranges sliced in Postgres, so pdf.js fetches the pages it renders first. The CORS middleware exposes `Accept-Ranges` and `Content-Range` for split-origin deployments. `POST /api/fileviewer/annotations` hashes without loading the PDF: assets and documents both keep a stored `sha256` column.
- `PdfAssistant.tsx` chats about a PDF in the tab's file panel. With host access the shell passes `renderChat` = `OmpFileChat`: an omp session per PDF in `WORKBENCH/pdf` with the PDF prompt, reading the file itself (see Workbench). Otherwise the local chat's system prompt embeds the MinerU Markdown, only for mounted (`/…`) and library (`PDFs/…`) PDFs, polled every 5 seconds while extraction runs. Selection questions add the selected passage and page. History lives only as long as the component, it is not a saved chat.

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
- Let `run-omp-web.sh` serve a detached worktree at `GIGACHAD_OMP_WEB_REF`, with the overlay as `custom/` and the small `seams.patch` that loads it (mount, `Bun.build` overrides plugin, frame headers). Overrides wrap upstream modules by relative path, overlay CSS is scoped to `html.cf-embed`. It reuses existing listeners, so updates require restart. Allow the frontend's frame origin. The `ompweb host` daemon runs from the same checkout and holds the omp runs across server restarts.

| Owner | Contract |
| --- | --- |
| Backend adapter | Only `src/lib/omp_web.py` knows upstream URLs/response shapes. Pydantic validates responses, `/api/omp/*` maps errors. Browser-side shell code never calls upstream APIs directly. |
| Gate | `src/backend/omp_web_gate.py` checks host identity before proxying the production frame. Keep upstream API knowledge out. |
| Frontend adapter | Only `features/omp/` owns iframe URLs and postMessage details. `OmpProvider` keeps one frame, deep-links the initial session, switches by message without reload/DOM relocation. |
| Shell composer | Send prompts/images/documents, abort, model/thinking changes, `/` and `@` completion. Merge returned drafts. gigachad's `/memorize` shares the `/` menu and runs in the shell (`ShellCommands`), never reaching omp. Lies over the full-height frame with the top bar like a chat, the `insets` message tells the overlay how far to pad the transcript (`embed/insets.ts`, `theme/embed.css`). |
| Slash menu | The overlay reports `composer-commands` state (overlay `embed/slash-commands.ts`): skills from omp-web's `/api/skills` for the cwd (no omp process), other commands once the shell's first `/` of an existing session sent `composer-load-commands` (resumes an idle omp process, the new-session screen never loads them). The shell filters locally (`features/omp/composer.ts`), `/skill:` first drills into the skills in the same menu. `@` files still go through `composer-complete`. |
| Frame dialogs | The overlay reports an open modal dialog (`modal`, `embed/modal.ts`). `OmpProvider` lays `FloatingWindow`'s backdrop over the whole shell and lifts the frame above it, omp-web's backdrop (the shell's scrim, `theme/tokens.css`) covers the frame. A click on it or Escape in the shell sends `close-modal`. |
| Extension select | An extension's `select` (e.g. `/files`) is reported as `select` (state message, request id keyed) and never reaches upstream's panel (overlay `embed/select.ts`). The active tab's composer shows it as a `DraftPicker` in its suggestion menu (`CompletionMenu.tsx`): the draft filters, Enter picks and never sends, Escape cancels, `select-answer` replies with the option or null. The picker takes focus, which the triggering send had dropped by disabling the input. |
| Engine | omp-web owns execution, queueing, shell commands, session creation. Shell does not expose word prediction or busy-run steering/queueing. |
| Navigation | Group project-linked mount sessions under code, others under OMP. Navigate forks/branches, restore archives, route file links through shell file policy. An extension's `Open file: /abs/path` notify (e.g. `/files`, `~/.omp/agent/extensions/files.ts`) opens the path in the file panel instead of showing, overlay `overrides/hooks/useAgentSession.ts`. |

- Keep shared chat rendering/protocol/palette canonical in `packages/chat-ui`. Local and OMP chat share Q&A turns, Markdown/LaTeX, fenced code with Mermaid and Markmap diagrams (each host passes its own `light`/`dark` scheme), source-preserving copy, and role colors. An omp compaction anchors a Q&A card in active-context order: the compaction (what it saved, summary folded beneath, overlay `overrides/components/MessageView.tsx`) asks, the kept messages and the continued run answer.
- Upstream renders the running turn's live tail (`ChatLiveTail`: streaming reply, pending tool rows, pending `!command`) after the committed transcript. Embedded, overlay `overrides/components/ChatWindow-live-tail.tsx` portals it through `components/LiveTail.tsx` into the newest Q&A card's answer region from its first token (inline when the newest turn has no card). `ChatLiveTail` is a #314 seam (PR #332).
- The omp composer sends while the agent runs (`ChatInput` `sendWhileStreaming`): the overlay submits it like omp-web's Enter key, a follow-up or steer per omp-web Settings. Stop shows only with an empty draft. omp-web's run status ("Running …") and queue bar (Edit, Delete, Steer) stay above the shell composer, restyled embedded (`theme/embed.css`) as one card in the composer's column.
- Map the palette to each side's tokens. Shell persists its theme preference and propagates it through the bridge. A theme switch is one animation, the shell's bloom (`lib/themeTransition.ts`), which reveals frames live. Frame owners (`OmpProvider`, `LearnProvider`) take their theme from `useFrameThemeName`, which flips only once the bloom shows: Chrome still paints a cross-origin frame's updates while it holds the page for the switch, the old snapshot then covers them. The overlay applies `set-theme` without omp-web's own view transition.
- Click hints (`packages/chat-ui/src/hints.ts`): `f` outside text entry labels every visible clickable, typing a label clicks or focuses it. A session spans the frame boundary through `hints-*` messages. The document that got `f` hosts the session, the other one collects its visible clickables and draws the labels the host assigns. Shell side: `features/omp/clickHints.ts`, installed by `OmpProvider`. Frame side: overlay `embed/hints.ts`, which skips frame content under the shell's top bar and composer.
- `run-omp-web.sh` installs `packages/chat-ui` into the served worktree on every start (the tarball `run.sh` packed, else its own pack), so overlay and shell always share one commit's package. A changed package also drops webpack's build cache, which takes `node_modules` packages as immutable per version.
- The protocol is unversioned. A running omp-web keeps its old bundle until restarted and silently drops messages it does not know.
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
- Run one `nvim -n` PTY per session through `WS /api/neovim/session`, rendered by xterm.js. The session opens one document, `{"type":"open_file","path"}` adds more. Each document is a private temporary copy in a directory of its own (equal basenames never collide) and its own buffer, kept loaded and hidden while another shows. Switching never saves. Start in the innermost containing mount or home for database documents.
- Route `:w` through RPC to the browser's normal document API, including architecture validation: each buffer has its own `BufWriteCmd` and the browser gets `{"type":"save","id","path","content"}` for the buffer written. Replace the buffer with stored text if rewritten. Leave modified on failure. The temporary copy is not canonical storage.
- The session keeps `base` per document, the text Neovim and storage last agreed on. Browser `{"type":"set_content","content","path"}` (agent or remote edit on disk) addresses one buffer, hidden ones too, and is ignored when `path` is not open. It replaces an unmodified buffer and merges into a modified one with `text_merge.merge_theirs` (`git merge-file -p --theirs`: the incoming side wins conflicting or adjacent lines), keeping it modified. Only changed line ranges are rewritten (`gigachad_set_text`, `vim.diff` hunks), so the cursor stays on its text.
- `:w` of an absolute disk file whose disk text differs from `base` saves the merge of buffer and disk instead, echoing "Saved, merged with the agent's edits". Unreadable or non-UTF-8 disk text or a refused merge fails the save and keeps the buffer modified. `set_text` waits for a running `:w` of its document, `set_content` frames run as ordered tasks so a save's reply is never blocked.
- `ask_keymap` on open installs visual `<leader>a`, which reports `{"type":"selection","path", start_line, end_line, text}` (full lines of the buffer at `path`) to the browser.
- Buffers: `open_file` loads the file at `path` from disk (`disk_file`, UTF-8) into a new private copy when it is not open yet and shows it, else it just shows the open buffer. A file that cannot be read answers `{"type":"open_failed","path","message"}`. `{"type":"switched","path"}` reports the tracked buffer Neovim shows whenever Neovim itself changes it (`:bnext`, `:b`), not after an `open_file`. `{"type":"insert_lines","text"}` inserts lines below the cursor of the shown buffer and puts the cursor on the first one.
- `{"type":"renamed","from","path"}` renames the buffer of `from` in place (`gigachad_rename`), keeping its text, cursor and modified flag, drops its old private copy and moves the path saves and merges use. It does nothing when `from` is not open and logs a refusal when `path` is open already. `useNeovimTerminal` sends it for `renameFile(from, to)` and by itself when `document.path` changes under the same `key`.
- Terminate/remove the copies on `:q`, document close, or disconnect. Unsaved edits are unrecoverable. Preserve the session across preview/fullscreen switches. Terminal focus owns all keys, including Escape.
- Yanks reach the browser clipboard: a `TextYankPost` autocmd notifies over RPC, the socket sends `{"type":"yank"}`, the browser writes it best effort.
- Theme the terminal from the app: xterm's background is the first opaque ancestor background (`--page`), its foreground is `--ink`, its cursor a 30% `--ink` blend. The browser sends `{"type":"background","light"}` on open and on every theme flip, and the backend sets Neovim's `'background'` over RPC once setup finished, so the user's colorscheme reloads its light or dark palette. The palettes themselves live in the user's Neovim config, not this repo.
- Treat as host code execution. Match WebSocket origins to `GIGACHAD_CORS_ORIGINS`, or loopback for development `*`.

## Workbench (Markdown and LaTeX)

```text
open .md/.tex -> POST /api/workbench/prepare -> {path on disk, cwd = WORKBENCH/<kind>, kind}
  -> Workbench: Neovim (useWorkbenchEditor) | renderPane | omp drawer (OmpSurface, cwd)
LaTeX: GET /api/latex/project?path= -> {folder, root, candidates} -> build root, chat keyed by folder
any writer (omp, another agent, a shell, another device): WS /api/files/watch -> changed path -> onFileChange -> useFollowBuffers reads it -> setContent(text, path) -> Neovim merge
other device: POST /api/fileroots/file -> change event (key = absolute path) -> same onFileChange listeners
project folder: the same watch on the folder -> LaTeX build
```

- **Routing.** `.md` project files and mounted files, and `.tex .bib .sty .cls` files (kind `latex`, `workbenchKindOf`), open as `workbench` tab items when Neovim is enabled and the login has host access (`opensInWorkbench`), else in `TextDocumentEditor`. Notes (`note/...`) never do, `prepare` answers 400. A LaTeX file that lies inside the `texFolder` of an open workbench tab does not open a tab: that tab shows it as another buffer (`texProjectTab`, `texBufferShown`), see Buffers.
- **Prepare** (`routes/workbench.py`). A stored project document moves to `WORKBENCH/<kind>/<slug>/<stem>/<name>` through `document_service.move_to_folder` (references rewritten, row deleted, 409 on a name conflict), only after `roots.contains` confirms the target is readable later (else 403, nothing moved). An absolute path is checked with `resolve_allowed` and returned resolved. Prepare is not idempotent for keys: the frontend dedupes it per path (`usePreparedWorkbench`, StrictMode-safe) and replaces the tab item's path with the returned one. `POST /create` writes a new `.tex` (`latex.NEW_DOCUMENT`) or empty `.md` there and lists it.
- **Prompts.** `src/lib/workbench/omp/{latex,markdown,pdf}.md` and `config.yml` are the source of truth, mirrored to `WORKBENCH/<kind>/.omp/{SYSTEM.md,config.yml}` on every start (only changed bytes written). Sessions start in `WORKBENCH/<kind>` because omp reads `config.yml` only from the cwd's `.omp/` but `SYSTEM.md` from the nearest ancestor: that gives the kind's prompt without omp's coding prompt and with user/project `AGENTS.md` and project `CLAUDE.md` disabled. The first plain message of a new session carries a lead (`fileLead`, `projectLead`): Markdown `File: <path>`, LaTeX `Project: <folder>, root: <file>` (just `Project: <folder>` without a root). The folder and a Markdown path are relative to the cwd when inside it, else absolute, the root is relative to the folder. The LaTeX prompt says the root builds, every other `.tex` starts with `% !TEX root = <relative path to the root>` (new ones too), images go in `imgs/` and are referenced relative to the project folder, never `\graphicspath`, and the PDF rebuilds on any file change so the agent never compiles.
- **Sessions.** `GET/PUT /api/workbench/session` keep every omp session of a chat key and its active one (`{session_id, session_ids}`): a session id becomes active and is listed once, `null` starts a new session and keeps the list, a bare id stored earlier reads as one session. The key is the absolute path of a Markdown file, or the absolute project folder for LaTeX, so every file of a project shares one chat. A `GET` for a folder without a link adopts the links stored under LaTeX files inside it (any depth): their sessions merge in stored order, the active one is the root file's (`latex.project_of`) else the first file's, `moved` merges, the file keys go and `sessions.json` is written once. That carries per-file chats from before project folders over. `features/omp` owns the primitive: `useFileChat` (view, adopt, reset, select) and `OmpChatPanel` (file panel with `ChatSessionMenu`: titles from omp-web's overview without the `File:` or `Project:` lead, newest first, archived ones dropped). A relocation that changes a key's path (file or project rename) marks all its sessions `moved` (`workbench.owe_new_paths`, applied by the relocator to `sessions.json`): the next message of each starts with the lead again, then `POST /api/workbench/session/informed` clears it, so a continued session works on the new path. The tab item's view wins over the active link. Concurrent first links can lose one.
- **LaTeX projects.** A project is a folder with a root `.tex`, a file with `\documentclass` before `\begin{document}` (comments ignored). `latex.project_of(path, ceiling=roots.ceiling_of(path))` resolves the root of any file and never reads above the ceiling (see `ceiling_of` under Primitives):
  1. A `.tex` whose first 5 lines hold `% !TEX root = <path>` (case-insensitive) takes that file, resolved against its own folder. The target must exist, be a `.tex` and lie inside the ceiling, else the comment is ignored.
  2. A `.tex` root document is its own root.
  3. `main.tex` of its folder.
  4. The only root document of its folder.
  5. Otherwise no root. The candidates are the root documents of its folder and of the 2 folders above, never above the ceiling.

  Other files (`.bib`, `.sty`, `.cls`, images) skip 1 and 2, take rules 3 and 4 from the nearest of those folders that has a root, and have no candidates. The project folder is the root's folder, else the file's. Routes (`routes/latex.py`, host access, 403 outside the roots):
  - `GET /api/latex/project?path=` answers `{folder, root, candidates}`.
  - `POST /api/latex/root {path, root}` writes `% !TEX root = <relative path>` as the first line of the `.tex` `path` (replacing its old one), records the change and answers the project. 400 unless both are `.tex` files and `root` is a root document inside `path`'s ceiling. The frontend asks for the project of every shown file (`useTexProject`, the last project stays while it loads). Without a root the PDF pane lists the candidates (`LatexRootPicker`, choosing calls `/root` then builds), or says "No root document" when there are none.
  - `POST /api/latex/file {folder, name}` creates a `.tex .bib .sty .cls` file through `DiskFiles` (one segment without a leading dot, else 400, 409 when it exists). A `.tex` starts with `% !TEX root = <relative path>` and a blank line when the folder has a root by rules 3 and 4, else with `NEW_DOCUMENT`. The others are empty. The sidebar offers it on a Code folder that directly holds a `.tex` and on a promoted folder, a name without extension gets `.tex`.
  - Build output (`latex.is_build_output`: inside `.latex-build/`, `*.view.pdf`, and the aux files `.aux .log .synctex.gz .fls .fdb_latexmk .out .toc .bbl .blg .bcf .run.xml` of a sibling `<stem>.tex`) is hidden from every tree, file list and the folder stamp (`file_walk`).
- **Promote.** `POST /api/workbench/promote {slug, path}` turns a listed `WORKBENCH/latex/<slug>/<stem>/<stem>.tex` into its folder: `ProjectStore.replace_file` swaps the entry in place, a display name `X.tex` becomes `X`. 404 when the project does not list the file, 400 for any other shape, 409 when the folder is listed already. There is no way back. `GET /api/documents` lists such an absolute directory as `{mime: "inode/directory", children: <build_tree>}` under its display name, and the sidebar shows it like a mounted folder (files open, rename works). The hover action "Make LaTeX project" sits on those `.tex` rows. Deleting the entry only unlists the folder, the files stay on disk. A project rename moves the folder with the rest (`move_project_directories`).
- **Rename** (`document_service.rename_on_disk`, see Project file names). The disk file moves inside its folder and `sessions.json` is rewritten where a key's path changed (the agent's own context still names the old path). `useFileRename` calls `announceFileMove({from, to})` (`lib/backend/fileEvents.ts`) and other devices reach the same `onFileMove` listeners through the `moved_to` change event. The shell moves the tabs, the sidebar reloads the Code tree and `useWorkbenchEditor` calls `renameFile(from, to)`, so `NeovimSession.rename` renames the buffer in place with its unsaved text and modified flag. The tab item takes the new path and keeps the path it was opened with as `WorkspaceItem.sessionKey`: the workbench stays mounted under that key, `usePreparedWorkbench` stays prepared and `Workbench` keeps its Neovim session key. `useLatexBuild` starts clean for a new root.
- **Buffers.** One tab shows every file of a LaTeX project. `WorkbenchItemBody` keeps the project folder on the tab item (`WorkspaceItem.texFolder`, from `loadTexProject`), and a path change under the same `sessionKey` is a buffer switch, not a remount. `useWorkbenchEditor` starts the session with the first path and loads each later file from disk once, then calls `openFile(path)` (retried every 200 ms until the session exists). It keeps a `synced` map of every open file to the text Neovim and the disk last agreed on, and saves each `:w` to the path the `save` frame names, one write at a time. `onSwitched` (`:bnext`) moves the item path to the shown buffer. A file that fails to load or open goes back to the previous buffer with a notice. A restart keeps only the first and the shown file. Without Neovim there is no buffer, the disk text is what the pane shows.
- **Follow buffers** (`useFollowBuffers`). Every open file is followed through `useFollowFiles` (see Follow primitive). A change of one reads the file and pushes it into Neovim with `setContent(text, path)` when it differs from its synced text, the backend merges it into unsaved edits. A push made while the Neovim session is still opening waits for its open frame (`useNeovimTerminal`), so none is lost. A `document` change event for any open path (other devices, the backend, this device's announced changes) reaches the same listeners.
- **Folder rebuild.** Files the user has not opened (the agent's edits, `.bib` files, images, shell edits) change the PDF too. `useLatexBuild` follows the project folder through `useFollowFiles([], build, [folder])` while a root exists, any changed file in it (not build output, `.git`, `node_modules` or editor scratch files) queues a build. There is no polling and no run-end rebuild: every file the agent writes already reports itself.
- **Images.** A PNG on the clipboard pasted into Neovim is intercepted, text pastes are not. `POST /api/workbench/image?path=<shown file>` (multipart `file`) stores it as `imgs/img-<YYYYMMDD-HHMMSS>.png` through `DiskFiles.create` (400 for a non-PNG signature, 413 above 20 MB, 409 when the name exists) and answers `{path, reference}`. LaTeX stores in the project folder and references `\includegraphics[width=\linewidth]{imgs/img-….png}`, Markdown next to the file with `![](imgs/img-….png)`. A LaTeX root without `graphicx` gets `\usepackage{graphicx}` below its `\documentclass` (`latex.ensure_package`), recorded as a change of the root. The workbench inserts `reference` with `insert_lines` and calls `announceFileChange(path)`, so a mount or promoted folder whose tree does not list the image yet reloads (`isNewFileIn`).
- **Kinds.** `MarkdownWorkbench` renders `LaTeXMarkdown` from the synced text, tool artifacts filling the page. `LatexWorkbench` builds the project's root (not the shown file) on open, and on every change in the project folder through one queue (builds never overlap, bursts collapse) (`POST /api/latex/build`: latexmk in the root's folder, output in `.latex-build/`, last good PDF published atomically as `.view.pdf`, 503 without latexmk). A build error names its file (`LatexIssue.file`) when it is not the shown one. `PdfViewer` swaps a rebuilt PDF in only after the new document drew the pages near the viewport, so it never blanks. A new kind is one wrapper around `Workbench` plus a prompt file and a `Kind` entry. The `pdf` kind has no wrapper: `prepare` resolves the PDF without moving it and `create` refuses it.
- **Primitives.** `DiskFiles` (`lib/disk_files.py`, provider `get_disk_files`) creates (`FileRoots.create`, exclusive, `FileExistsError` maps to 409) and overwrites files under the roots and records the `document` change keyed by the resolved path. Its `record` serves a writer that wrote already. `FileRoots.ceiling_of(path)` is the file root holding the path, else the home folder when only host access reaches the path (host access is limited to home), else `ValueError`. `naming.segment(name, hidden=False)` is the one single-segment name check (`ValueError` for empty, separators, control characters, `.`, `..` and, unless `hidden`, a leading dot). Frontend: `lib/paths.ts` (`isInside`, `relativeIn`, `relativeTo`), `lib/backend/fileEvents.ts` (`announceFileMove`/`onFileMove`, `announceFileChange`/`onFileChange`) and `lib/domain/files.ts` `isNewFileIn`.
- **Limits.** A write made while the watch socket is down is picked up when it reconnects. Users sharing a project slug share its workbench folders. Moving to another backend host needs the same absolute `GIGACHAD_BASE_DIR` (references are absolute) and a copy of `~/.omp` (sessions).
- **Follow primitive.** Every editor where the user writes text follows its file. Backend `WS /api/files/watch` (`routes/file_watch.py`, trusted origin from `backend/websocket_origin.py` plus login): `{type: "watch", files, folders}` replaces the socket's watch set, `{type: "changed", path}` and `{type: "refused", path, message}` come back. `lib/file_watch.py` watches files through their parent folder (atomic saves) and folders recursively with `watchfiles`, debounced 100 ms, without build output, `.git`, `node_modules` and editor scratch (`4913`, `*.swp`, `*.swo`, `*~`). Each watch blocks a thread the watcher starts itself (never anyio's shared pool of 40 that sync routes use), it ends within a polling step of a replaced set or a closed socket. Frontend `lib/backend/fileWatch.ts` keeps one lazy ref-counted socket per tab with backoff reconnect and a catch-up report of every watched path after a reconnect, every `changed` goes through `announceFileChange`. `hooks/useFollowFiles.ts` registers absolute paths and subscribes `onFileChange`. Adopters: the workbench (`useFollowBuffers`, `useLatexBuild`), `useRootFileDocument` (clean editors reload, Neovim ones merge through `onFollowed`), `FilePanel`, `TextDocumentEditor`, `useDiffNeovim`. Stored keys (`project/...`) are not on disk and keep their sync change events.

## Memory and deletion semantics

| Memory stage | Contract |
| --- | --- |
| Scope | Document-backed JSON, not vector search. Atomic memories (`{id, text, topic, scope, ...}`) belong to topics (`{name, summary}`), the complete memories. A `MemoryScope(kind, slug)` names the global profile or one project. Global: `memory/global-topics.json` + `global-profile.json`, seeded with `profile` and `preferences`. Project: `memory/project/<slug>/memory/topics.json` + `memory.json`, no seed topics. Schema step `0010_memory_topics.sql` migrated the former category files and fields. Persistence lives in `memory_store.py`, LLM extraction/regroup in `memory_extract.py` (`MemoryExtractor`), prompt injection in `memory_context.py`. |
| Extract | Manual (`/memorize`, Alt+M), using the stored `memory_model`. Always the global scope, plus the project scope when the conversation has a project. A chat sends its messages. An omp tab reads its session through `GET /api/omp/sessions/{id}/history` (`omp_web.session_history`: active branch with compacted turns, user and assistant text only, no thinking, tool calls or notices), the session id keying the watermark. One call per scope after its `(chat_id, scope)` watermark returns operations (`add`, `update`/`delete` by short fact id) plus summaries for new or outgrown topics. `build_proposal` drops unknown ids, repeated targets, duplicates and no-ops, and gives every new topic a proposal. |
| Review | One screen (`MemoryPanel`): accept, edit, move or deny each operation, rename or summarize new topics (renaming onto an existing topic merges), accept or deny summary revisions. |
| Commit / cancel | `POST /api/memory/commit` applies accepted operations deterministically (stale targets skipped), upserts topics, then advances that scope's watermark, also when every operation was denied. Topics, memories, watermark and pending buffer are written in one `DataStore.apply` transaction. Cancellation discards the review without advancing. |
| Injection | `memory_context.augment_system_prompt` renders every fact of the global scope and, in a project, of its scope under `Persistent Memory Context`, as `- [Topic] text` in stored order. Stored order is the relevance order, index 0 first, there is no budget or pinned topic. The extraction prompt lists facts in that order (`m<N>` is 1-based position N), the model may give `position` on add and update, `build_proposal` turns it into `MemoryOperation.index` (0-based, no-op moves dropped) and `MemoryStore.commit` places the fact there, appending when absent or past the end. Manual adds append, the memory window's relevance page (`RankingPage`, `shiftMemory`) swaps neighbours and saves the whole list. |
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
