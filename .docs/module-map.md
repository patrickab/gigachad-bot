Generated from bd63559a on 2026-11-25. The working tree also had uncommitted changes in `.docs/agent-factory.md` and the untracked `features/pi/SubagentMessages.tsx`, which this file lists. The conversation entries (chat, conversation, pi, chat search, turn delete) edited by hand from 78140998 on 2026-11-26 (no `/repomix` run). Corrected by hand on 2026-11-27 after the complexity cleanup (removed files and names dropped, new files added; no `/repomix` run). Edited by hand on 2026-11-27 after the reuse pass (removed the desktop target, listed the new shared pieces; no `/repomix` run).

# Module map

Use this file to locate the owner of a change. Read [system overview](system-overview.md) for concepts and workflows, [architecture](architecture.md) for runtime contracts, persistence, security, and failure semantics. Follow [repository documentation guidance](../AGENTS.md#documentation) for optional operational runbooks.

## Path conventions and source tree

- Resolve paths below from the repository root unless a section explicitly gives another root. In Python, `backend`, `lib`, and `config` resolve under `src/`; frontend `@/` imports resolve under `src/frontend/`.
- Distinguish `src/lib/` (Python services/storage/adapters), `src/frontend/lib/` (browser contracts/utilities), and `src/frontend/packages/chat-ui/` (a separately distributed shared package).
- Read this as a selective ASCII tree: listed leaves are representative real files, not an exhaustive inventory. Omitted siblings are not implied absent.

```text
.
|-- AGENTS.md                         Repository rules; operational documentation index
|-- README.md                         Setup and developer entry guide
|-- PRODUCT.md                        Product direction
|-- CONTEXT.md                        Domain language glossary
|-- .docs/                            Complementary generated context and maintained guidance
|-- pyproject.toml                    Python dependencies, packaging and tool configuration
|-- compose.yaml                      Local PostgreSQL service definition
|-- db/
|   `-- schema/                       Numbered PostgreSQL migrations
|       |-- 0001_postgres_storage.sql Initial persistent storage
|       |-- 0002_change_notify.sql    Change notification trigger
|       |-- 0004_canvas_mutation_log.sql
|       |-- 0005_canvas_shares.sql
|       |-- 0007_model_reasoning_levels.sql
|       |-- 0010_memory_topics.sql    Old memory topics (memory now lives in memory-store files)
|       |-- 0014_change_moved_to.sql  `changes.moved_to` for renames
|       |-- 0015_change_operation.sql `changes.operation` (`write`, `move`, `delete`)
|       |-- 0016_drop_model_reasoning_levels.sql Effort ladders come from the gateway, the cache table goes
|       |-- 0017_factory_tasks.sql, 0020_factory_rows_migrated.sql Old factory rows and their one-time move into the core
|       |-- 0018_repositories.sql    Pi start folders
|       |-- 0019_user_animated_background.sql Per-user display preference
|       `-- 0020_share_links.sql, 0021_share_links_derived_access.sql Share links (secret hash, one per resource)
|-- src/
|   |-- config.py                     Env-derived backend settings (imports nothing from lib)
|   |-- backend/
|   |   |-- server.py                 FastAPI composition, lifecycle (shutdown work in the lifespan, uvicorn keeps its own signal handling) and shared error mapping
|   |   |-- http_plumbing.py          Exception-to-HTTP mapping, SSE responses, inline image decoding, `not_modified` (ETag/304 answer)
|   |   |-- identity.py               Request identity and host-access capability
|   |   |-- websocket_origin.py       `is_trusted_origin`, `is_allowed_websocket`, `require_trusted_request`: origin and host-access gates of WebSockets and changing requests
|   |   |-- sync.py                   PostgreSQL notification broker
|   |   |-- learnharness_gate.py      Identity-checking LearnHarness frame proxy, refuses its credential routes
|   |   |-- jupyter_service.py        Jupyter service: the one process (Unix socket) holding the Jupyter Server, its token and kernel map for all workers
|   |   |-- preview_gate.py           Identity-checking proxy to every factory preview, picked by the port in the path
|   |   |-- reverse_proxy.py          Streaming and WebSocket proxy plumbing shared by the gates and the Jupyter proxy route
|   |   |-- install_deps.py           gigachad-install entry point
|   |   `-- routes/                   HTTP, SSE and WebSocket boundaries
|   |       |-- deps.py               Identity-scoped store/service injection, `resolve_allowed` (path inside the allowed file roots)
|   |       |-- schemas.py            Shared file-route request/response models
|   |       |-- chat.py               Chat generation and tools
|   |       |-- histories.py          Chat persistence, branching and merge endpoints; every chat (`/all`) and full-text search (`/search`)
|   |       |-- documents.py          Logical/project documents and library attachment APIs (over `lib/document_service.py`)
|   |       |-- files.py              Chat uploads and extraction APIs (over `lib/attachments.py`)
|   |       |-- fileviewer.py         Generic text/raw preview APIs
|   |       |-- file_roots.py         Mounts, trees, host browsing, mounted-file read/write
|   |       |-- file_attach.py        Attach mounted files to chats and projects as live references
|   |       |-- file_git.py           Git diffs and worktrees of mounted roots
|   |       |-- architecture.py       Project LikeC4 workspace APIs
|   |       |-- canvases.py           Collaborative mutation and canvas stream APIs
|   |       |-- assets.py             Stored binary serving
|   |       |-- projects.py           Project catalog, kanban and saved tabs
|   |       |-- memory.py             Memory window: list, read, apply ops on the login's memory-store root
|   |       |-- share_links.py        `/api/shares` (owner) and `/api/shared/{secret}` (guest: content, shared canvas)
|   |       |-- models.py             Models, providers, usage, defaults and prompts
|   |       |-- config.py             Browser configuration and capability flags
|   |       |-- ocr.py                Image OCR generation
|   |       |-- neovim.py             Neovim terminal WebSocket: typed control frames in, browser notifications out through one ordered queue, notebook text validation
|   |       |-- terminal.py           `WS /api/terminal/session`: login shell on a PTY, `open` (with the tab's `cwd`)/`resize`/`escape` in, `leave`/`exit` out
|   |       |-- jupyter.py            Kernelspecs, `POST`/`DELETE /api/jupyter/kernel`, HTTP and WebSocket proxy `/api/jupyter/server/` (trusted-request and host-access gates)
|   |       |-- file_watch.py         `WS /api/files/watch`: replaceable watch set in, `changed`/`refused` frames out
|   |       |-- workbench.py          Workbench prepare (move stored doc to disk), create, promote a `.tex` to a folder entry, pasted images, chat-key session links
|   |       |-- latex.py              LaTeX build (`POST /api/latex/build`), tex project resolution, root choice, new project files, SyncTeX search and slide gaps, template sets and primitive previews
|   |       |-- learn.py              LearnHarness session API boundary
|   |       |-- sync.py               Durable change-log SSE
|   |       |-- pi.py                 `/api/pi/*`: repositories, sessions, turn delete, tool-image media (`/media/<sha256>`, `/thumb`), the session WebSocket relay to pi-host (host access)
|   |       |-- factory.py            `/api/factory/call`, `/actions`, `/images`: core pass-through, login tag, revert on remove, preview stop (host access)
|   |       |-- factory_preview.py    `/api/factory/tasks/{id}/preview`: start, stop, status (host access)
|   |       |-- deploy.py             Host-gated production redeploy trigger
|   |       `-- upstream.py           Host-access dependency and adapter error mapping
|   |-- lib/                          Python domain services, stores and external adapters
|   |   |-- data_store.py             Storage contract, logical keys and errors
|   |   |-- postgres_data_store.py    PostgreSQL text/document implementation
|   |   |-- postgres_pool.py          Process-wide PostgreSQL connection pool
|   |   |-- changes.py                `UserScope`: per-user pool access and change-log rows; `latest_changes` sync replay
|   |   |-- identity_store.py         Tailscale login to user id, device registration
|   |   |-- canvas_store.py           Transactional collaborative canvas mutation authority
|   |   |-- json_docs.py              JSON documents over a DataStore
|   |   |-- asset_store.py            PostgreSQL binary assets
|   |   |-- storage_namespace.py      Canonical database namespaces and key builders
|   |   |-- db_schema.py              Checksum-protected migration application
|   |   |-- chat_store.py             Chat model, I/O, indexing, atomic tree operations, the listing and search of every chat
|   |   |-- project_store.py          Catalog, slugs, kanban and tab persistence, project file references with display names
|   |   |-- document_service.py       `DocumentScope` (notes or a project), document write/move with reference repointing, file rename (stored key move, disk rename)
|   |   |-- chat_cleanup.py           `ChatCleanup`: release uploads, derived data and sandbox state after chat deletion
|   |   |-- references.py             Reference model, post-delete cleanup of unused derived data, `--dry-run` orphan report, `annotation_canvas` (PDF annotation key, recorded host files)
|   |   |-- project_rename.py         Project rename transaction and reference rewriting
|   |   |-- memory_context.py         Login's memory-store root, project memory_store name, chat system-prompt memory block
|   |   |-- share_links.py            `ShareLinks`, `ShareTarget`, `find_grant`: secret-hash links to one document or asset
|   |   |-- user_preferences.py       Per-user display preferences (`users` columns)
|   |   |-- pi_pastes.py              Videos pasted into a Pi composer, stored in `<cwd>/.pi/pastes/`
|   |   |-- model_provider_store.py   Provider catalog and model defaults
|   |   |-- reasoning_levels.py       Reasoning-effort levels from the gateway's `thinking_levels`, send-time kwargs
|   |   |-- prompt_store.py           User prompt storage (read lazily) and seed/block loading
|   |   |-- file_roots.py             Mounted-file roots, path safety, read/write/create, `ceiling_of` (root, else home for host access), tree
|   |   |-- disk_files.py             `DiskFiles`: create/write/record files under the roots with their `document` change
|   |   |-- file_roots_repository.py  Per-user root registry (`vault_roots`)
|   |   |-- repository_index.py       Pi start folders stored in `repositories`, swept against `~` at startup
|   |   |-- file_walk.py              `.gitignore`-aware directory walk and tree, build output hidden
|   |   |-- host_access.py            Host-access reach: home dir minus credential stores
|   |   |-- wiki_links.py             Obsidian wiki-link rewriting
|   |   |-- git_diff.py               Diff listing, patches and file content
|   |   |-- git.py                    One git runner plus `GitError`/`git_text`/`git_ok`
|   |   |-- git_worktree.py           Mounted repository worktree operations
|   |   |-- architecture_workspace.py Disk/database architecture source storage
|   |   |-- architecture_service.py   Architecture writes: apply through the C4 service, commit, conflict retry
|   |   |-- c4_service.py             LikeC4 service client on `UpstreamClient`, typed results
|   |   |-- attachment_materialize.py Shared attachment/PDF materialization
|   |   |-- attachments.py            `ChatUploads`, `attach_host_file`, `kind_of`: chat uploads and live-reference attaches
|   |   |-- document_library.py       Document metadata, storage keys and mounted-path helpers
|   |   |-- mime.py                   `mime_for`: MIME type of a document path, `IMAGE_SUFFIXES`
|   |   |-- extract_queue.py          Background PDF extraction queue (callers pass the persist callback)
|   |   |-- mineru.py                 MinerU client/process lifecycle
|   |   |-- image_paths.py            Chat image payload resolution
|   |   |-- toolcalling.py            Generic tool-turn runtime
|   |   |-- tools.py                  Application tool bindings
|   |   |-- tool_context.py           Per-call tool context and browser tool options
|   |   |-- web_search.py             Brave query planning and evidence retrieval
|   |   |-- research.py               GPT-Researcher adapter (gateway chat models, Ollama embeddings)
|   |   |-- sandbox_service.py        Workspace/output orchestration
|   |   |-- sandbox_store.py          Durable sandbox metadata
|   |   |-- agent_sandbox_adapter.py  Native sandbox runner adapter
|   |   |-- sandbox_plot.py           Plot generation
|   |   |-- pi_host.py                pi-host URL/response-contract owner: sessions, WebSocket, gateway URL and token, factory core calls
|   |   |-- pi_models.py              Pi model catalog for local chats, stored selector -> gateway id (`gateway_model_id`)
|   |   |-- ollama_models.py          Installed Ollama chat models, synced into Pi's `models.json` as provider `ollama`
|   |   |-- pi_media.py               480px WebP thumbnails of Pi tool-result images (Pillow, in-memory cache)
|   |   |-- completions.py            Every backend model call: `openai` SDK on pi-host's gateway, message and image helpers
|   |   |-- factory_migration.py      One-time move of the old `factory_repo_settings`/`factory_tasks` rows into the factory core, per row (`migrated_at`)
|   |   |-- factory_revert.py         Dry-run and `git revert` of a merged task's commit before its removal
|   |   |-- factory_images.py         Images attached to task requests/messages, stored outside every repo
|   |   |-- factory_preview.py        Task previews: processes, ports, throwaway database, state files, gate URLs
|   |   |-- upstream.py               Shared server-to-server request/validation plumbing, `UpstreamClient`
|   |   |-- async_utils.py            `aiter_in_thread`, `KeyedLocks` (one `asyncio.Lock` per key)
|   |   |-- naming.py                 `slugify` for project slugs, `segment` single-segment name check
|   |   |-- ttl_cache.py              Lock-free single-value TTL cache
|   |   |-- provider_usage.py         Pi subscription usage reader/cache (pi-host `/usage`)
|   |   |-- neovim_session.py         Neovim PTY/process lifecycle, buffers per session, `base` tracking and merges, one typed backing per document (`_PrivateCopy`, `_NotebookText`)
|   |   |-- pty_process.py            `PtyProcess` (child on a PTY, output queue, resize, foreground check, hangup on close) and the PTY helpers Neovim shares
|   |   |-- file_watch.py             `FileWatcher`: `watchfiles` over files (via parent folder) and folders, own threads, no build output, `.git`, `node_modules` or editor scratch
|   |   |-- neovim_rpc.py             Neovim msgpack-RPC client and Lua bridge (saves, yanks, selections, buffer switches, hunk apply, notebook text buffer)
|   |   |-- jupyter.py                Only Jupyter knowledge: supervised private `jupyter_server`, kernelspecs, kernel per path, proxy allowlist (runs in the Jupyter service)
|   |   |-- jupyter_service.py        `RemoteJupyterHost`: the workers' client of the Jupyter service, `JupyterHost`'s methods over its Unix socket
|   |   |-- process_owner.py          Host children never outlive the backend: `owned_environment` mark, `stop_owned` at shutdown, `reap_orphans` at startup
|   |   |-- text_merge.py             `merge_theirs`: `git merge-file --theirs`, the incoming side wins
|   |   |-- latex.py                  latexmk build, TeX error parsing, atomic content-named `.view.pdf` publish, tex project root resolution (`project_of`, `set_root`), build-output test, `NEW_DOCUMENT`
|   |   |-- latex_references.py       `rewrite_references`: repoint `\input`, `\include`, `\includegraphics`, `\bibliography` and `\addbibresource` references of a LaTeX project after a rename
|   |   |-- latex_templates.py        `TemplateSet`s of `templates/latex/`: presentation detection (`presentation_theme`), primitives, document population, image macro
|   |   |-- synctex.py                `synctex edit`/`view` wrappers: inverse and forward search, beamer frame refinement, `slide_end` for slide insertion
|   |   |-- tex_source.py             Plain-text TeX helpers: comment stripping, beamer frame bounds and new-frame insertion point, image placeholder fill
|   |   |-- learnharness.py           LearnHarness URL/response-contract owner (sessions adapter)
|   |   |-- learnharness_config.py    `python -m lib.learnharness_config write-config`: LearnHarness's custom provider (pi-host's gateway) snapshot, tiers and auth binding
|   |   |-- workbench/                Workbench kinds, folders, rename folder moves, Pi chat prompts (`prompts.py`), `links.py` (sole owner of `workbench/chats.json`)
|   |   `-- prompts/                  Internal, non-user-facing prompts
|   |       `-- non_user_prompts.py
|   |-- c4/                           Separate stateless TypeScript LikeC4 service
|   |   |-- server.ts                 Loopback JSON API
|   |   |-- workspace.ts              Parse a workspace and index its syntax-tree nodes
|   |   |-- ast.ts                    Narrow LikeC4 syntax-node types and guards
|   |   |-- fqn.ts                    Element FQN helpers
|   |   |-- payload.ts                JSON model projection for the UI
|   |   |-- ops/                      Surgical source edits: `operation.ts` types, `plan.ts` planner table, `plan-*.ts` per op family, `apply.ts` batch apply
|   |   |-- layout.ts                 Saved layout snapshots/pins
|   |   |-- operations.test.ts        Service operation regressions
|   |   `-- package.json              Service dependencies and scripts
|   `-- frontend/                     Next.js application root (not repository root)
|       |-- AGENTS.md                 Frontend-specific framework guidance
|       |-- package.json              Frontend scripts/dependencies
|       |-- eslint.config.mjs         Enforced layer/feature import boundaries
|       |-- next.config.mjs           Next.js/export configuration, copies the pdf.js worker and libavoid wasm into public/
|       |-- vitest.config.ts          Browser unit/component test configuration
|       |-- app/
|       |   |-- layout.tsx            Root providers, font and styles
|       |   |-- present/page.tsx      Presenter/audience popup route of a built LaTeX deck (`features/latex/present/`)
|       |   |-- share/page.tsx        Share link page `/share#<secret>`: shared canvas or sandboxed content, full page
|       |   |-- page.tsx              Client shell entry
|       |   |-- globals.css           Shell theme and application styles
|       |   `-- _components/
|       |       |-- App.tsx, AppEntry.tsx, AppEntry.dev.tsx App root with a fullscreen error boundary; hydrated build entry, browser-only `next dev` entry
|       |       |-- AppShell.tsx      Cross-feature composition and workspace navigation
|       |       |-- TabContent.tsx    Tab layout: sidebar, surface, overlays and modals, per-tab error boundary
|       |       |-- TabSidebar.tsx    The tab's Sidebar apart from its chat: memoized, so stream tokens never re-render navigation
|       |       |-- WorkspaceSurface.tsx Per-mode table of top-bar controls and surface body
|       |       |-- useTabChat.ts     Tab chat hooks, kept mounted in every mode
|       |       |-- deepLink.ts       Preview/open-target parsing
|       |-- features/                 Domain-owned interactive surfaces
|       |   |-- chat/                 Turns, attachments, persistence and streaming (`useChatStream` over `lib/backend/tokenPacer.ts`)
|       |   |   |-- ChatContainer.tsx Chat on the shared ConversationColumn; settles a reveal and paints a search hit
|       |   |   |-- ChatPair.tsx      A pair's label, collapsed preview and open body (`renderPreview`/`renderBody`)
|       |   |   |-- useChatTurns.ts   Send, pair delete and edit (cut, save, resend), attachment removal
|       |   |   |-- ChatPicker.tsx, useChatPicker.ts Chat picker (Alt+C): every chat by project, full-text search with a row per hit
|       |   |   |-- toolRun.tsx, toolRenderers.tsx Chat tool calls as conversation tool rows: record -> `{call, run}` adapter, chat renderers and display
|       |   |   |-- attachments.ts
|       |   |   |-- composer/         Input, completions, drawings and attachment chips (useDraftAttachments, dictation via hooks/useSpeechInput)
|       |   |   |-- messages/         Question/answer messages, rich output rendering
|       |   |   `-- preview/          Attachment previews
|       |   |-- conversation/         What the chat and Pi share
|       |   |   |-- ConversationColumn.tsx The turn column (scroll, follow the end, paging, open state, keyboard focus, lazy previews, reveal, width)
|       |   |   |-- TurnQuestion.tsx  The turn card's question (copy, edit in place, delete, confirmations) and answer (copy, branch)
|       |   |   |-- WideModeToggle.tsx, useTextHighlights.ts Wide mode button; search-hit painter (CSS Custom Highlights)
|       |   |   `-- tools/            Tool-call rows, process details, diff view and the tool renderer registry (rows ported from omp-web, MIT)
|       |   |-- editor/               Documents, viewers and terminal editors
|       |   |   |-- DocumentEditor.tsx Routes a document path to its viewer or editor (image, architecture source, canvas, text), `WorkspaceSurface` shows notebooks in `NotebookWorkbench` instead
|       |   |   |-- TextDocumentEditor.tsx Markdown/text save and preview owner (EditorShell, EditorViewSwitch, useEditorView)
|       |   |   |-- useRootFileDocument.ts Load, dirty, remote reload and chained saves of one file, codec-aware
|       |   |   |-- FileViewer.tsx    File and document picker on `FloatingPicker`: fuzzy search, folder drill-down, file preview window
|       |   |   |-- FilePanel.tsx     Per-tab side panel slot (Neovim, PDF assistant), one owner at a time
|       |   |   |-- NeovimEditor.tsx  `useNeovimTerminal`: terminal hook of every Neovim surface, browser clipboard, `NeovimStopped`
|       |   |   |-- ShellTerminal.tsx `ShellTerminals`: one floating login shell per tab (host access only)
|       |   |   |-- terminalSlot.ts   `TerminalHandle`: a running xterm whose host element survives React remounts, `POLICY_VIOLATION`
|       |   |   |-- terminalTheme.ts  xterm colors and fonts from the app theme, shared by Neovim and the shell
|       |   |   |-- clipboardPaste.ts Browser clipboard read for a paste, and the screenshot store (`imgs/`, `announceFileChange`) behind it
|       |   |   |-- pdf/              AnnotatedPdfViewer (ink over pages), pdfInk.ts, PdfAssistant (local MinerU-grounded chat, or the owner's `renderChat`)
|       |   |   |-- architecture/     LikeC4 graph/source editing
|       |   |   |   `-- ArchitectureWindow.tsx
|       |   |   |-- canvas/           Canvas entities, collaboration, history and camera
|       |   |   |   |-- CanvasEditor.tsx  Wiring and layout over the hooks below, plus the nested canvas window body
|       |   |   |   |-- exportCanvasPdf.ts `exportCanvasPdf`: canvas to a PDF (lazy `pdf-lib`)
|       |   |   |   |-- useCanvasPointer.ts, useCanvasTextNotes.ts, CanvasContextMenu.tsx
|       |   |   |   |-- ink/useInkSurface.ts Draw/erase gesture, straighten, long press; canvasInk.ts shared with the PDF viewer
|       |   |   |   |-- selection/        useRubberBand (tool, lasso/screenshot band), useCanvasDrag (drag sessions), useCanvasSelection
|       |   |   |   |-- attachments/      useCanvasAttachments, useCanvasInsert, useCanvasScreenshot, windows
|       |   |   |   `-- useCollaborativeCanvas.ts
|       |   |   `-- diff/             Diff display and review drafts
|       |   |       |-- DiffViewer.tsx Layout root over the hooks below
|       |   |       |-- useRootDiff.ts, useReviewSession.ts, useDiffNeovim.ts, useDiffScroll.ts, useDiffLayout.ts
|       |   |       |-- FileSection.tsx, ReviewAnnotations.tsx, diffTree.tsx, diffSelection.ts
|       |   |       `-- review.ts
|       |   |-- navigation/           Sidebar, tabs, commands, mounts and worktrees
|       |   |   |-- Sidebar.tsx       Composition of the hooks below
|       |   |   |-- workspaceTree.ts  Pure Projects and Code tree builders, promoted LaTeX project folders, new-file action
|       |   |   |-- useWorkspaceDocuments.ts, useCodeMounts.ts, useProjectActions.ts
|       |   |   |-- TabManager.tsx    Tab state provider, live preview of a background tab scaled into the picker card; tabs.ts pure helpers, useWorkspaceSwitch.ts, TabPicker.tsx (tab picker, opened by Space)
|       |   |   |-- ArchiveSelector.tsx, useArchivedDirectories.ts Restore archived projects/Code directories
|       |   |   `-- useWorktrees.tsx
|       |   |-- pi/                   Pi sessions rendered natively, the only frontend code that knows Pi's protocol
|       |   |   |-- PiTab.tsx         A Pi tab: transcript, composer, top bar; turn delete and edit (`deletePiTurn`)
|       |   |   |-- PiTranscript.tsx  Turns on the shared ConversationColumn: pieces, answers, search jump, held messages, factory view
|       |   |   |-- PiComposer.tsx    Composer: steer/follow-up while streaming, `/` and `@` completion, chips
|       |   |   |-- outbox.ts, HeldMessages.tsx Held (unconfirmed) messages sent during a run
|       |   |   |-- sessionStore.ts, usePiSession.ts, piSocket.ts Attach, replay, commands, extension dialogs over `/api/pi/ws`
|       |   |   |-- subagents.ts, SubagentRoster.tsx Subagent roster above the composer (pi-subagents widget and tool details, M26)
|       |   |   |-- SubagentMessages.tsx `intercom_message` and `subagent-notify` cards, one line that folds open
|       |   |   |-- FactoryNotices.tsx, FactoryTool.tsx Factory notices and `factory` tool calls: their card, else one line
|       |   |   |-- factoryCard.ts, FactoryCard.tsx, factoryView.tsx Card parsing and view, the card/agent-text switch (Alt+V), decision links
|       |   |   |-- ShellNotices.tsx  Background shell sessions that ended well left out
|       |   |   |-- renderers.ts      Custom-message renderer registry, re-exports the conversation tool registry
|       |   |   |-- SessionWaves.tsx  The session's factory waves, decisions and cards waiting for a ruling above the composer (`waves.list`, `decisions.list`, `origin`)
|       |   |   |-- usePiSidebar.tsx  Sessions per repository and mount, worktrees under their main repository
|       |   |   |-- PiSessionPicker.tsx Top-bar session picker (Alt+A): sessions by last use, full-text search with a row per hit
|       |   |   |-- PiSessionPreview.tsx A session's transcript read from its file, the picker's preview
|       |   |   |-- PiWorkspacePicker.tsx Composer project selector of an empty chat, a pick makes the first message start a Pi session
|       |   |   `-- tools/            Re-exports of `features/conversation/tools/`
|       |   |-- workbench/            Editor | rendered file | Pi chat drawer primitive
|       |   |   |-- Workbench.tsx     Neovim, quoting, status slot and editor notes on `WorkbenchFrame`, kinds supply `renderPane`
|       |   |   |-- WorkbenchFrame.tsx `WorkbenchFrame`: the one layout of `Workbench` and `NotebookWorkbench`, `EditorNote`
|       |   |   |-- WorkbenchChat.tsx `useWorkbenchChat` and `WorkbenchChat`: the Pi chat drawer shared by `Workbench` and `NotebookWorkbench`
|       |   |   |-- DocumentChat.tsx, useDocumentChat.ts, fileLead.ts Pi `document` session chat per file or LaTeX project folder, session menu, New session (`File:`/`Project:` leads)
|       |   |   |-- FileChat.tsx      File chat for owners that only view the file (PDF, canvas), prepares it first
|       |   |   |-- pages.ts           `EDITOR_PAGE`/`PREVIEW_PAGE`: the panel classes of the Neovim panel and the preview page, shared with `NotebookWorkbench` and the notebook's markdown cells
|       |   |   |-- WorkbenchSplit.tsx Editor/split/preview view and switch, shared page box (`usePaneLayout`, `PANE_TOP_PX`)
|       |   |   |-- usePreparedWorkbench.ts Prepare once per path
|       |   |   |-- useWorkbenchEditor.ts One Neovim session over several buffers: `openFile`, saves per path, renames, every open file follows disk changes into Neovim (`hooks/useFollowFiles`)
|       |   |   |-- NotebookWorkbench.tsx A notebook tab on `WorkbenchFrame`: Neovim on the notebook text, the notebook, the chat drawer
|       |   |   `-- MarkdownWorkbench.tsx Markdown kind
|       |   |-- notebook/             Jupyter notebook kind, the only importer of `@datalayer/*`, `@jupyterlab/*`, `@jupyter/*` and `@lumino/*`
|       |   |   |-- NotebookDocument.tsx, NotebookEditor.tsx Public entry (`next/dynamic`) and the library notebook editor
|       |   |   |-- NotebookToolbar.tsx, useNotebookKernel.ts, serviceManager.ts Run/kernel buttons and kernelspec picker in the top bar's `center` slot, kernel start by path, `ServiceManager` through the backend proxy
|       |   |   |-- useNotebookFile.ts, notebookSync.ts Load, debounced save and follow over `useRootFileDocument`, pure nbformat serialization and cell diff (merge theirs)
|       |   |   |-- useKernelShutdown.ts Kernel shutdown when a notebook's last tab closes
|       |   |   |-- syntaxHighlight.ts Highlight style and `self`/decorator marks, passed through the fork's `editorExtensions` prop
|       |   |   |-- notebookText.ts   Percent text (`# %%`) of a notebook and back, text applied to the model with outputs kept
|       |   |   |-- notebookTextStore.ts `NotebookTextStore`: the bound model's text for Neovim, stale-save check, `runAt`
|       |   |   `-- notebook-theme.css, markdownRenderer.tsx, empty.css App theme through `--jp-*` variables, Markdown cells through `LaTeXMarkdown` on the split view's preview page, stub for an unused library stylesheet
|       |   |-- latex/                LaTeX kind: LatexWorkbench, LatexPdfPane, LatexBuildStatus, DownloadPdfButton
|       |   |   |-- useTexProject.ts, LatexRootPicker.tsx Project of the shown file (folder, root, candidates), root choice
|       |   |   |-- useLatexBuild.ts  Root build queue, rebuilds on any change in the folder
|       |   |   |-- useSynctex.ts     Inverse search (PDF double-click), forward search (`<leader>v`), slide-gap insertion
|       |   |   |-- AddSlide.tsx, useLatexTemplates.ts Primitive picker of a presentation root, template set per root
|       |   |   `-- present/          PresentButton, presenter and audience windows over one `BroadcastChannel`, slide notes from PDF annotations
|       |   |-- learn/                Learn mode frame, sidebar and sessions (LearnProvider, LearnSurface)
|       |   |-- factory/              Boards, task view, settings and previews (core shapes in `lib/domain/factory.ts`)
|       |   |   |-- FactoryBoard.tsx  Columns per repo (Working, Merging, Finished, Waiting block), delete with revert
|       |   |   |-- waveTree.ts, WaveTree.tsx, WaveOutcome.tsx Waves as column containers grouped by tag prefix; reviews, gates, merge
|       |   |   |-- states.ts         Shown state, labels, board columns, allowed actions
|       |   |   |-- useFactoryTasks.ts Task and wave lists per repo, refetched on pi-host session changes
|       |   |   |-- FactorySettingsDialog.tsx, NewTaskDialog.tsx, ConfigDraft.tsx Small/big models and defaults; new task with images; `factory.yaml` draft
|       |   |   |-- TaskView.tsx      Diff or preview, commands, `AgentPanel` (timeline, Pi session)
|       |   |   |-- taskReview.ts     Review comments on the task diff, sent as one message that opens a round
|       |   |   `-- TaskTimeline.tsx, useTaskTimeline.ts Rounds, gates, review and merge from `tasks.timeline`
|       |   |-- sharing/              `ShareButton` (create, copy, rotate, revoke a link), `useSharedResource`
|       |   |-- memory/               Memory window over the memory-store root
|       |   |   |-- MemoryViewer.tsx  Window shell: root, memory_stores, folders, memories, documents; Esc goes up
|       |   |   |-- useMemoryLibrary.ts Listing of a level, open memory or document, applied ops
|       |   |   `-- Overview.tsx, MemoryPage.tsx, EditableText.tsx Level listing, facts and description edited in place
|       |   |-- projects/             Project dashboard/kanban
|       |   |   `-- ProjectDashboard.tsx
|       |   `-- settings/             Model, reasoning, prompts, preferences and theme
|       |       |-- SettingsMenu.tsx
|       |       |-- ModelDropdown.tsx Chat model picker
|       |       `-- ModelSettingsPanel.tsx Provider/model admin view and default models
|       |-- contexts/                 Shared project/settings/tab/undo/host-scale state
|       |   |-- ProjectContext.tsx
|       |   |-- StartupReadContext.tsx Bounded initial reads, failure banner and targeted retry
|       |   |-- TabContext.tsx        `fileItem`: the workspace item of a file path
|       |   `-- UndoDeleteContext.tsx
|       |-- components/
|       |   |-- ui/                   Reusable interaction primitives
|       |   |   |-- Dialog.tsx
|       |   |   |-- Composer.tsx      Composer card, `ComposerTextarea`, `ComposerMicButton`, `ComposerSendButton`: shared by ChatInput, AssistantChat and the factory AgentPanel
|       |   |   |-- FloatingWindow.tsx Modal floating window, `FloatingPanel` its panel look shared with the picker preview
|       |   |   |-- CommandPicker.tsx The one search picker: list at a trigger (`CommandPicker`) or docked (`FloatingPicker`), with a preview a sidebar-card finder beside a centered preview window; `useAltKeyPicker` opens it on an Alt+key
|       |   |   |-- AmbientCopy.tsx The app background's picture as a window's background (picker preview)
|       |   |   |-- ErrorBoundary.tsx Render-error fallback with a reset (app-wide and per tab)
|       |   |   |-- Disclosure.tsx, Switch.tsx Collapsible section; on/off switch
|       |   |   |-- MenuPopover.tsx   Trigger plus popover menu, closed by outside press or Escape
|       |   |   |-- PillTabs.tsx, BoardColumn.tsx, Field.tsx
|       |   |   |-- IconButton.tsx, StatusDot.tsx Icon button, status tone dot (`StatusTone`)
|       |   |   `-- file-tree/        Shared file tree component and hook
|       |   `-- content/              Markdown, code, PDF and plot rendering
|       |       |-- LaTeXMarkdown.tsx Markdown renderer shell
|       |       |-- AssistantChat.tsx Ephemeral side chat shared by the editor sidebar and PDF assistant
|       |       |-- PdfViewer.tsx     PDF viewer orchestrator plus PdfPages
|       |       |-- pdf/              Page slots, page window, zoom, zoom controls, layout and canvas helpers
|       |       |-- PlotElement.tsx
|       |       `-- plotTheme.ts      Pure Plotly theme merge and plot sizing
|       |-- hooks/                    Feature-independent interaction hooks
|       |   |-- useEventListener.ts, useEscapeLayer.ts, useOutsidePress.ts
|       |   |-- useFollowFiles.ts     Follow absolute files and folders on disk, `onChange(path)`
|       |   |-- useDebouncedSave.ts   Quiet-time autosave plus unmount and forced flush
|       |   |-- useDebouncedSearch.ts Hits of a debounced query, one abortable request, shared by the chat and Pi session pickers
|       |   |-- useLoaded.ts          `useLoaded(key, load)`: the value or error of one keyed read
|       |   |-- useResource.ts        `useResource`, `useResourceState`, `useResourceFamily`: subscribe to a shared resource, first read as a startup read
|       |   |-- useFrameOverSlot.ts, useFrameAttachment.ts Lay the learn frame over a tab's slot, previewed tab first
|       |   `-- usePolling.ts, useViewportReframe.ts, useSelectionAction.ts, ...
|       |-- lib/
|       |   |-- backend/              HTTP, URL, runtime API base and stream clients
|       |   |   |-- api/                  Resource endpoints: `index.ts` (public entry `@/lib/backend/api`), `http.ts` (fetch core), one client per route family
|       |   |   |-- resources.ts          Shared instances of user-global reads (models, prompts, usage, sidebar lists), `refresh*` helpers, `forgetProjectDocuments`, `revalidateOnActivation`, `watchDocumentLists` (takes this device's change events too)
|       |   |   |-- references.ts         Live document references (attach, write back)
|       |   |   |-- config.ts
|       |   |   |-- urls.ts
|       |   |   |-- canvasStream.ts
|       |   |   |-- fileEvents.ts         File move and file change events: this device's announcements and other devices' change events
|       |   |   |-- fileWatch.ts          One ref-counted WebSocket per tab to `/api/files/watch`, reconnect with catch-up, changes into `fileEvents`
|       |   |   `-- syncStream.ts         Change feed: `subscribeToChanges(listener, { ownDevice })`, `syncCursorReady`
|       |   |-- resource.ts           `createResource`, `resourceFamily`: one cached value and one request for every tab, `set`/`invalidate`/`refresh` restart a running read, `revalidateStale`
|       |   |-- domain/               Browser domain types and pure document/graph logic
|       |   |   |-- types.ts              Chat core, models, Pi
|       |   |   |-- files.ts, projects.ts, histories.ts, memory.ts, factory.ts, pairs.ts    `files.ts` holds `isNewFileIn`, a folder entry of a project lists `children`
|       |   |   |-- jupyter.ts            Kernelspec list and running kernel shapes of `/api/jupyter/*`
|       |   |   |-- workbench.ts, latex.ts    Workbench kinds, notes guard, rename path rewrite, LaTeX build result and tex project
|       |   |   |-- architecture.ts
|       |   |   |-- canvasDocument.ts
|       |   |   `-- edgeRouting.ts
|       |   |-- paths.ts              File-kind and path helpers (`isInside`, `relativeIn`, `relativeTo`)
|       |   |-- fuzzy.ts              `fuzzyScore` for picker search
|       |   |-- utils.ts              Small shared helpers, `isRecord` (plain-object check before reading parsed JSON)
|       |   |-- pointer.ts            Follow the pointer on window until it lifts
|       |   `-- diff.ts               Diff parsing helpers
|       `-- packages/
|           `-- chat-ui/              Independently packaged local/learn chat UI contract: turns, markdown, code/Mermaid/Markmap renderers, click hints, palette, learn protocol
|               |-- src/
|               |   |-- index.ts      Public rendering exports
|               |   |-- Turn.tsx      Shared turn rendering
|               |   |-- MarkdownRenderer.tsx
|               |   |-- learn-protocol.ts LearnHarness iframe message protocol
|               |   |-- hints.ts/.css Vimium-style `f` click hints, installed app-wide by `AppShell`
|               |   `-- palette.css   Canonical role palette
|               |-- pack.mjs          Build/pack/sync tooling
|               `-- package.json      Package-local dependencies and distribution contract
|-- templates/                        Shipped sources read through `config.TEMPLATES`
|   |-- prompts/                      User-selectable seed prompts (`_order`, `_blocks/` fragments, e.g. `tutor.md`)
|   |-- workbench/                    Workbench Pi chat prompts `latex.md`, `markdown.md`, `pdf.md`, `canvas.md`, `notebook.md` (source of truth, `lib/workbench/prompts.py`)
|   `-- latex/                        LaTeX template sets, rules in `templates/latex/AGENTS.md`
|       |-- render.py                 Primitives -> `example.tex`/`example.pdf` -> preview PNGs
|       |-- presentation/             Beamer set: theme `.sty`, `skeleton.tex`, `STYLE.md`, `primitives/<group>/`, generated example and previews
|       |-- draft/                    Plain Beamer draft set: theme `.sty` (`\visual`, `\question`), `skeleton.tex`, `STYLE.md`, no primitives
|       `-- letter/                   KOMA-Script letter document set
|-- tests/                            Python route/service/store/security regressions
|   |-- conftest.py                    Shared fixtures, Postgres store helpers, `route_client`
|   |-- fakes.py                       Shared in-memory stores, `RecordingScope`, sandbox fakes, completion stream chunks
|   |-- shell.py                       Run repository shell scripts against fake commands
|   |-- test_postgres_mode_end_to_end.py
|   |-- test_data_store_conformance.py
|   `-- test_file_roots.py
|-- scripts/
|   |-- setup-postgres.sh              PostgreSQL setup helper
|   |-- migrate_memory.py              One-time move of old Postgres memories into memory-store roots
|   |-- check.sh                       Forwards to `factory/gates.sh` for tasks started before the move
|-- factory/gates.sh                  The one test pipeline (lint, types, python, frontend, pi-host; `--changed`, `--reuse`, `--dry-run`; test steps take a factory test slot)
|-- redeploy.sh                       Production rebase/restart of pi-host, backend, learn mode and the preview gate
|-- run.sh                            Development launcher composition
|-- run-backend.sh                    Backend and LikeC4 launch/setup
|-- run-frontend.sh                   Frontend launch/setup
|-- run-pi-host.sh                    pi-host launcher (Pi sessions, completion gateway, usage)
|-- pi-host/                          pi-host Bun service, see `.docs/pi-migration.md` (turn delete: `src/host/local/turns.ts`, `delete-turn.ts`; `src/host/entries.ts`: session `Entry`, `isPrompt`, `pathToRoot`; `src/json.ts`: `isRecord`)
|-- pi-extensions/                    Pi extensions loaded into every `coding` session
|                                     (the factory is the agent-factory package, `~/git/agent-factory`)
|-- run-learnharness.sh               LearnHarness sidecar launcher
|-- learnharness-overlay/             Private LearnHarness overlay, see `.docs/integration-learnharness.md`
|-- run-preview-gate.sh               Factory preview gate launcher
|-- factory.yaml                      Factory hooks, gates, protected paths and preview configuration
|-- .factory/agents/                  Factory role prompts `coder.md`, `reviewer.md`
`-- deploy/                           Production units and service/ingress helpers
    |-- gigachad-bot.service
    |-- gigachad-bot-pi-host.service
    |-- gigachad-bot-learnharness*.service/.timer  Learn mode units, see `.docs/integration-learnharness.md`
    |-- gigachad-bot-preview-gate.service
    |-- run-production-*.sh           Production runners of pi-host and LearnHarness
    |-- lib.sh                        Shared launcher helpers (ports, waits, deps, secrets, Postgres profile)
    |-- install-systemd-service.sh
    |-- load-env.sh                   Literal runtime configuration loader
    `-- tailscale-serve.sh
```

## Ownership and import boundaries

### Browser code

- Compose features in `src/frontend/app/_components/`; do not make a feature import the app shell.
- Put feature behavior under its owner in `features/`. Import another feature only through permitted public files at the feature root or one sub-feature level; concept-folder internals are not a general cross-feature API.
- Follow the actual allow-list in `eslint.config.mjs`: `chat -> editor/learn/conversation`, `conversation -> navigation`, `editor -> navigation/sharing`, `notebook -> navigation`, `factory -> editor/navigation/pi`, `navigation -> factory/learn/memory/pi`, `learn -> settings`, `latex -> workbench`, `workbench -> editor/navigation/notebook/pi`, and `pi -> chat/editor/settings/conversation`. Other cross-feature edges are disallowed by default, and `workbench` must not import a kind feature except `notebook` (`NotebookWorkbench`). Add shared domain logic below the features rather than bypassing the boundary.
- Keep `components/`, `contexts/`, `hooks/`, and `lib/` independent of `features/` and `app/`. Reuse `components/ui/` before creating local interaction primitives.
- Put browser API transport in `lib/backend/` and shared browser models/pure transformations in `lib/domain/`. These TypeScript contracts are separate from Python route models; coordinate both sides when changing payloads.
- Keep all Pi protocol knowledge in `features/pi/`, and the learn frame's iframe URL and message protocol (`packages/chat-ui/src/learn-protocol.ts`) in `features/learn/`.
- Keep `packages/chat-ui/` independently distributable: no `@/` app imports and no undeclared reliance on the shell's dependencies. `pack.mjs` packs it for `run-learnharness.sh`, which installs it into the served LearnHarness for `learnharness-overlay/`.

### Backend and separate services

- Start at `backend/server.py` for route registration/lifecycle, `routes/deps.py` for identity-scoped dependency construction (one `UserScope` per request), and `backend/identity.py` for access identity. Routes own transport validation and HTTP/SSE/WebSocket translation; shared error mapping and SSE helpers live in `backend/http_plumbing.py`.
- Put reusable storage/domain/external-service work in `src/lib/`, not in a second route's handler. `src/lib/` is a flat collection of concrete modules, not separate `services/`, `stores/`, or `adapters/` directories.
- Use `data_store.py` / `postgres_data_store.py` for app-owned text (`json_docs.py` for JSON values) and `asset_store.py` for bytes; use domain stores for chat/project/memory operations. Keep logical namespaces and key builders in `storage_namespace.py`. Postgres stores take a `changes.UserScope` and record change-log rows through it. `lib/canvas_store.py` is the explicit transactional canvas authority, not a generic document-store helper.
- Route mounted-file work through `lib/file_roots.py` (listing in `file_walk.py`, host-access policy in `host_access.py`, wiki links in `wiki_links.py`); use `git_diff.py` and `git_worktree.py` for git operations, both over `git.py` (`GitError`, `git_text`, `git_ok`; `WorktreeError` is a `GitError`). Keep host paths distinct from database keys and client URLs.
- Split architecture changes by ownership: `architecture_workspace.py` reads/commits sources; `architecture_service.py` runs validated writes and the commit-conflict retry; `c4_service.py` calls the service; `src/c4/` parses, lays out and edits source in memory; `features/editor/architecture/` handles browser interaction. The C4 service does not own persistence.
- Keep pi-host contract knowledge in `lib/pi_host.py`, the factory core's call contract in `lib/pi_host.py` (`factory_call`), and shared request/validation plumbing (`UpstreamClient`) in `lib/upstream.py`. Their route adapters map failures; `backend/learnharness_gate.py` and `backend/preview_gate.py` are identity proxies, not upstream API adapters.
- Treat Pi and the native sandbox implementation (Agents-in-a-box) as external dependencies. This repository owns their integration surfaces (pi-host and `pi-extensions/` included), not their internal session/execution engines.
- Keep sandbox execution and persistence distinct: `sandbox_service.py` stages workspace state, `routes/histories.py` checkpoints it after a successful chat save, and `sandbox_store.py` records durable manifests/runs. See [architecture](architecture.md#sandbox-checkpoint-boundary).
- Add a new numbered SQL migration under `db/schema/` for schema changes; `lib/db_schema.py` records checksums, so applied migration files are not editable history.

## Entry points and source limits

| Entry point | Follow next |
| --- | --- |
| Browser page | `src/frontend/app/page.tsx` -> `app/_components/AppShell.tsx` / `TabContent.tsx`; providers also enter through `app/layout.tsx` |
| Backend app | `src/backend/server.py` -> `routes/` -> `lib/`; settings in `src/config.py` |
| LikeC4 service | `src/c4/server.ts` -> `ops/apply.ts`, `payload.ts`, `workspace.ts`, `layout.ts` |
| Dependency installer | `gigachad-install` in `pyproject.toml` -> `src/backend/install_deps.py` |
| Development/production launch | Root `run*.sh`, `deploy/`; load operations guidance through [AGENTS.md](../AGENTS.md#documentation) |
| Runtime environment loading | `deploy/load-env.sh` -> root launchers / production runners; configuration consumers in `src/config.py` and individual adapters |
| Factory integration | agent-factory package (`~/git/agent-factory`, `factory --core-path`), `pi-host/src/factory.ts`, `factory.yaml`, `.factory/agents/`, `factory/gates.sh`, `src/backend/routes/factory.py`, `src/lib/factory_revert.py`, `factory_images.py`, `factory_migration.py`, `src/frontend/features/factory/`; previews: `src/lib/factory_preview.py`, `src/backend/routes/factory_preview.py`, `run-preview-gate.sh`, `src/backend/preview_gate.py`, `tests/test_preview_gate.py`, `appPath` in `src/frontend/lib/paths.ts` |

## Where to change

Full paths are repository-relative. Abbreviated sibling filenames share the preceding directory. `features/`, `contexts/`, `app/`, `components/`, `hooks/`, `packages/`, and browser `lib/backend/` paths resolve under `src/frontend/`. Backend `routes/` resolves under `src/backend/`. Pair UI changes with the transport/domain owner when behavior crosses the API boundary.

| Change | Primary source owner | Adjacent contracts and regression locations |
| --- | --- | --- |
| Shell layout, deep links, surface selection | `src/frontend/app/_components/AppShell.tsx`, `TabContent.tsx`, `deepLink.ts` | `features/navigation/TabManager.tsx`, `contexts/TabContext.tsx`, `app/_components/__tests__/deepLink.test.ts` |
| Chat generation, streaming or tool behavior | `src/backend/routes/chat.py`, `src/lib/toolcalling.py`, `src/lib/tools.py` | `src/frontend/features/chat/useChatStream.ts`, `useChat.ts`; `tests/test_toolcalling.py`, `tests/test_builtin_tools.py`, `tests/test_sandbox_plot.py`, `tests/test_chat_route.py`, chat `__tests__/useChatStream.test.tsx` |
| Chat save, branches or merge | `src/lib/chat_store.py`, `src/backend/routes/histories.py` | `features/chat/useChatPersistence.ts`, `features/navigation/BranchContext.tsx`; `tests/test_chat_store_message_preservation.py` |
| Conversation column, turn card, turn edit or delete | `src/frontend/features/conversation/`, `features/chat/ChatContainer.tsx`, `ChatPair.tsx`, `useChatTurns.ts`, `features/pi/PiTranscript.tsx`, `PiTab.tsx` | `packages/chat-ui/src/Turn.tsx`, `turn.css`, `pi-host/src/host/local/turns.ts`, `delete-turn.ts`, `src/backend/routes/pi.py`; conversation `__tests__/`, `app/_components/__tests__/chatEditDelete.test.tsx`, `pi-host/test/turns.test.ts`, `delete-turn.test.ts` |
| Chat list, search or picker | `src/lib/chat_store.py`, `src/backend/routes/histories.py`, `src/frontend/features/chat/ChatPicker.tsx`, `useChatPicker.ts` | `lib/backend/api/chat.ts`, `features/conversation/useTextHighlights.ts`, `app/_components/TabContent.tsx`; `tests/test_chat_search.py`, chat `__tests__/ChatPicker.test.tsx`, `ChatContainer.test.tsx` |
| Chat tool rows | `src/frontend/features/conversation/tools/`, `features/chat/toolRun.tsx`, `toolRenderers.tsx` | `features/pi/MemoryTool.tsx` (imported by `app/_components/TabContent.tsx`); chat `__tests__/toolRenderers.test.tsx` |
| Attachment live references, uploads or PDF extraction | `src/frontend/features/chat/attachments.ts`, `src/lib/attachments.py`, `attachment_materialize.py`, `extract_queue.py`, `mineru.py` | `src/backend/routes/files.py`, `documents.py`, `file_attach.py`; `tests/test_attachment_materialize.py`, `tests/test_extract_queue.py`, `tests/test_upload_assets.py`, chat `__tests__/attachments.test.ts` |
| Text/PDF/plot viewing or editing | `src/frontend/features/editor/DocumentEditor.tsx`, `FileViewer.tsx`, `features/editor/pdf/`, `src/frontend/components/content/` | `src/backend/routes/documents.py`, `src/lib/document_service.py`, `fileviewer.py` (PDF annotation canvas key); `tests/test_document_state.py`, `tests/test_postgres_mode_end_to_end.py`, editor/content `__tests__/` |
| Collaborative canvas or canvas entity shape | `src/frontend/features/editor/canvas/`, `src/frontend/lib/domain/canvasDocument.ts` | `src/lib/canvas_store.py`, `routes/canvases.py`, `lib/backend/canvasStream.ts`; `tests/test_canvas_mutations.py`, canvas `__tests__/` |
| Architecture syntax edits, diagram or layout | `src/c4/ops/`, `workspace.ts`, `payload.ts`, `layout.ts`; `src/frontend/features/editor/architecture/` | `src/lib/architecture_workspace.py`, `architecture_service.py`, `c4_service.py`, `src/backend/routes/architecture.py`; `src/c4/operations.test.ts`, `tests/test_architecture_workspace.py` |
| Mount browsing or filesystem access policy | `src/lib/file_roots.py`, `disk_files.py`, `host_access.py`, `file_walk.py`, `src/backend/routes/file_roots.py` | `features/navigation/FileMountPicker.tsx`, `components/ui/file-tree/`; `tests/test_file_roots.py`, `tests/test_disk_files.py` |
| Git diffs, review drafts or worktree switching | `src/lib/git_diff.py`, `git_worktree.py`; `src/frontend/features/editor/diff/`, `features/navigation/useWorktrees.tsx` | `src/backend/routes/file_git.py`; `tests/test_git_diff.py`, `tests/test_git_worktree.py`, diff `__tests__/` |
| Neovim editing, terminal save bridge or agent-edit merge | `src/lib/neovim_session.py`, `neovim_rpc.py`, `text_merge.py`, `src/backend/routes/neovim.py`, `src/frontend/features/editor/NeovimEditor.tsx` | `tests/test_neovim.py` (notebook text too), `tests/test_text_merge.py`; filesystem/identity policy stays with its existing owners |
| Workbench layout, sync, routing or a new file kind | `src/frontend/features/workbench/`, a kind wrapper (`MarkdownWorkbench.tsx`, `features/latex/`), `lib/domain/workbench.ts`; backend `src/lib/workbench/`, `routes/workbench.py` | Kind prompt in `templates/workbench/` (`lib/workbench/prompts.py`), routing in `TabContent.tsx` / `useWorkspaceDocuments.ts`, buffer routing in `features/navigation/tabs.ts`, rename cascade in `project_rename.py`, `lib/backend/fileEvents.ts`; `tests/test_workbench.py`, `tests/test_project_rename.py`, workbench `__tests__/` |
| LaTeX build, tex projects or PDF pane | `src/lib/latex.py`, `routes/latex.py`, `features/latex/` | `file_walk.py` (build output), `routes/workbench.py` (promote, images), `components/content/PdfViewer.tsx` (`pageBox`, layered swap under one `documentKey`); `tests/test_latex.py`, latex `__tests__/`, `PdfViewer.swap.test.tsx` |
| SyncTeX, presentation templates, slides or present mode | `src/lib/synctex.py`, `tex_source.py`, `latex_templates.py`, `templates/latex/`, `features/latex/useSynctex.ts`, `AddSlide.tsx`, `features/latex/present/` | `templates/latex/AGENTS.md` (re-render after every change), `components/content/pdf/pdfText.ts`, `routes/workbench.py` (create from a set, image references), `neovim_rpc.py` (slide/image insertion); `tests/test_synctex.py`, `tests/test_tex_source.py`, `tests/test_latex_templates.py`, present `__tests__/` |
| Jupyter notebooks (editor, kernels, proxy, file sync) | `src/lib/jupyter.py`, `src/backend/routes/jupyter.py`, `src/frontend/features/notebook/` | `src/backend/reverse_proxy.py`, `websocket_origin.py` (`require_trusted_request`), rename/delete hooks in `routes/documents.py`, `lib/domain/jupyter.ts`, `hooks/useRootFileDocument.ts`, `lib/backend/fileEvents.ts` (`announceFileClosed`), `next.config.mjs`, `AGENTS.md` "Jupyter notebooks"; Neovim on the notebook: `features/workbench/NotebookWorkbench.tsx`, `WorkbenchFrame.tsx`, `features/notebook/notebookText.ts`, `notebookTextStore.ts`, `NeovimEditor.tsx`, `routes/neovim.py`, `tests/test_neovim.py`; `tests/test_jupyter.py`, notebook and workbench `__tests__/` |
| File chat sidebar (Pi sessions per file or LaTeX project folder) | `src/frontend/features/workbench/DocumentChat.tsx`, `useDocumentChat.ts`, `fileLead.ts`, `FileChat.tsx` | `routes/workbench.py` (`POST /api/workbench/chat`), `lib/workbench/links.py`, `features/editor/FilePanel.tsx`; workbench `__tests__/DocumentChat.test.tsx`, `tests/test_workbench_chat.py` |
| Learn mode | `src/lib/learnharness.py`, `src/lib/learnharness_config.py`, `routes/learn.py`, `features/learn/`, `learnharness-overlay/` | `.docs/integration-learnharness.md`, `run-learnharness.sh`, `lib/models.ts` (`gatewayModelId`) |
| Projects, kanban, rename, archive or persisted tabs | `src/lib/project_store.py`, `project_rename.py`, `src/backend/routes/projects.py`, `src/frontend/features/projects/` | `src/frontend/contexts/ProjectContext.tsx`, `features/navigation/TabManager.tsx`, `useProjectActions.ts`; `tests/test_document_state.py`, `tests/test_project_rename.py`, projects/context `__tests__/` |
| Memory store, memory tool or memory window | `src/lib/memory_context.py`, `src/lib/tools.py` (memory tools), `src/backend/routes/memory.py`, `src/frontend/features/memory/` | `memory-store` package (`pyproject.toml`), `routes/chat.py` (`_memory_block`), `project_rename.py`, `scripts/migrate_memory.py`; `tests/test_memory_context.py`, `tests/test_memory_route.py`, `tests/test_memory_tools.py`, `tests/test_migrate_memory.py` |
| Share links | `src/lib/share_links.py`, `src/backend/routes/share_links.py`, `src/frontend/features/sharing/`, `app/share/page.tsx` | `db/schema/0020_share_links.sql`, `routes/canvases.py` (shared canvas engine), `features/editor/canvas/SharedCanvas.tsx`, `lib/backend/api/shareLinks.ts` |
| Model providers, reasoning or prompts | `src/lib/model_provider_store.py`, `reasoning_levels.py`, `prompt_store.py`, `src/backend/routes/models.py`, `src/frontend/features/settings/` | `templates/prompts/` for user seeds, `src/lib/prompts/` for internal instructions; `tests/test_model_provider_store.py`, `tests/test_reasoning_levels.py` |
| Provider usage or Pi model gateway | `src/lib/provider_usage.py`, `pi_models.py`, `src/frontend/features/navigation/ProviderUsage.tsx` | `src/backend/routes/models.py`, `pi-host/`; `tests/test_provider_usage.py`, `tests/test_pi_models.py` |
| Pi sessions, tabs or composer | `src/lib/pi_host.py`, `src/backend/routes/pi.py`, `src/frontend/features/pi/` | `pi-host/` (`src/media.ts`: tool-image store), `src/lib/pi_media.py`, `features/pi/toolImages.ts`, `.docs/pi-migration.md`; `tests/test_pi_host.py`, pi `__tests__/` |
| Factory board, tasks, waves or preview UI | agent-factory package (core, runners, extension; its own tests), `pi-host/src/factory.ts`, `src/backend/routes/factory.py`, `src/frontend/features/factory/` | `src/lib/factory_preview.py`, `factory_revert.py`, `factory_images.py`, `factory_migration.py`, `factory.yaml`, `.docs/agent-factory.md`; `tests/test_factory*.py`, `pi-host/test/factory.test.ts`, factory `__tests__/` |
| Sandbox execution or durable rich outputs | `src/lib/agent_sandbox_adapter.py`, `sandbox_service.py`, `sandbox_store.py`, `sandbox_plot.py` | `src/lib/tools.py`, chat message renderers; `tests/test_sandbox_service.py`, `tests/test_sandbox_runtime_state.py`, `tests/test_sandbox_native_runs.py` |
| Identity, host capability or ingress | `src/backend/identity.py`, `routes/upstream.py`, `learnharness_gate.py`, `preview_gate.py`, `src/config.py` | `deploy/`; `tests/test_identity.py`, `tests/test_learnharness_gate.py`, `tests/test_preview_gate.py` |
| Production redeploy | `redeploy.sh`, `src/backend/routes/deploy.py`, `features/settings/SettingsMenu.tsx` | `deploy/lib.sh`, `.docs/deployment.md` |
| Storage schema, change streams or API transport | `db/schema/`, `src/lib/db_schema.py`, `postgres_data_store.py`, `asset_store.py`; `src/backend/sync.py` | `routes/sync.py`, `src/frontend/lib/backend/`; `tests/test_data_store_conformance.py`, `tests/test_sync.py`, `tests/test_postgres_mode_end_to_end.py` |
| Reusable interaction or content rendering | `src/frontend/components/ui/`, `components/content/`, `hooks/`, `packages/chat-ui/` | Colocated `__tests__/`; preserve feature-independent imports and package distribution boundaries |

## Regression navigation

- Use `tests/` for Python route/service/store and access-policy coverage. `tests/conftest.py` and `pyproject.toml` own fixture/test setup: Postgres fixtures (`database_url`, `postgres_pool`, `clean_database`, `data_store`, `asset_store`) and the `route_client` app builder live only in `conftest.py`, in-memory fakes in `tests/fakes.py`. PostgreSQL integration behavior is represented by `test_postgres_mode_end_to_end.py` and store conformance tests.
- Use colocated `src/frontend/**/__tests__/` for browser logic/components, including shared package tests. Configuration lives at `src/frontend/vitest.config.ts` and `vitest.setup.ts`.
- Use `src/c4/operations.test.ts` for LikeC4 source-edit/layout regressions; this service has its own `package.json`, separate from the frontend package.
- Use `factory.yaml` to locate the repository-wide factory hooks, gates and preview process configuration.

## Refresh evidence and scope

- The 2026-11-25 refresh re-verified against source the memory-store replacement, share links, the factory's gigachad-side contracts (pi-host core loading, login tag, revert, images, waves on the board), Pi transcript cards, Pi start folders, provider usage and chat transport, and checked every tree leaf against the tracked tree. Other sections were maintained commit by commit and not re-verified line by line. The agent-factory and memory-store packages' internals, and deployed service behavior, are outside this refresh.
- The 2026-11-26 hand edit re-verified against source the conversation column, turn card, turn delete, chat list and search, chat picker, chat tool rows and the factory cards in a Pi tab; the rest was not re-checked.
