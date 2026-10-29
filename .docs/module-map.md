Generated from 9f773d5 on 2026-10-27

# Module map

Use this file to locate the owner of a change. Read [system overview](system-overview.md) for concepts and workflows, [architecture](architecture.md) for runtime contracts, persistence, security, and failure semantics. Follow [repository documentation guidance](../AGENTS.md#documentation) for optional operational runbooks.

## Path conventions and source tree

- Resolve paths below from the repository root unless a section explicitly gives another root. In Python, `backend`, `lib`, and `config` resolve under `src/`; frontend `@/` imports resolve under `src/frontend/`.
- Distinguish `src/lib/` (Python services/storage/adapters), `src/frontend/lib/` (browser contracts/utilities), and `src/frontend/packages/chat-ui/` (a separately distributed shared package).
- Read this as a selective ASCII tree: listed leaves are representative real files, not an exhaustive inventory. Omitted siblings are not implied absent.
- Exclude dependency trees, lockfile contents, build outputs, caches, runtime data, copied browser workers/WASM, and packaged binaries from the module map.

```text
.
|-- AGENTS.md                         Repository rules; operational documentation index
|-- README.md                         Setup and developer entry guide
|-- PRODUCT.md                        Product direction
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
|       `-- 0010_memory_topics.sql    Memory topics migration
|-- src/
|   |-- config.py                     Env-derived backend settings (imports nothing from lib)
|   |-- backend/
|   |   |-- server.py                 FastAPI composition, lifecycle and shared error mapping
|   |   |-- http_plumbing.py          Exception-to-HTTP mapping, SSE responses, inline image decoding
|   |   |-- identity.py               Request identity and host-access capability
|   |   |-- sync.py                   PostgreSQL notification broker
|   |   |-- omp_web_gate.py           Identity-checking frame proxy
|   |   |-- install_deps.py           gigachad-install entry point
|   |   |-- sidecar.py                Desktop (Tauri) entry: prints `{"event":"ready","port":…}` once FastAPI listens
|   |   `-- routes/                   HTTP, SSE and WebSocket boundaries
|   |       |-- deps.py               Identity-scoped store/service injection
|   |       |-- schemas.py            Shared file-route request/response models
|   |       |-- chat.py               Chat generation and tools
|   |       |-- histories.py          Chat persistence, branching and merge endpoints
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
|   |       |-- memory.py             Memory extraction, review and CRUD
|   |       |-- models.py             Models, providers, usage, defaults and prompts
|   |       |-- config.py             Browser configuration and capability flags
|   |       |-- ocr.py                Image OCR generation
|   |       |-- neovim.py             Terminal WebSocket and document-save bridge
|   |       |-- sync.py               Durable change-log SSE
|   |       |-- omp.py                omp-web API boundary
|   |       |-- factory.py            Agent factory API boundary
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
|   |   |-- chat_store.py             Chat model, I/O, indexing and atomic tree operations
|   |   |-- project_store.py          Catalog, slugs, kanban and tab persistence, project file references with display names
|   |   |-- document_service.py       `DocumentScope` (notes or a project), document write/move with reference repointing, display-name rename
|   |   |-- chat_cleanup.py           `ChatCleanup`: release uploads, derived data and sandbox state after chat deletion
|   |   |-- references.py             Reference model, post-delete cleanup of unused derived data, `--dry-run` orphan report, `annotation_canvas` (PDF annotation key, recorded host files)
|   |   |-- project_rename.py         Project rename transaction and reference rewriting
|   |   |-- memory_store.py           `MemoryScope`, `MemoryOperation`, topic/fact persistence, pending buffer, watermarks, atomic commit
|   |   |-- memory_extract.py         `MemoryExtractor`: LLM extraction proposals and regroup
|   |   |-- memory_context.py         Memory injection into chat system prompts
|   |   |-- model_provider_store.py   Provider catalog and model defaults
|   |   |-- reasoning_levels.py       Reasoning-effort levels: live discovery, stored answers, send-time kwargs
|   |   |-- prompt_store.py           User prompt storage (read lazily) and seed/block loading
|   |   |-- file_roots.py             Mounted-file roots, path safety, read/write, tree
|   |   |-- file_roots_repository.py  Per-user root registry (`vault_roots`)
|   |   |-- file_walk.py              `.gitignore`-aware directory walk and tree
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
|   |   |-- research.py               GPT-Researcher adapter
|   |   |-- sandbox_service.py        Workspace/output orchestration
|   |   |-- sandbox_store.py          Durable sandbox metadata
|   |   |-- agent_sandbox_adapter.py  Native sandbox runner adapter
|   |   |-- fake_sandbox_runner.py    Docker-free runner (selected by env in `routes/deps.py`)
|   |   |-- sandbox_plot.py           Plot generation
|   |   |-- omp_source.py             OMP auth-gateway model integration
|   |   |-- omp_web.py                omp-web URL/response-contract owner
|   |   |-- factory.py                External factory engine adapter
|   |   |-- upstream.py               Shared server-to-server request/validation plumbing, `UpstreamClient`
|   |   |-- llm_resilience.py         One retry for stale LiteLLM parameter rejections, `completion_text`
|   |   |-- omp_cli.py                `run_omp`: one runner for the `omp` CLI
|   |   |-- naming.py                 `slugify` for project slugs
|   |   |-- ttl_cache.py              Lock-free single-value TTL cache
|   |   |-- provider_usage.py         OMP subscription usage reader/cache
|   |   |-- neovim_session.py         Neovim PTY/process lifecycle
|   |   |-- neovim_rpc.py             Neovim msgpack-RPC client and save/yank Lua bridge
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
|       |-- next.config.js            Next.js/export configuration
|       |-- vitest.config.ts          Browser unit/component test configuration
|       |-- app/
|       |   |-- layout.tsx            Root providers, font and styles
|       |   |-- page.tsx              Client shell entry
|       |   |-- globals.css           Shell theme and application styles
|       |   `-- _components/
|       |       |-- AppShell.tsx      Cross-feature composition and workspace navigation
|       |       |-- TabContent.tsx    Tab layout: sidebar, surface, overlays and modals
|       |       |-- WorkspaceSurface.tsx Per-mode table of top-bar controls and surface body
|       |       |-- useTabChat.ts     Tab chat hooks, kept mounted in every mode
|       |       |-- deepLink.ts       Preview/open-target parsing
|       |       |-- DesktopBackendProvider.tsx Tauri-only API base discovery
|       |       `-- MemoryCommandLayer.tsx
|       |-- features/                 Domain-owned interactive surfaces
|       |   |-- chat/                 Turns, attachments, persistence and streaming
|       |   |   |-- ChatContainer.tsx Message column; pair expansion and scroll pinning in usePairExpansion/useStickToBottom
|       |   |   |-- attachments.ts
|       |   |   |-- composer/         Input, completions, drawings and attachment chips (useDraftAttachments, dictation via hooks/useSpeechInput)
|       |   |   |-- messages/         Tools, research and rich output rendering
|       |   |   `-- preview/          Attachment previews
|       |   |-- editor/               Documents, viewers and terminal editors
|       |   |   |-- DocumentEditor.tsx
|       |   |   |-- TextDocumentEditor.tsx Markdown/text save and preview owner (EditorShell, EditorViewSwitch, useEditorView)
|       |   |   |-- useRootFileDocument.ts Load, dirty, remote reload and chained saves of one file, codec-aware
|       |   |   |-- useDebouncedSave.ts Quiet-time autosave plus unmount and forced flush
|       |   |   |-- FileViewer.tsx
|       |   |   |-- FilePanel.tsx     Per-tab side panel slot (Neovim, PDF assistant), one owner at a time
|       |   |   |-- NeovimEditor.tsx
|       |   |   |-- pdf/              AnnotatedPdfViewer (ink over pages), pdfInk.ts, PdfAssistant (MinerU-grounded chat)
|       |   |   |-- architecture/     LikeC4 graph/source editing
|       |   |   |   `-- ArchitectureWindow.tsx
|       |   |   |-- canvas/           Canvas entities, collaboration, history and camera
|       |   |   |   |-- CanvasEditor.tsx  Wiring and layout over the hooks below, plus the nested canvas window body
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
|       |   |   |-- workspaceTree.ts  Pure Projects and Code tree builders
|       |   |   |-- useWorkspaceDocuments.ts, useCodeMounts.ts, useProjectActions.ts
|       |   |   |-- TabManager.tsx    Tab state provider; tabs.ts pure helpers, useWorkspaceSwitch.ts, TabSwitcher.tsx
|       |   |   |-- ArchiveSelector.tsx, useArchivedDirectories.ts Restore archived projects/Code directories
|       |   |   `-- useWorktrees.tsx
|       |   |-- omp/                  Single-frame surface, bridge, composer and sessions
|       |   |   |-- OmpProvider.tsx
|       |   |   |-- OmpComposer.tsx
|       |   |   `-- OmpSelect.tsx     Fuzzy finder for an omp extension's `select` (e.g. `/files`)
|       |   |-- factory/              Boards, task details, settings and previews
|       |   |   |-- FactoryBoard.tsx
|       |   |   `-- TaskView.tsx
|       |   |-- memory/               Memory window, review panel and memory commands
|       |   |   |-- MemoryViewer.tsx  Memory window shell and view switching
|       |   |   |-- useMemoryLibrary.ts Scope loading, queued per-scope writes, delete with undo
|       |   |   |-- scopeEdits.ts     Pure scope changes (rename, merge, move, delete, restore)
|       |   |   |-- Overview.tsx / TopicPage.tsx / RegroupPage.tsx Memory window pages
|       |   |   |-- MemoryPanel.tsx   Single-screen review of proposed operations
|       |   |   `-- useMemorizeFlow.ts Extraction, review and commit orchestration
|       |   |-- projects/             Project dashboard/kanban
|       |   |   `-- ProjectDashboard.tsx
|       |   `-- settings/             Model, reasoning, prompts, preferences and theme
|       |       |-- SettingsMenu.tsx
|       |       |-- ModelDropdown.tsx Chat model picker
|       |       `-- ModelSettingsPanel.tsx Provider/model admin view and default models
|       |-- contexts/                 Shared project/settings/tab/undo/host-scale state
|       |   |-- ProjectContext.tsx
|       |   |-- StartupReadContext.tsx Bounded initial reads, failure banner and targeted retry
|       |   |-- TabContext.tsx
|       |   `-- UndoDeleteContext.tsx
|       |-- components/
|       |   |-- ui/                   Reusable interaction primitives
|       |   |   |-- Dialog.tsx
|       |   |   |-- FloatingWindow.tsx
|       |   |   |-- MenuPopover.tsx   Trigger plus popover menu, closed by outside press or Escape
|       |   |   |-- PillTabs.tsx, BoardColumn.tsx, Field.tsx
|       |   |   `-- file-tree/        Shared file tree component and hook
|       |   `-- content/              Markdown, code, PDF and plot rendering
|       |       |-- LaTeXMarkdown.tsx Markdown renderer shell
|       |       |-- AssistantChat.tsx Ephemeral side chat shared by the editor sidebar and PDF assistant
|       |       |-- diagrams/         CitationPill (code, Mermaid and Markmap render from packages/chat-ui)
|       |       |-- PdfViewer.tsx     PDF viewer orchestrator plus PdfPages
|       |       |-- pdf/              Page slots, page window, zoom, zoom controls, layout and canvas helpers
|       |       |-- PlotElement.tsx
|       |       `-- plotTheme.ts      Pure Plotly theme merge and plot sizing
|       |-- hooks/                    Feature-independent interaction hooks
|       |   |-- useEventListener.ts, useEscapeLayer.ts, useOutsidePress.ts
|       |   `-- usePolling.ts, useViewportReframe.ts, useSelectionAction.ts, ...
|       |-- lib/
|       |   |-- backend/              HTTP, URL, runtime API base and stream clients
|       |   |   |-- api.ts                Resource endpoints over one fetch core
|       |   |   |-- references.ts         Live document references (attach, write back)
|       |   |   |-- config.ts
|       |   |   |-- urls.ts
|       |   |   |-- canvasStream.ts
|       |   |   `-- syncStream.ts
|       |   |-- domain/               Browser domain types and pure document/graph logic
|       |   |   |-- types.ts              Chat core, models, omp
|       |   |   |-- files.ts, projects.ts, histories.ts, memory.ts, factory.ts, pairs.ts
|       |   |   |-- architecture.ts
|       |   |   |-- canvasDocument.ts
|       |   |   `-- edgeRouting.ts
|       |   |-- paths.ts              File-kind/path helpers
|       |   |-- pointer.ts            Follow the pointer on window until it lifts
|       |   `-- diff.ts               Diff parsing helpers
|       `-- packages/
|           `-- chat-ui/              Independently packaged local/OMP chat UI contract: turns, markdown, code/Mermaid/Markmap renderers, palette, protocol
|               |-- src/
|               |   |-- index.ts      Public rendering exports
|               |   |-- Turn.tsx      Shared turn rendering
|               |   |-- MarkdownRenderer.tsx
|               |   |-- protocol.ts   Versioned iframe message protocol
|               |   `-- palette.css   Canonical role palette
|               |-- pack.mjs          Build/pack/sync tooling
|               `-- package.json      Package-local dependencies and distribution contract
|-- prompts/                          User-selectable seed prompts
|   |-- _order                        Prompt ordering
|   |-- tutor.md                      Representative user prompt
|   `-- _blocks/                      Reusable prompt fragments
|-- tests/                            Python route/service/store/security regressions
|   |-- conftest.py                    Shared fixtures, Postgres store helpers, `route_client`
|   |-- fakes.py                       Shared in-memory stores, sandbox and LLM client fakes
|   |-- shell.py                       Run repository shell scripts against fake commands
|   |-- test_postgres_mode_end_to_end.py
|   |-- test_data_store_conformance.py
|   |-- test_file_roots.py
|   `-- test_omp_web.py
|-- scripts/
|   |-- setup-postgres.sh              PostgreSQL setup helper
|   `-- build-tauri-linux.sh           Desktop sidecar/AppImage build (expects absent `src/frontend/src-tauri/`)
|-- redeploy.sh                       Production rebase/restart, also for omp-web and its gate
|-- run.sh                            Development launcher composition
|-- run-backend.sh                    Backend and LikeC4 launch/setup
|-- run-frontend.sh                   Frontend launch/setup
|-- run-omp.sh                        OMP auth gateway launcher
|-- run-omp-web.sh                    External omp-web worktree launcher
|-- run-factory.sh                    External factory engine launcher
|-- factory.yaml                      Factory hooks, test gate and preview configuration
`-- deploy/                           Production units and service/ingress helpers
    |-- gigachad-bot.service
    |-- gigachad-bot-omp-web-gate.service
    |-- gigachad-bot-factory.service
    |-- lib.sh                        Shared launcher helpers (ports, waits, deps, secrets, Postgres profile)
    |-- install-systemd-service.sh
    |-- load-env.sh                   Literal runtime configuration loader
    `-- tailscale-serve.sh
```

## Ownership and import boundaries

### Browser code

- Compose features in `src/frontend/app/_components/`; do not make a feature import the app shell.
- Put feature behavior under its owner in `features/`. Import another feature only through permitted public files at the feature root or one sub-feature level; concept-folder internals are not a general cross-feature API.
- Follow the actual allow-list in `eslint.config.mjs`: `chat -> editor`, `factory -> editor`, `navigation -> factory/memory/omp`, and `omp -> chat/editor/settings`. Other cross-feature edges are disallowed by default. Add shared domain logic below the features rather than bypassing the boundary.
- Keep `components/`, `contexts/`, `hooks/`, and `lib/` independent of `features/` and `app/`. Reuse `components/ui/` before creating local interaction primitives.
- Put browser API transport in `lib/backend/` and shared browser models/pure transformations in `lib/domain/`. These TypeScript contracts are separate from Python route models; coordinate both sides when changing payloads.
- Keep all iframe URLs, bridge knowledge and message-protocol imports in `features/omp/`. The canonical protocol lives in `packages/chat-ui/src/protocol.ts`; other shell modules may consume the shared rendering package but not that protocol directly.
- Keep `packages/chat-ui/` independently distributable: no `@/` app imports and no undeclared reliance on the shell's dependencies. `pack.mjs` packs it for `run-omp-web.sh`, which installs it into the served omp-web for `ompweb-overlay/`; do not vendor omp-web code here.

### Backend and separate services

- Start at `backend/server.py` for route registration/lifecycle, `routes/deps.py` for identity-scoped dependency construction (one `UserScope` per request), and `backend/identity.py` for access identity. Routes own transport validation and HTTP/SSE/WebSocket translation; shared error mapping and SSE helpers live in `backend/http_plumbing.py`.
- Put reusable storage/domain/external-service work in `src/lib/`, not in a second route's handler. `src/lib/` is a flat collection of concrete modules, not separate `services/`, `stores/`, or `adapters/` directories.
- Use `data_store.py` / `postgres_data_store.py` for app-owned text (`json_docs.py` for JSON values) and `asset_store.py` for bytes; use domain stores for chat/project/memory operations. Keep logical namespaces and key builders in `storage_namespace.py`. Postgres stores take a `changes.UserScope` and record change-log rows through it. `lib/canvas_store.py` is the explicit transactional canvas authority, not a generic document-store helper.
- Route mounted-file work through `lib/file_roots.py` (listing in `file_walk.py`, host-access policy in `host_access.py`, wiki links in `wiki_links.py`); use `git_diff.py` and `git_worktree.py` for git operations, both over `git.py` (`GitError`, `git_text`, `git_ok`; `WorktreeError` is a `GitError`). Keep host paths distinct from database keys and client URLs.
- Split architecture changes by ownership: `architecture_workspace.py` reads/commits sources; `architecture_service.py` runs validated writes and the commit-conflict retry; `c4_service.py` calls the service; `src/c4/` parses, lays out and edits source in memory; `features/editor/architecture/` handles browser interaction. The C4 service does not own persistence.
- Keep omp-web contract knowledge in `lib/omp_web.py`, factory contracts in `lib/factory.py`, and shared request/validation plumbing (`request_json`, `UpstreamClient`) in `lib/upstream.py`. Their route adapters map failures; `backend/omp_web_gate.py` is an identity proxy, not another upstream API adapter.
- Treat omp-web, the Agents-in-a-box factory engine, and native sandbox implementation as external dependencies. This repository owns their integration surfaces, not their internal task/session/execution engines.
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
| Factory integration | `factory.yaml`, `run-factory.sh`, `src/lib/factory.py`, `src/backend/routes/factory.py`, `src/frontend/features/factory/` |

## Where to change

Full paths are repository-relative. Abbreviated sibling filenames share the preceding directory. `features/`, `contexts/`, `app/`, `components/`, `hooks/`, `packages/`, and browser `lib/backend/` paths resolve under `src/frontend/`. Backend `routes/` resolves under `src/backend/`. Pair UI changes with the transport/domain owner when behavior crosses the API boundary.

| Change | Primary source owner | Adjacent contracts and regression locations |
| --- | --- | --- |
| Shell layout, deep links, surface selection | `src/frontend/app/_components/AppShell.tsx`, `TabContent.tsx`, `deepLink.ts` | `features/navigation/TabManager.tsx`, `contexts/TabContext.tsx`, `app/_components/__tests__/deepLink.test.ts` |
| Chat generation, streaming or tool behavior | `src/backend/routes/chat.py`, `src/lib/toolcalling.py`, `src/lib/tools.py` | `src/frontend/features/chat/useChatStream.ts`, `useChat.ts`; `tests/test_toolcalling.py`, `tests/test_builtin_tools.py`, `tests/test_sandbox_plot.py`, `tests/test_chat_route.py`, chat `__tests__/useChatStream.test.tsx` |
| Chat save, branches or merge | `src/lib/chat_store.py`, `src/backend/routes/histories.py` | `features/chat/useChatPersistence.ts`, `features/navigation/BranchContext.tsx`; `tests/test_histories.py`, `tests/test_chat_store_message_preservation.py` |
| Attachment live references, uploads or PDF extraction | `src/frontend/features/chat/attachments.ts`, `src/lib/attachments.py`, `attachment_materialize.py`, `extract_queue.py`, `mineru.py` | `src/backend/routes/files.py`, `documents.py`, `file_attach.py`; `tests/test_attachment_materialize.py`, `tests/test_extract_queue.py`, `tests/test_upload_assets.py`, chat `__tests__/attachments.test.ts` |
| Text/PDF/plot viewing or editing | `src/frontend/features/editor/DocumentEditor.tsx`, `FileViewer.tsx`, `features/editor/pdf/`, `src/frontend/components/content/` | `src/backend/routes/documents.py`, `src/lib/document_service.py`, `fileviewer.py` (PDF annotation canvas key); `tests/test_document_state.py`, `tests/test_postgres_mode_end_to_end.py`, editor/content `__tests__/` |
| Collaborative canvas or canvas entity shape | `src/frontend/features/editor/canvas/`, `src/frontend/lib/domain/canvasDocument.ts` | `src/lib/canvas_store.py`, `routes/canvases.py`, `lib/backend/canvasStream.ts`; `tests/test_canvas_mutations.py`, canvas `__tests__/` |
| Architecture syntax edits, diagram or layout | `src/c4/ops/`, `workspace.ts`, `payload.ts`, `layout.ts`; `src/frontend/features/editor/architecture/` | `src/lib/architecture_workspace.py`, `architecture_service.py`, `c4_service.py`, `src/backend/routes/architecture.py`; `src/c4/operations.test.ts`, `tests/test_architecture_workspace.py` |
| Mount browsing or filesystem access policy | `src/lib/file_roots.py`, `host_access.py`, `file_walk.py`, `src/backend/routes/file_roots.py` | `features/navigation/FileMountPicker.tsx`, `components/ui/file-tree/`; `tests/test_file_roots.py` |
| Git diffs, review drafts or worktree switching | `src/lib/git_diff.py`, `git_worktree.py`; `src/frontend/features/editor/diff/`, `features/navigation/useWorktrees.tsx` | `src/backend/routes/file_git.py`; `tests/test_git_diff.py`, `tests/test_git_worktree.py`, diff `__tests__/` |
| Neovim editing or terminal save bridge | `src/lib/neovim_session.py`, `neovim_rpc.py`, `src/backend/routes/neovim.py`, `src/frontend/features/editor/NeovimEditor.tsx` | `tests/test_neovim.py`; filesystem/identity policy stays with its existing owners |
| Projects, kanban, rename, archive or persisted tabs | `src/lib/project_store.py`, `project_rename.py`, `src/backend/routes/projects.py`, `src/frontend/features/projects/` | `src/frontend/contexts/ProjectContext.tsx`, `features/navigation/TabManager.tsx`, `useProjectActions.ts`; `tests/test_document_state.py`, `tests/test_project_rename.py`, projects/context `__tests__/` |
| Memory extraction/review/scope | `src/lib/memory_store.py`, `memory_extract.py`, `memory_context.py`, `src/backend/routes/memory.py`, `src/frontend/features/memory/` | Chat request project scope and `src/frontend/contexts/ProjectContext.tsx` |
| Model providers, reasoning or prompts | `src/lib/model_provider_store.py`, `reasoning_levels.py`, `prompt_store.py`, `src/backend/routes/models.py`, `src/frontend/features/settings/` | `prompts/` for user seeds, `src/lib/prompts/` for internal instructions; `tests/test_model_provider_store.py`, `tests/test_reasoning_levels.py` |
| Provider usage or OMP model gateway | `src/lib/provider_usage.py`, `omp_source.py`, `src/frontend/features/navigation/ProviderUsage.tsx` | `src/backend/routes/models.py`, `run-omp.sh`; `tests/test_provider_usage.py`, `tests/test_omp_source.py` |
| OMP sessions, embed composer or shared protocol | `src/lib/omp_web.py`, `src/backend/routes/omp.py`, `src/frontend/features/omp/` | `packages/chat-ui/src/protocol.ts`, `AGENTS.md` integration rules; `tests/test_omp_web.py`, omp `__tests__/` |
| Factory tasks, review submission or preview UI | `src/lib/factory.py`, `src/backend/routes/factory.py`, `src/frontend/features/factory/` | `src/frontend/features/editor/diff/review.ts`, `app/_components/deepLink.ts`, `factory.yaml`; `tests/test_factory.py`, factory/diff `__tests__/` |
| Sandbox execution or durable rich outputs | `src/lib/agent_sandbox_adapter.py`, `fake_sandbox_runner.py`, `sandbox_service.py`, `sandbox_store.py`, `sandbox_plot.py` | `src/lib/tools.py`, chat message renderers; `tests/test_sandbox_service.py`, `tests/test_sandbox_runtime_state.py`, `tests/test_sandbox_native_runs.py`, `tests/test_agent_sandbox_adapter.py` |
| Identity, host capability or ingress | `src/backend/identity.py`, `routes/upstream.py`, `omp_web_gate.py`, `src/config.py` | `deploy/`; `tests/test_identity.py`, `tests/test_omp_web_gate.py` |
| Production redeploy | `redeploy.sh`, `src/backend/routes/deploy.py`, `features/settings/SettingsMenu.tsx` | `deploy/lib.sh`, `.docs/deployment.md` |
| Storage schema, change streams or API transport | `db/schema/`, `src/lib/db_schema.py`, `postgres_data_store.py`, `asset_store.py`; `src/backend/sync.py` | `routes/sync.py`, `src/frontend/lib/backend/`; `tests/test_data_store_conformance.py`, `tests/test_sync.py`, `tests/test_postgres_mode_end_to_end.py` |
| Reusable interaction or content rendering | `src/frontend/components/ui/`, `components/content/`, `hooks/`, `packages/chat-ui/` | Colocated `__tests__/`; preserve feature-independent imports and package distribution boundaries |

## Regression navigation

- Use `tests/` for Python route/service/store and access-policy coverage. `tests/conftest.py` and `pyproject.toml` own fixture/test setup: Postgres fixtures (`database_url`, `postgres_pool`, `clean_database`, `data_store`, `asset_store`) and the `route_client` app builder live only in `conftest.py`, in-memory fakes in `tests/fakes.py`. PostgreSQL integration behavior is represented by `test_postgres_mode_end_to_end.py` and store conformance tests.
- Use colocated `src/frontend/**/__tests__/` for browser logic/components, including shared package tests. Configuration lives at `src/frontend/vitest.config.ts` and `vitest.setup.ts`.
- Use `src/c4/operations.test.ts` for LikeC4 source-edit/layout regressions; this service has its own `package.json`, separate from the frontend package.
- Use `factory.yaml` to locate the repository-wide factory gate and preview process configuration, not to infer factory engine internals. Keep generated factory output and runtime worktrees out of source ownership documentation.

## Refresh evidence and scope

- Refresh with Repomix 1.18.0 compressed, parsable XML, default ignore filtering and security scanning enabled. The inspected final inventory contains 620 files across backend routes/services, browser features/shared layers/package, LikeC4, schemas, launchers, deployment, prompts, maintained docs, and Python/browser/service regressions.
- Exclude only these three generated context docs from documentation inputs. Retain maintained guidance through the [repository documentation index](../AGENTS.md#documentation). Exclude dependencies, lockfile bodies, public assets, fonts, images, caches and Tauri build output. The Tauri host project (`src/frontend/src-tauri/`) is not in the repository, only its sidecar entry, provider and build script.
- Treat compressed declarations as navigation evidence, not proof of behavior. Check original implementations for persistence authority, revision conflicts, access precedence, capability flags, integration defaults/errors, environment loading, and sandbox checkpoints. External engine/fork internals and deployed service behavior are outside this source refresh.
