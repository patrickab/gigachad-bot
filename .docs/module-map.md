Generated from d6bc359 on 2026-10-22

Working-tree caveat: generated from the inspected working tree, including uncommitted documentation migration, README, and instruction updates. The revision identifies the source baseline, not a reproducible committed snapshot.

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
|       `-- 0007_model_reasoning_levels.sql
|-- src/
|   |-- config.py                     Backend configuration and PostgreSQL pool
|   |-- backend/
|   |   |-- server.py                 FastAPI composition, lifecycle and shared error mapping
|   |   |-- identity.py               Request identity and host-access capability
|   |   |-- canvas.py                 Transactional collaborative canvas mutation authority
|   |   |-- sync.py                   PostgreSQL notification broker
|   |   |-- omp_web_gate.py           Identity-checking frame proxy
|   |   |-- install_deps.py           gigachad-install entry point
|   |   `-- routes/                   HTTP, SSE and WebSocket boundaries
|   |       |-- deps.py               Identity-scoped store/service injection and stream helpers
|   |       |-- schemas.py            Shared file-route request/response models
|   |       |-- chat.py               Chat generation and tools
|   |       |-- histories.py          Chat persistence, branching and merge endpoints
|   |       |-- documents.py          Logical/project documents and library attachment APIs
|   |       |-- files.py              Chat uploads and extraction APIs
|   |       |-- fileviewer.py         Generic text/raw preview APIs
|   |       |-- file_roots.py         Mounted files, git diffs and worktrees
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
|   |       `-- upstream.py           Host-access dependency and adapter error mapping
|   |-- lib/                          Python domain services, stores and external adapters
|   |   |-- data_store.py             Storage contract, logical keys and errors
|   |   |-- postgres_data_store.py    PostgreSQL text/document implementation
|   |   |-- asset_store.py            PostgreSQL binary assets
|   |   |-- storage_namespace.py      Canonical database namespace names
|   |   |-- db_schema.py              Checksum-protected migration application
|   |   |-- chat_store.py             Chat I/O, validation, branching and indexing
|   |   |-- project_store.py          Catalog, slugs, kanban and tab persistence
|   |   |-- memory_store.py           Memory extraction/review/persistence
|   |   |-- model_provider_store.py   Provider catalog and model defaults
|   |   |-- prompt_store.py           User prompt storage and seed/block loading
|   |   |-- file_roots.py             Mounted-file registry and filesystem policy
|   |   |-- git_diff.py               Diff listing, patches and file content
|   |   |-- git_worktree.py           Mounted repository worktree operations
|   |   |-- architecture_workspace.py Disk/database architecture source storage
|   |   |-- c4_service.py             LikeC4 service HTTP client
|   |   |-- attachment_materialize.py Shared attachment/PDF materialization
|   |   |-- document_library.py       MIME, library keys and mounted-path helpers
|   |   |-- extract_queue.py          Background PDF extraction queue
|   |   |-- mineru.py                 MinerU client/process lifecycle
|   |   |-- image_paths.py            Chat image payload resolution
|   |   |-- toolcalling.py            Generic tool-turn runtime
|   |   |-- tools.py                  Application tool bindings
|   |   |-- web_search.py             Brave query planning and evidence retrieval
|   |   |-- research.py               GPT-Researcher adapter
|   |   |-- sandbox_service.py        Workspace/output orchestration
|   |   |-- sandbox_store.py          Durable sandbox metadata
|   |   |-- agent_sandbox_adapter.py  Native sandbox runner adapter
|   |   |-- sandbox_plot.py           Plot generation
|   |   |-- omp_source.py             OMP auth-gateway model integration
|   |   |-- omp_web.py                omp-web URL/response-contract owner
|   |   |-- factory.py                External factory engine adapter
|   |   |-- upstream.py               Shared server-to-server request/validation plumbing
|   |   |-- provider_usage.py         OMP subscription usage reader/cache
|   |   |-- neovim_session.py         Neovim PTY/RPC lifecycle
|   |   `-- prompts/                  Internal, non-user-facing prompts
|   |       |-- internal.py
|   |       `-- non_user_prompts.py
|   |-- c4/                           Separate stateless TypeScript LikeC4 service
|   |   |-- server.ts                 Loopback JSON API
|   |   |-- workspace.ts              Parse/project models and syntax-tree lookups
|   |   |-- operations.ts             Surgical source edits
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
|       |       |-- TabContent.tsx    Active surface composition
|       |       |-- deepLink.ts       Preview/open-target parsing
|       |       `-- MemoryCommandLayer.tsx
|       |-- features/                 Domain-owned interactive surfaces
|       |   |-- chat/                 Turns, attachments, persistence and streaming
|       |   |   |-- ChatContainer.tsx
|       |   |   |-- attachments.ts
|       |   |   |-- composer/         Input, completions, drawings and attachment chips
|       |   |   |-- messages/         Tools, research and rich output rendering
|       |   |   `-- preview/          Attachment previews
|       |   |-- editor/               Documents, viewers and terminal editors
|       |   |   |-- DocumentEditor.tsx
|       |   |   |-- TextDocumentEditor.tsx Markdown/text save and preview owner
|       |   |   |-- FileViewer.tsx
|       |   |   |-- NeovimEditor.tsx
|       |   |   |-- architecture/     LikeC4 graph/source editing
|       |   |   |   `-- ArchitectureWindow.tsx
|       |   |   |-- canvas/           Canvas entities, collaboration, history and camera
|       |   |   |   |-- CanvasEditor.tsx
|       |   |   |   `-- useCollaborativeCanvas.ts
|       |   |   `-- diff/             Diff display and review drafts
|       |   |       |-- DiffViewer.tsx
|       |   |       `-- review.ts
|       |   |-- navigation/           Sidebar, tabs, commands, mounts and worktrees
|       |   |   |-- Sidebar.tsx
|       |   |   |-- TabManager.tsx
|       |   |   `-- useWorktrees.tsx
|       |   |-- omp/                  Single-frame surface, bridge, composer and sessions
|       |   |   |-- OmpProvider.tsx
|       |   |   `-- OmpComposer.tsx
|       |   |-- factory/              Boards, task details, settings and previews
|       |   |   |-- FactoryBoard.tsx
|       |   |   `-- TaskView.tsx
|       |   |-- memory/               Memory window, review panel and memory commands
|       |   |   |-- MemoryViewer.tsx  Topic overview, topic page, regroup review
|       |   |   |-- MemoryPanel.tsx   Single-screen review of proposed operations
|       |   |   `-- useMemorizeFlow.ts Extraction, review and commit orchestration
|       |   |-- projects/             Project dashboard/kanban
|       |   |   `-- ProjectDashboard.tsx
|       |   `-- settings/             Model, reasoning, prompts, preferences and theme
|       |       `-- SettingsMenu.tsx
|       |-- contexts/                 Shared project/settings/tab/undo/host-scale state
|       |   |-- ProjectContext.tsx
|       |   |-- StartupReadContext.tsx Bounded initial reads, failure banner and targeted retry
|       |   |-- TabContext.tsx
|       |   `-- UndoDeleteContext.tsx
|       |-- components/
|       |   |-- ui/                   Reusable interaction primitives
|       |   |   |-- Dialog.tsx
|       |   |   |-- FloatingWindow.tsx
|       |   |   `-- file-tree/        Shared file tree component and hook
|       |   `-- content/              Markdown, code, PDF and plot rendering
|       |       |-- LaTeXMarkdown.tsx
|       |       |-- PdfViewer.tsx
|       |       `-- PlotElement.tsx
|       |-- hooks/                    Feature-independent interaction hooks
|       |   `-- useEscapeLayer.ts
|       |-- lib/
|       |   |-- backend/              HTTP, URL, runtime API base and stream clients
|       |   |   |-- api.ts
|       |   |   |-- config.ts
|       |   |   |-- urls.ts
|       |   |   |-- canvasStream.ts
|       |   |   `-- syncStream.ts
|       |   |-- domain/               Browser domain types and pure document/graph logic
|       |   |   |-- types.ts
|       |   |   |-- architecture.ts
|       |   |   |-- canvasDocument.ts
|       |   |   `-- edgeRouting.ts
|       |   |-- paths.ts              File-kind/path helpers
|       |   `-- diff.ts               Diff parsing helpers
|       `-- packages/
|           `-- chat-ui/              Independently packaged local/OMP chat UI contract
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
|   |-- conftest.py                    Shared fixtures
|   |-- test_postgres_mode_end_to_end.py
|   |-- test_data_store_conformance.py
|   |-- test_file_roots.py
|   `-- test_omp_web.py
|-- scripts/
|   `-- setup-postgres.sh              PostgreSQL setup helper
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
- Keep `packages/chat-ui/` independently distributable: no `@/` app imports and no undeclared reliance on the shell's dependencies. `pack.mjs` owns packing/synchronization with the external fork; do not vendor the fork's implementation here.

### Backend and separate services

- Start at `backend/server.py` for route registration/lifecycle, `routes/deps.py` for identity-scoped dependency construction, and `backend/identity.py` for access identity. Routes own transport validation and HTTP/SSE/WebSocket translation.
- Put reusable storage/domain/external-service work in `src/lib/`, not in a second route's handler. `src/lib/` is a flat collection of concrete modules, not separate `services/`, `stores/`, or `adapters/` directories.
- Use `data_store.py` / `postgres_data_store.py` for app-owned text and `asset_store.py` for bytes; use domain stores for chat/project/memory operations. Keep logical namespaces in `storage_namespace.py`. `backend/canvas.py` is the explicit transactional canvas authority, not a generic document-store helper.
- Route mounted-file work through `lib/file_roots.py`; use `git_diff.py` and `git_worktree.py` for git operations. Keep host paths distinct from database keys and client URLs.
- Split architecture changes by ownership: `architecture_workspace.py` reads/commits sources; `c4_service.py` calls the service; `src/c4/` parses, lays out and edits source in memory; `features/editor/architecture/` handles browser interaction. The C4 service does not own persistence.
- Keep omp-web contract knowledge in `lib/omp_web.py`, factory contracts in `lib/factory.py`, and shared request/validation plumbing in `lib/upstream.py`. Their route adapters map failures; `backend/omp_web_gate.py` is an identity proxy, not another upstream API adapter.
- Treat omp-web, the Agents-in-a-box factory engine, and native sandbox implementation as external dependencies. This repository owns their integration surfaces, not their internal task/session/execution engines.
- Keep sandbox execution and persistence distinct: `sandbox_service.py` stages workspace state, `routes/histories.py` checkpoints it after a successful chat save, and `sandbox_store.py` records durable manifests/runs. See [architecture](architecture.md#sandbox-checkpoint-boundary).
- Add a new numbered SQL migration under `db/schema/` for schema changes; `lib/db_schema.py` records checksums, so applied migration files are not editable history.

## Entry points and source limits

| Entry point | Follow next |
| --- | --- |
| Browser page | `src/frontend/app/page.tsx` -> `app/_components/AppShell.tsx` / `TabContent.tsx`; providers also enter through `app/layout.tsx` |
| Backend app | `src/backend/server.py` -> `routes/` -> `lib/`; settings in `src/config.py` |
| LikeC4 service | `src/c4/server.ts` -> `workspace.ts`, `operations.ts`, `layout.ts` |
| Dependency installer | `gigachad-install` in `pyproject.toml` -> `src/backend/install_deps.py` |
| Development/production launch | Root `run*.sh`, `deploy/`; load operations guidance through [AGENTS.md](../AGENTS.md#documentation) |
| Runtime environment loading | `deploy/load-env.sh` -> root launchers / production runners; configuration consumers in `src/config.py` and individual adapters |
| Factory integration | `factory.yaml`, `run-factory.sh`, `src/lib/factory.py`, `src/backend/routes/factory.py`, `src/frontend/features/factory/` |

## Where to change

Full paths are repository-relative. Abbreviated sibling filenames share the preceding directory. `features/`, `contexts/`, `app/`, `components/`, `hooks/`, `packages/`, and browser `lib/backend/` paths resolve under `src/frontend/`. Backend `routes/` resolves under `src/backend/`. Pair UI changes with the transport/domain owner when behavior crosses the API boundary.

| Change | Primary source owner | Adjacent contracts and regression locations |
| --- | --- | --- |
| Shell layout, deep links, surface selection | `src/frontend/app/_components/AppShell.tsx`, `TabContent.tsx`, `deepLink.ts` | `features/navigation/TabManager.tsx`, `contexts/TabContext.tsx`, `app/_components/__tests__/deepLink.test.ts` |
| Chat generation, streaming or tool behavior | `src/backend/routes/chat.py`, `src/lib/toolcalling.py`, `src/lib/tools.py` | `src/frontend/features/chat/useChatStream.ts`, `useChat.ts`; `tests/test_tools.py`, chat `__tests__/useChatStream.test.tsx` |
| Chat save, branches or merge | `src/lib/chat_store.py`, `src/backend/routes/histories.py` | `features/chat/useChatPersistence.ts`, `features/navigation/BranchContext.tsx`; `tests/test_histories.py`, `tests/test_chat_store_message_preservation.py` |
| Attachment live references, uploads or PDF extraction | `src/frontend/features/chat/attachments.ts`, `src/lib/attachment_materialize.py`, `extract_queue.py`, `mineru.py` | `src/backend/routes/files.py`, `documents.py`, `file_roots.py`; `tests/test_attachment_materialize.py`, `tests/test_extract_queue.py`, chat `__tests__/attachments.test.ts` |
| Text/PDF/plot viewing or editing | `src/frontend/features/editor/DocumentEditor.tsx`, `FileViewer.tsx`, `src/frontend/components/content/` | `src/backend/routes/documents.py`, `fileviewer.py`; `tests/test_document_state.py`, editor/content `__tests__/` |
| Collaborative canvas or canvas entity shape | `src/frontend/features/editor/canvas/`, `src/frontend/lib/domain/canvasDocument.ts` | `src/backend/canvas.py`, `routes/canvases.py`, `lib/backend/canvasStream.ts`; `tests/test_canvas_mutations.py`, canvas `__tests__/` |
| Architecture syntax edits, diagram or layout | `src/c4/operations.ts`, `workspace.ts`, `layout.ts`; `src/frontend/features/editor/architecture/` | `src/lib/architecture_workspace.py`, `c4_service.py`, `src/backend/routes/architecture.py`; `src/c4/operations.test.ts`, `tests/test_architecture_workspace.py` |
| Mount browsing or filesystem access policy | `src/lib/file_roots.py`, `src/backend/routes/file_roots.py` | `features/navigation/FileMountPicker.tsx`, `components/ui/file-tree/`; `tests/test_file_roots.py` |
| Git diffs, review drafts or worktree switching | `src/lib/git_diff.py`, `git_worktree.py`; `src/frontend/features/editor/diff/`, `features/navigation/useWorktrees.tsx` | `src/backend/routes/file_roots.py`; `tests/test_git_diff.py`, `tests/test_git_worktree.py`, diff `__tests__/` |
| Neovim editing or terminal save bridge | `src/lib/neovim_session.py`, `src/backend/routes/neovim.py`, `src/frontend/features/editor/NeovimEditor.tsx` | `tests/test_neovim.py`; filesystem/identity policy stays with its existing owners |
| Projects, kanban or persisted tabs | `src/lib/project_store.py`, `src/backend/routes/projects.py`, `src/frontend/features/projects/` | `src/frontend/contexts/ProjectContext.tsx`, `features/navigation/TabManager.tsx`; `tests/test_document_state.py`, projects/context `__tests__/` |
| Memory extraction/review/scope | `src/lib/memory_store.py`, `src/backend/routes/memory.py`, `src/frontend/features/memory/` | Chat request project scope and `src/frontend/contexts/ProjectContext.tsx` |
| Model providers, reasoning or prompts | `src/lib/model_provider_store.py`, `prompt_store.py`, `src/backend/routes/models.py`, `src/frontend/features/settings/` | `prompts/` for user seeds, `src/lib/prompts/` for internal instructions; `tests/test_model_provider_store.py`, `tests/test_reasoning_levels.py` |
| Provider usage or OMP model gateway | `src/lib/provider_usage.py`, `omp_source.py`, `src/frontend/features/navigation/ProviderUsage.tsx` | `src/backend/routes/models.py`, `run-omp.sh`; `tests/test_provider_usage.py`, `tests/test_omp_source.py` |
| OMP sessions, embed composer or shared protocol | `src/lib/omp_web.py`, `src/backend/routes/omp.py`, `src/frontend/features/omp/` | `packages/chat-ui/src/protocol.ts`, `AGENTS.md` integration rules; `tests/test_omp_web.py`, omp `__tests__/` |
| Factory tasks, review submission or preview UI | `src/lib/factory.py`, `src/backend/routes/factory.py`, `src/frontend/features/factory/` | `src/frontend/features/editor/diff/review.ts`, `app/_components/deepLink.ts`, `factory.yaml`; `tests/test_factory.py`, factory/diff `__tests__/` |
| Sandbox execution or durable rich outputs | `src/lib/agent_sandbox_adapter.py`, `sandbox_service.py`, `sandbox_store.py`, `sandbox_plot.py` | `src/lib/tools.py`, chat message renderers; `tests/test_sandbox_service.py`, `tests/test_sandbox_runtime_state.py`, `tests/test_agent_sandbox_adapter.py` |
| Identity, host capability or ingress | `src/backend/identity.py`, `routes/upstream.py`, `omp_web_gate.py`, `src/config.py` | `deploy/`; `tests/test_identity.py`, `tests/test_omp_web_gate.py` |
| Storage schema, change streams or API transport | `db/schema/`, `src/lib/db_schema.py`, `postgres_data_store.py`, `asset_store.py`; `src/backend/sync.py` | `routes/sync.py`, `src/frontend/lib/backend/`; `tests/test_data_store_conformance.py`, `tests/test_sync.py`, `tests/test_postgres_mode_end_to_end.py` |
| Reusable interaction or content rendering | `src/frontend/components/ui/`, `components/content/`, `hooks/`, `packages/chat-ui/` | Colocated `__tests__/`; preserve feature-independent imports and package distribution boundaries |

## Regression navigation

- Use `tests/` for Python route/service/store and access-policy coverage. `tests/conftest.py` and `pyproject.toml` own fixture/test setup; PostgreSQL integration behavior is represented by `test_postgres_mode_end_to_end.py` and store conformance tests.
- Use colocated `src/frontend/**/__tests__/` for browser logic/components, including shared package tests. Configuration lives at `src/frontend/vitest.config.ts` and `vitest.setup.ts`.
- Use `src/c4/operations.test.ts` for LikeC4 source-edit/layout regressions; this service has its own `package.json`, separate from the frontend package.
- Use `factory.yaml` to locate the repository-wide factory gate and preview process configuration, not to infer factory engine internals. Keep generated factory output and runtime worktrees out of source ownership documentation.

## Refresh evidence and scope

- Refresh with Repomix 1.18.0 compressed, parsable XML, default ignore filtering and security scanning enabled. The inspected final inventory contains 428 files across backend routes/services, browser features/shared layers/package, LikeC4, schemas, launchers, deployment, prompts, maintained docs, and Python/browser/service regressions.
- Exclude only these three generated context docs from documentation inputs. Retain maintained guidance through the [repository documentation index](../AGENTS.md#documentation). Exclude dependencies, outputs/caches, public/generated assets, fonts, lockfile bodies, credentials/runtime state, prior packs, and unsupported integration remnants from architecture evidence.
- Treat compressed declarations as navigation evidence, not proof of behavior. Check original implementations for persistence authority, revision conflicts, access precedence, capability flags, integration defaults/errors, environment loading, and sandbox checkpoints. External engine/fork internals and deployed service behavior are outside this source refresh.
