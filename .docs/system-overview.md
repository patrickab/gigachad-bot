Generated from d6bc359 on 2026-10-22

Working-tree caveat: generated from the inspected working tree, including uncommitted documentation migration, README, and instruction updates. The revision identifies the source baseline, not a reproducible committed snapshot.

# System overview

## Purpose and concepts

GigaChat Bot is a self-hosted, local-first LLM workspace for conversations, research, documents, visual thinking, and coding agents. Connect cloud or local models and keep histories and artifacts on infrastructure you control.

| Concept | Meaning |
| --- | --- |
| Chat | A model conversation with its own history, branches, attachments, and optional tools |
| Project | A collection of conversations, documents, canvases, architecture, project memory, and a lightweight task board; optionally linked to a mounted code directory |
| Document | An app-owned artifact or a reference to an existing mounted file; references edit the original rather than an independent copy |
| File mount | A backend directory made available for browsing, editing, git review, and worktree selection; it is not a browser-local folder |
| Canvas | An infinite drawing workspace with ink, notes, page/image frames, and document windows |
| Architecture workspace | A project's merged LikeC4 model, editable as diagrams and `.c4` source |
| OMP session | An external coding-agent conversation provided by omp-web, distinct from local model chat |
| Factory task | An agent workflow running in its own git worktree, with review, gates, messages, and optional previews |
| Memory | Reviewed, document-backed facts supplied to later chats, globally or within a project; not vector search |

Use [architecture](architecture.md) for implementation, persistence, access boundaries, integration contracts, and failure semantics. Use the [module map](module-map.md) to locate source owners. Find operational instructions through the [repository documentation index](../AGENTS.md#documentation).

## Runtime and prerequisites

- Run the development workspace on Linux with Bash, Python 3.12 or 3.13, `uv`, and running Docker with Compose accessible to the user. The launcher installs dependencies, including Node.js; the default local MinerU OCR stack makes first installation large.
- Let the user start `./run.sh` or the individual launchers. Coding agents must not start development servers; factory-managed task previews are the exception, started by the engine itself.
- Open the development UI at `localhost:2999`; the API runs at `127.0.0.1:8001`. LikeC4, omp-web, and factory are supporting services, not alternate user workspaces.
- Startup reuses PostgreSQL credentials and data. If `POSTGRES_PASSWORD_DEV` is missing, setup asks before generating credentials and creating the database; declining stops startup without changes. Restore an existing database's original password rather than replacing it.
- Keep provider credentials in private `~/.secrets` or the environment. Optional nonsecret configuration belongs in `${XDG_CONFIG_HOME:-$HOME/.config}/gigachad-bot/env`.
- OMP tabs require `omp` on backend `PATH` and access to the private `patrickab/ompweb` fork. Startup can provision its checkout; set `GIGACHAD_OMP_WEB=0` to skip it.
- The factory launcher can provision its separate engine checkout, configuration, and database while preserving existing state. Set `GIGACHAD_FACTORY=0` to skip it. Running sandboxed tasks still requires the sandbox runtime setup.
- Keep production backend services private. Development launchers are not production deployment. Remote inference, identity, production setup, and service updates belong to the maintained runbooks indexed in [AGENTS.md](../AGENTS.md#documentation).
- Failed startup reads show a shared warning after an error or a 15-second timeout. Retry reloads only failed data without losing drafts or already-loaded content.

## Model chat, context, and projects

- Select cloud/provider models through LiteLLM or local Ollama models. Configure model defaults and editable system prompts for the task; supported models expose reasoning controls.
- Distinguish OMP-backed models in local chat from OMP coding-session tabs. The former use OMP's model gateway through LiteLLM, the latter use the separate omp-web service.
- Stream replies, revisit saved histories, branch from a message, and merge branches back into a conversation.
- Organize related chats and artifacts in a project. Use its task board for lightweight planning; factory boards run agent workflows and are separate.
- Keep each chat's project membership independent of the currently selected project elsewhere. Its containing project determines project memory, even when another tab is active.
- Attach uploads, mounted files, global/project documents, architecture sources, or drawings. Toggle attachments out of model context without deleting their source; open them to preview or edit.
- Create project documents in their intended project. Reuse generated tool artifacts as documents and retain source references from web research.
- Remove chat turns or attachments from context with undo available; resource deletion also offers a deferred undo window. Canvas editing has its own history.
- Inspect provider subscription usage when available: five-hour limits are distinct from weekly/monthly windows. Missing five-hour data is not replaced with a weekly value.

### Live references versus snapshots

| Attachment | What later sends use |
| --- | --- |
| Active text/document reference | Re-read source content when sending and reopening the chat; inactive references stay excluded until enabled |
| Missing or ambiguously moved reference | Last known text as a read-only snapshot; opening the preview retries the source rather than recreating it |
| Uniquely relocated mounted file | Follow the moved file when its original mount has exactly one matching filename |
| Referenced PDF | Shared extracted text; changed bytes trigger replacement OCR, but the previous extraction remains usable while it runs, so an early send may use old text |
| Pasted image, drawing, canvas image, architecture attachment | Snapshot; reattach to include subsequent changes |

- Save text and task-list changes to the source before updating chat context. PDFs and images are not text-editable attachments.
- Other source-read failures retain existing attachment state rather than discarding the last good content. Detailed refresh, extraction, and retry contracts belong in [architecture](architecture.md).

## Files, PDFs, Study, and research

- Mount an existing backend directory, including an Obsidian vault, to work on its files in place. Browse its filtered tree and open the type-aware text, PDF, canvas, architecture, plot, or mind-map view.
- Add a mounted file to a project as a reference. Editing changes the original disk file; removing the project reference does not delete that source.
- Upload PDFs or attach mounted PDFs, view their pages, and use MinerU extraction to supply readable Markdown to chat. Image OCR can transcribe text and LaTeX with the configured vision model.
- Keep PDF library names unique within your account. Uploading or syncing another PDF with the same filename replaces the shared library entry, rather than creating a separate library item. Extraction is also keyed by filename stem.
- For study, attach extracted lecture/document material, choose a tutoring prompt, and ask for explanations, active-recall material, diagrams, or mind maps. The current sources do not expose a separate automatic **Study** generation workflow.
- Enable tools for the current conversation rather than assuming every reply uses them. Use web search for current or source-dependent facts and deep research for broad, multi-part investigations; deep research can take minutes.
- Web search uses Brave source evidence and the selected answering model. Deep research uses GPT-Researcher with a DuckDuckGo retriever and configurable research models, depth, and breadth.
- Generate Mermaid diagrams and mind maps in chat. Use the sandbox plot tool for interactive charts or opt into the workspace agent for file/code work in a persistent chat-scoped sandbox.
- Save the conversation to checkpoint its sandbox workspace. Rich outputs are stored during execution, but the latest workspace remains staged in backend memory until the chat save succeeds.
- Optional external services and credentials are real prerequisites. An unavailable integration does not imply a transparent local replacement. See [architecture](architecture.md) for service boundaries and failures.

## Canvas

- Start a scratch canvas, then save it as a named global or project `.canvas` document. Scratch content is device/browser-local until saved; named canvases can synchronize edits across open clients.
- Draw and erase with a pen, pan/zoom, select and transform ink, insert Markdown notes, and arrange page frames and images.
- Keep PDFs, documents, architecture diagrams, and nested canvases alongside the artwork as interactive windows. Open a window for focused work, then return to its containing canvas.
- Capture a canvas region for sharing and attach a drawing snapshot to a model conversation.
- Add page frames to export an A4 PDF. Exports and model-facing raster artwork include ink, text, and image frames, not the contents of live document/PDF/architecture/nested-canvas windows.
- Use independent undo/redo for the canvas being edited. Nested canvas history is separate; file-window navigation is not an artwork undo step.
- Canvas nesting is limited to prevent recursive file cycles. A failed named-canvas load leaves stored content untouched; interrupted synchronization exposes retry and retries changes automatically. See [architecture](architecture.md) for collaboration and persistence guarantees.

## Architecture workspaces

- Open a project's architecture and work either in the diagram or its `.c4` sources. Systems, modules, and saved views share one merged LikeC4 model per project.
- Use a linked mount's `docs/architecture/` when present; otherwise use the project's app-owned architecture sources. Linking matches the mount folder name to the project slug. Switching a mount's worktree does not redirect the linked clone's architecture source.
- Create elements, relations, modules, and saved views. Distinguish removing an element from a view from deleting it from the model.
- Use source text for unsupported visual edits and compound descriptions. Views based on filters/styles are read-only on the diagram canvas.
- Expect relations to appear in any view containing both endpoints. Saved layout retains intentional placement; invalid saved layout can fall back to automatic layout.
- Attach path-labelled source snapshots to chat, or select individual sources as documents.
- Invalid edits are rejected with file/line diagnostics. Concurrent source changes require re-reading and revalidating rather than overwriting newer content. See [architecture](architecture.md) for revision rules, reference cleanup, and the editing pipeline.

## Mounted git review, comments, and Neovim

- Open a mount's diff to compare working-tree changes with `HEAD`; navigate changed files in unified or split view.
- Select code in one file and, in split view, one side to add a review note. Notes retain selected code/context per worktree across reloads.
- Copy ordinary diff notes without clearing drafts. For factory tasks in `human_review` or `blocked`, send notes as feedback with the selected code and current locations/context.
- When anchors cannot be resolved uniquely, preserve original locations and mark the note outdated. Failed sends retain notes; successful sends clear only submitted notes that have not changed. Sent reviews appear in task messages, not inline reply threads.
- Optionally edit a diff in Neovim with read-only `HEAD` beside the working file. Save with `:w` and return with `:qa`. Deleted, binary, oversized, and symlinked diff files cannot use this path.
- Enable Neovim per device when available: host-access identity, `nvim` on backend `PATH`, and an existing `$XDG_CONFIG_HOME/nvim` configuration directory (default `~/.config/nvim`). Otherwise use the built-in editor.
- Neovim `:w` uses the document's normal save path, including architecture validation; failed saves leave the buffer modified. Preview/fullscreen switches preserve the session, but closing/quitting/disconnecting terminates it and loses unsaved edits. While focused, its terminal owns all keys, including Escape.

## Worktrees

- Switch a mounted git clone between worktrees without renaming its mount. The file tree, diff, and editor follow the selection, which persists per mount.
- Find the remote default through `origin/HEAD`, falling back to the main checkout's branch. Reuse or create that checkout without pulling or switching the main checkout merely to find it.
- Create a worktree from `dev` by default, otherwise the remote default. Creation fetches and fast-forwards the checked-out base before creating `~/git/worktrees/<repo>/<name>`; existing branches or folders are refused.
- Require explicit force to remove a dirty worktree. Branch deletion is separate, unmerged deletion requires confirmation, and remote branches are never deleted by this workflow.

## omp tabs (omp-web)

- Open external OMP coding sessions as workspace tabs. Sessions for a project's linked code mount appear under code; other workspaces appear under OMP.
- Navigate session forks and in-session branches, restore archives, and open returned file links under the shell's file-access policy.
- Use the shell composer for prompts, images, documents, abort, model/thinking selection, and `/` or `@` completion. Omp-web owns execution, queueing, shell commands, and session creation.
- Shell composition does not provide local-chat word prediction or busy-run steering/queueing. Persisted OMP messages cannot be deleted, and frame DOCX/PDF previews are blocked by `SAMEORIGIN`.
- OMP features require host access and a running compatible omp-web service. Matching shell/fork versions matter; restart after updates. Integration ownership, shared rendering/bridge contracts, and deployment compatibility belong in [architecture](architecture.md); operational steps are indexed in [AGENTS.md](../AGENTS.md#documentation).

## Agent factory

The independent Agents-in-a-box engine runs agent workflows in task git worktrees. GigaChat provides repository boards, task messages, review findings, gates, diffs, and previews.

1. Open a repository's factory board and register the repository if needed.
2. Choose a workflow and submit a request. Use native OMP `provider/model` overrides, not `litellm_proxy/` model names.
3. Follow phases, findings, gate results, messages, and changes. Approve, retry, cancel, or send feedback when the task's state permits.
4. Review task diffs and send selected-code comments during human review or when blocked. Host users can inspect admitted factory worktrees without first creating a mount.
5. Open a preview when the workflow provides one. Preview data starts empty; a task scenario can seed representative content and select a chat, document, canvas, or architecture deep link.

- Agent/default configuration affects new tasks, not already-running tasks. Repository conventions and workflow configuration determine agents, sandboxing, hooks, gates, and preview processes; see [architecture](architecture.md) for precedence and the repository contract.
- The engine alone starts preview services on ports 8100–8199. Their backend uses development identity and an empty database. Remote production previews must be disabled because remote browsers cannot reach host-loopback ports.
- Factory features require host access, the separate engine, and its configured sandbox prerequisites. Engine refusals remain refusals; a disconnected engine does not silently turn into an in-process task runner.

## Memory system

- Keep personal facts/preferences in global memory and project facts in project memory. Configure categories, edit entries, move them between scopes, and remap orphaned entries through the app.
- Extract proposed facts and corrections from a chat with the configured memory model. Global and project extraction are independent.
- Accept, edit, or reject candidates before reconciling them into affected categories. Review the merged result before committing; untouched categories remain unchanged.
- Commit to update canonical memory and advance that chat/scope's extraction position. Cancel without advancing it, so the same material remains eligible later.
- Later local chats receive global memory plus their containing project's memory in the system prompt. Memory is explicit document-backed context, not automatic retrieval from every past conversation. Detailed storage and watermark contracts belong in [architecture](architecture.md).

## User-visible access and failure limits

- Require host access for OMP, factory, Neovim, and files outside ordinary mounts. Other identities remain restricted to mounted roots. Outside those roots, credential-store protection limits host file access. Do not mount secret directories expecting a blanket denylist to protect their contents.
- Mounted trees omit ignored/internal directories. Text editing refuses non-UTF-8 data rather than risking a lossy binary overwrite.
- A reference is not a backup of the original file; missing references retain only their last available snapshot. PDF extraction may lag a changed file, and visual attachments do not update a previously sent snapshot automatically.
- Model/provider, OCR, research, sandbox, architecture, OMP, and factory services have independent prerequisites. Service unavailability, invalid upstream responses, rejected edits, and failed saves must remain visible rather than being represented as successful empty results.
- Keep operational recovery and exact status/error/security contracts in [architecture](architecture.md) and the maintained runbooks linked from [AGENTS.md](../AGENTS.md#documentation).
