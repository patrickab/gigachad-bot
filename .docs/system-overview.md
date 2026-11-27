Generated from bd63559a on 2026-11-25. Corrected by hand on 2026-11-27 after the complexity cleanup (removed files and names dropped, new files added; no `/repomix` run). Edited by hand on 2026-11-27 after the reuse pass (no `/repomix` run).
Reduced by hand on 2026-11-25.

# System overview

## Purpose and concepts

GigaChad Bot is a self-hosted, local-first LLM workspace for conversations, research, documents, visual thinking, and coding agents. Connect cloud or local models and keep histories and artifacts on infrastructure you control.

| Concept | Meaning |
| --- | --- |
| Chat | A model conversation with its own history, branches, attachments, and optional tools |
| Project | Conversations, documents, canvases, architecture, project memory, and a lightweight task board; optionally linked to a mounted code directory |
| Document | An app-owned artifact or a reference to an existing mounted file; references edit the original, not a copy |
| File mount | A backend directory made available for browsing, editing, git review, and worktree selection; not a browser-local folder |
| Canvas | An infinite drawing workspace with ink, notes, page/image frames, and document windows |
| Architecture workspace | A project's merged LikeC4 model, editable as diagrams and `.c4` source |
| Pi session | A coding-agent conversation run by Pi through pi-host, distinct from local model chat |
| Factory task / wave | A task is a Pi coding session that the agent-factory package drives through gates, review and merge in its own git worktree, with an optional preview. A wave is a set of tasks built on one branch, reviewed once and merged as a whole |
| Workbench | One `.md` file or LaTeX project as Neovim beside its rendered view with a Pi chat drawer; files live on disk and the agent edits them there |
| Learn session | A LearnHarness teaching session (probe, plan, teach, recall), host-only |
| Memory | Durable facts as plain files in a memory-store root, offered to chats as tools and a prompt index; not vector search |
| Share link | A revocable secret URL granting one stored resource (today a canvas, edited together) to whoever holds it |

Implementation, persistence, access boundaries, contracts and failure semantics: [architecture](architecture.md). Source owners: [module map](module-map.md). Operational instructions: [repository documentation index](../AGENTS.md#documentation).

## Prerequisites

- Linux with Bash, Python 3.12 or 3.13, `uv`, and running Docker with Compose accessible to the user. The launcher installs dependencies, including Node.js; the default local MinerU OCR stack makes first installation large.
- Start with `./run.sh` or the individual launchers (agents: [AGENTS.md](../AGENTS.md#running-the-dev-server)). UI at `localhost:2999`, API at `127.0.0.1:8001`. LikeC4, pi-host and the preview gate are supporting services, not alternate workspaces.
- Startup reuses PostgreSQL credentials and data. If `POSTGRES_PASSWORD_DEV` is missing, setup asks before generating credentials and creating the database; declining stops startup without changes. Restore an existing database's original password rather than replacing it.
- Provider credentials go in private `~/.secrets` or the environment; optional nonsecret configuration in `${XDG_CONFIG_HOME:-$HOME/.config}/gigachad-bot/env`.
- Pi tabs need `pi` at the version `pi-host/package.json` pins on the backend's `PATH`. `run.sh` starts pi-host (`GIGACHAD_PI_HOST=0` skips it) and the factory preview gate (`GIGACHAD_FACTORY=0` skips it).
- Neovim, per device: host-access identity, `nvim` on backend `PATH`, and an existing `$XDG_CONFIG_HOME/nvim` directory (default `~/.config/nvim`). Otherwise the built-in editor is used.
- Keep production backend services private. Development launchers are not production deployment. Host-access users can trigger a production redeploy from settings. Remote inference, identity and production setup are in the runbooks indexed in [AGENTS.md](../AGENTS.md#documentation).
- Failed startup reads show a warning; retry reloads only failed data and keeps drafts.

## Model chat and projects

- Select models from providers Pi has a login or API key for, or local Ollama models; defaults and system prompts are editable and supported models expose reasoning controls. Every local-chat model, Ollama included, answers through pi-host's completion gateway; Pi coding-session tabs are separate.
- Stream replies, revisit histories, branch from a message, merge branches back. A request on a dead connection is sent again (at worst a reply is generated twice); a transport failure shows in the reply.
- Projects organize chats and artifacts, with a lightweight task board (separate from factory boards). A chat's containing project, not the selected one, determines its project memory.
- Rename projects, learn sessions, project files and mounted files from the sidebar. Every reference follows, open tabs on all devices move, an open Neovim keeps unsaved text, a name in use is refused, a LaTeX project rebuilds. A project rename moves its chats, documents, canvases, architecture, memory and attachments to the new slug; a taken slug is refused.
- Archive projects (and, per device, Code directories); restore from the archive list. A tab picker (Space outside text fields) searches open tabs; Delete at the search's end or Alt+W closes the highlighted tab, Alt+T opens a background tab, and the picker stays open.
- Attach uploads, mounted files, documents, architecture sources, or drawings. Toggle attachments out of context without deleting them; open them to preview or edit. Generated tool artifacts can be reused as documents; web research keeps source references.
- Remove turns or attachments from context with undo; resource deletion has a deferred undo window.
- Deleting a project, chat (with branches), canvas, project file or chat upload also deletes what derived from it (uploads, library PDF copy, annotation canvas) once nothing else uses it. Annotations of a mounted PDF stay while the file keeps its bytes; extracted Markdown and images are kept; mounted disk files are never deleted. Deleting a workbench document removes its whole folder from disk (undo only until the deferred delete fires).
- Provider subscription usage is shown when available; five-hour limits stay distinct from weekly/monthly windows, and missing five-hour data is not replaced by a weekly value.

### Live references versus snapshots

| Attachment | What later sends use |
| --- | --- |
| Active text/document reference | Source re-read when sending and reopening; inactive references stay excluded until enabled |
| Missing or ambiguously moved reference | Last known text, read-only; opening the preview retries the source |
| Uniquely relocated mounted file | Follows the move when the original mount has exactly one matching filename |
| Referenced PDF | Shared extracted text; changed bytes trigger replacement OCR while the previous extraction stays usable, so an early send may use old text |
| Pasted image, drawing, canvas image, architecture attachment | Snapshot; reattach to include later changes |

- Save text and task-list changes to the source before updating chat context. PDFs and images are not text-editable attachments. Other source-read failures keep the last good content. Contracts: [architecture](architecture.md).

## Files, PDFs, study, and research

- Mount a backend directory (including an Obsidian vault) to work on its files in place, with type-aware views: text, PDF, canvas, architecture, plot, mind map, images. Tool artifacts (plots, mind maps, Mermaid) open rendered, text secondary.
- Add a mounted file to a project as a reference; editing changes the original, removing the reference does not delete it.
- Upload files into a project and drag them to another project (users follow). Drag one onto a Code folder to move it to disk, where it behaves like a mounted file. Uploaded and moved PDFs get MinerU extraction.
- Notebooks: open `.ipynb` to read outputs and edit and run cells on a kernel of the chosen environment (a `.venv` near the notebook or the app's own). Agent or other-device edits merge in live; renaming or deleting the file moves or ends its kernel (needs host access).
- With Neovim, a notebook opens split: Neovim holds the whole notebook as one text with a marker line per cell, beside the outputs. `:w` applies and saves; Shift+Enter runs the cell under the cursor and moves on, Ctrl+Enter runs in place. A `:w` that would undo an unseen notebook change is refused ("The notebook changed, write again"). `:q` leaves the notebook alone with a "Start Neovim" button.
- PDFs: view pages; MinerU extraction supplies Markdown to chat (a PDF/Text switch shows it once finished; it has no sidebar row and is deleted with its PDF); image OCR transcribes text and LaTeX with the configured vision model.
- Annotate PDFs with pen and highlighter. Annotations belong to the PDF's content: copies share them, a rename keeps them, changed bytes start empty.
- Ask about a PDF or selected passage in a side chat: a Pi session per PDF that reads it directly with host access, otherwise an unsaved local chat grounded in the extracted text.
- PDF library names are unique per account: another PDF with the same filename replaces the shared entry. Extraction is keyed by filename stem.
- Study: attach extracted material, choose a tutoring prompt, ask for explanations, active-recall material, diagrams, or mind maps.
- Tools are enabled per conversation. Web search (Brave evidence plus the selected model) for current facts; deep research (GPT-Researcher, DuckDuckGo retriever, configurable models, depth and breadth) for broad investigations, which can take minutes.
- Generate Mermaid diagrams and mind maps in chat. Use the sandbox plot tool for interactive charts, or opt into the workspace agent for file/code work in a persistent chat-scoped sandbox. Save the conversation to checkpoint the sandbox workspace; until then the latest workspace stays staged in backend memory.
- Optional external services and credentials are real prerequisites; an unavailable integration has no transparent local replacement.

## Workbench (Markdown and LaTeX)

- With Neovim and host access, `.md` project and mounted files and `.tex`, `.bib`, `.sty`, `.cls` files open in the workbench: Neovim, rendered view, Pi chat drawer. Otherwise they open in the regular editor; notes always do. Switch editor only, split or preview only (remembered per device).
- A stored project document moves to disk on first open (`GIGACHAD_BASE_DIR/workbench/<kind>/<project>/<name>/`) and stays there. Mounted files are edited in place; new LaTeX documents are created on disk.
- Markdown renders live from saved text, including Mermaid and mind-map documents. A failed LaTeX build keeps the last good PDF.
- Visual-mode `<leader>a` quotes selected lines into the chat draft. Each chat keeps a session list shared by all devices. Workbench chats run the user's full Pi, including the cwd's `AGENTS.md`, with a short Markdown or LaTeX prompt appended.
- Every text editor follows its file live: edits by an agent, terminal or another device merge into Neovim's unsaved edits or reload a clean editor. Unsaved edits on lines the incoming change touches or adjoins are replaced without notice, so save before an agent edits those lines. A save never overwrites an unseen agent edit: `:w` merges first and reports "Saved, merged with the agent's edits".

## LaTeX projects and presentations

- A LaTeX project is a folder with a root `.tex` (with `\documentclass`). Opening another of its files from a tab showing the project switches that tab; Neovim keeps every file as a buffer. One chat belongs to the folder.
- The PDF is the root's: the `.tex` named by a `% !TEX root = <path>` line in the first 5 lines, else the file itself if it has `\documentclass`, else `main.tex` of its folder, else the only root document there. Without one, the PDF pane lists nearby root documents and choosing one writes the `% !TEX root` line; without any it says "No root document".
- The PDF rebuilds on open, on every `:w`, agent edit, and change to any file of the folder. Build output is hidden and triggers no rebuild. A build error names its file when it is not the open one.
- Renaming a project file also rewrites references to it in `.tex` and `.bib` files (`\input`, `\include`, `\includegraphics`, `\bibliography`, `\addbibresource`, `% !TEX root`); a failed rewrite leaves the rename in place.
- "Make LaTeX project" (folder button on a workbench `.tex`, one way) turns it into a project folder that lists like a mounted folder; deleting its row only unlists it. Code folders holding a `.tex`, and project folders, offer a create action for `.tex`, `.bib`, `.sty`, `.cls`.
- Paste a PNG into Neovim (`p`, `"+p`, Ctrl+Shift+V) to store it as `imgs/img-<timestamp>.png` and insert an `\includegraphics` (Markdown `![]()`) line below the cursor, in any Neovim editor for a `.tex` or `.md` file on disk; a root without `graphicx` gets `\usepackage{graphicx}`. In other files Neovim shows an error, other image formats paste nothing, text pastes are unchanged.
- Double-click the PDF to jump Neovim to the source line; `<leader>v` scrolls the PDF to the cursor line. Download the built PDF from the top bar.
- New LaTeX documents (files folder `+`) start Blank or from a template set (Beamer presentation, presentation draft, KOMA-Script letter). The project gets its own copy of the set's files, so it builds anywhere (e.g. Overleaf) and may diverge.
- A root using a presentation set's theme is a presentation. With Neovim, Add slide offers searchable primitives with previews, inserted after the frame around the cursor (clicking the gap between PDF pages inserts after the slide above). Ctrl+Shift+V of an image fills the placeholder on the cursor line. Present opens a presenter popup (current and next slide, `\note{}` notes, timer) and optionally a synchronized audience popup.
- LaTeX workbench chats follow the shared slide rules and the set's `STYLE.md`: a presentation is built from primitives; a draft brainstorms with plain frames and `\visual{}` placeholders, polished later by switching theme. Editing a template set follows `templates/latex/AGENTS.md`.

## Canvas

- Start a scratch canvas (browser-local until saved), then save it as a named global or project `.canvas`; named canvases synchronize across open clients.
- Draw and erase with a pen, pan/zoom, select and transform ink, insert Markdown notes, arrange page frames and images. Undo/redo is independent per canvas.
- Keep PDFs, documents, architecture diagrams, and nested canvases as interactive windows. Capture a region for sharing and attach a drawing snapshot to a chat.
- Page frames export an A4 PDF. Exports and model-facing raster artwork include ink, text and image frames, not the contents of live document/PDF/architecture/nested-canvas windows.
- Nesting is limited to prevent cycles. A failed named-canvas load leaves stored content untouched; interrupted synchronization offers retry and retries automatically.
- Share a stored canvas from its top bar (`/share#<secret>`): holders open it full page, edited live with the owner. A new link replaces the old one; revoking ends it at once. The link follows renames and dies with the canvas.

## Architecture workspaces

- Work in a project's diagram or its `.c4` sources; systems, modules and saved views share one merged LikeC4 model.
- A linked mount's `docs/architecture/` is used when present, else the project's app-owned sources. Linking matches mount folder name to project slug; switching the mount's worktree does not redirect the linked clone's source.
- Create elements, relations, modules and saved views; removing from a view differs from deleting from the model. Use source text for unsupported visual edits and compound descriptions; filter/style-based views are read-only on the diagram.
- Relations appear in any view containing both endpoints. Saved layout keeps placement; an invalid one falls back to automatic layout. Attach path-labelled source snapshots to chat, or select sources as documents.
- Invalid edits are rejected with file/line diagnostics; concurrent source changes require re-reading and revalidating. Rules: [architecture](architecture.md).

## Git review, Neovim, and worktrees

- A mount's diff compares working-tree changes with `HEAD` (unified or split). Select code (in split view, one side) to add a review note; notes keep their code/context per worktree across reloads and can be copied without clearing drafts. Notes with no unique anchor keep their original location and are marked outdated. Failed sends retain notes; successful sends clear only submitted, unchanged ones.
- Optionally edit a diff in Neovim with read-only `HEAD` beside the working file (`:w` saves, `:qa` returns). Deleted, binary, oversized and symlinked diff files cannot use this.
- Neovim `:w` uses the document's normal save path, including architecture validation; a failed save leaves the buffer modified. Preview/fullscreen switches keep the session; closing, quitting or disconnecting ends it and loses unsaved edits. While focused, its terminal owns all keys, including Escape.
- The browser is Neovim's clipboard: yanks copy to it, and `p` and `"+p` paste the browser's clipboard, never the Neovim host's. Chrome asks once to allow reading, Firefox and Zen show a paste button; if the browser does not answer within 15 seconds, Neovim shows the reason and pastes the last Neovim yank.
- Paste an https or ssh git link into the Code `+` dialog to clone into `~/git/<repo>` and mount it; an existing folder is refused, a failed clone leaves nothing.
- Switch a mounted clone between worktrees without renaming its mount; tree, diff and editor follow, persisted per mount. The remote default is found via `origin/HEAD`, falling back to the main checkout's branch, without pulling or switching the main checkout.
- A new worktree branches from `dev` by default, else the remote default: the base is fetched and fast-forwarded, then `~/git/worktrees/<repo>/<name>` is created; existing branches or folders are refused.
- Removing a dirty worktree requires explicit force. Branch deletion is separate, unmerged deletion needs confirmation, and remote branches are never deleted.

## Pi tabs

- Pi coding sessions open as workspace tabs. The sidebar lists sessions per repository and mount (worktrees under their main repository) with running, waiting-on-a-dialog and unread state. A session can start in any folder: Git checkouts, `~/git`, `~/dev` folders and mounted folders are offered.
- Branch from any turn into a new session. Returned file links open under the shell's file-access policy; an extension command can open a file too (e.g. `/files`).
- The composer supports prompts, images, abort, model/thinking selection, and `/` or `@` completion. While the agent runs, Enter steers and Alt+Enter queues a follow-up; Esc or Stop aborts.
- The transcript folds agent-directed content (skills, compactions, factory notices, subagent results) into lines that open to their text. The session's subagents and declared factory waves sit above the composer.
- From an empty chat, the composer's project selector (`None` by default; host access only) starts a Pi session: no folder is a plain chat, a folder turns the first message into a Pi session and the tab into a Pi tab. The session picker (Alt+A; host access only) lists running, recent and older sessions with full-text search.
- Pi features require host access and a running pi-host. Runs continue while no tab shows them and across backend reloads. Old omp sessions are not shown; they stay reachable through the `omp` CLI.

## Agent factory

The agent-factory package (outside this repository) runs each task's coder, the repository's gates and a reviewer, feeds problems back as the next round, and merges when a round passes. GigaChad is a thin client: boards, timelines, diffs, previews, per-project settings, and reverting a merged task's code on delete. Details: [agent factory](agent-factory.md).

1. Open a repository's factory board (sidebar, Pi section). Without a `factory.yaml` it shows an editable draft to save, uncommitted, into the checkout; New task stays disabled until then.
2. Set defaults: branch, merge `auto` or `review`, max rounds, review on/off, preview, and a small and a big model (easy wave tasks run on small; everything else and the reviewer on big).
3. Create a task: title, request, optional images, optional overrides of branch, merge and models. It is tagged with your login.
4. Follow it on the board (Working, Merging, Finished, and a Waiting block for questions, approvals, blocked or interrupted work). Waves are containers with their tasks, reviews, merge and a link to the declaring session. The task view shows the diff, round timeline with gate output and review findings, and the agent's Pi session. Comments on diff lines go out as one review message that starts a new round.
5. Approve, answer, retry a blocked or interrupted task, cancel a running one, or write to it (a message to a resting task starts new rounds on the same branch).
6. Start a preview from the task view when `factory.yaml` defines one; preview data starts empty. Previews run on loopback ports 8100–8199 with the development identity, reachable only through the preview gate.
7. Delete a task: its session, worktree and branch go. A merged task's commit is reverted in the checkout first; a conflicting revert refuses the delete.

- `factory.yaml` and `.factory/agents/coder.md` and `reviewer.md` come from the target's commit; the coder cannot change them.
- Requires host access, a running pi-host and the installed `factory` CLI; without it the board says where to get it.

## Learn mode

- LearnHarness inside gigachad probes what you know, plans concepts toward a goal, teaches with checks and keeps a global knowledge graph. Sessions appear in the sidebar and open as tabs.
- Host-only, single learner. LearnHarness owns its knowledge; gigachad chats never read it. More: [integration-learnharness](integration-learnharness.md).

## Memory

- Memory is the memory-store package: JSON fact lists (each with a description) and Markdown documents in named memory_stores under one root. The host login shares the machine's root (`~/.pi/memory_store`) with Pi; other logins get their own folder in the checkout.
- `user` holds cross-chat facts; each project has its own memory_store (named after its linked folder, else its slug; renaming an unlinked project moves it).
- The chat Memory tool toggle (on by default) lets the model list, read and change memories and puts `profile` and `preferences` in full, plus an index of `user` and the project's store, into the system prompt. The model reads before answering from memory and saves only durable facts. There is no extraction step, review screen or slash command.
- The memory window (sidebar brain button) browses and edits stores, folders, memories and documents: facts are edited, added and retired in place, descriptions are editable, documents are read-only.
- Memories stored before memory-store stay hidden until `scripts/migrate_memory.py` has run.

## Access and failure limits

- Pi sessions, factory, Neovim, the workbench, learn mode, production redeploy and files outside ordinary mounts require host access; other identities stay within mounted roots. Credential-store protection limits host file access outside those roots; do not mount secret directories expecting a blanket denylist. A share link is the one way past identity: its holder reaches its one resource.
- Mounted trees omit ignored/internal directories. Text editing refuses non-UTF-8 data.
- A reference is not a backup: a missing one keeps only its last snapshot. PDF extraction may lag a changed file; visual attachments do not update a sent snapshot.
- Model/provider, OCR, research, sandbox, architecture, Pi and factory services have independent prerequisites. Unavailability, invalid upstream responses, rejected edits and failed saves stay visible, never shown as successful empty results.
- Recovery and exact status/error/security contracts: [architecture](architecture.md) and the runbooks in [AGENTS.md](../AGENTS.md#documentation).
