Generated from bd63559a on 2026-11-25. The working tree also had uncommitted changes in `.docs/agent-factory.md` and the Pi transcript's subagent/intercom message cards (`features/pi/SubagentMessages.tsx`), which these docs describe.

# System overview

## Purpose and concepts

GigaChad Bot is a self-hosted, local-first LLM workspace for conversations, research, documents, visual thinking, and coding agents. Connect cloud or local models and keep histories and artifacts on infrastructure you control.

| Concept | Meaning |
| --- | --- |
| Chat | A model conversation with its own history, branches, attachments, and optional tools |
| Project | A collection of conversations, documents, canvases, architecture, project memory, and a lightweight task board; optionally linked to a mounted code directory |
| Document | An app-owned artifact or a reference to an existing mounted file; references edit the original rather than an independent copy |
| File mount | A backend directory made available for browsing, editing, git review, and worktree selection; it is not a browser-local folder |
| Canvas | An infinite drawing workspace with ink, notes, page/image frames, and document windows |
| Architecture workspace | A project's merged LikeC4 model, editable as diagrams and `.c4` source |
| Pi session | A coding-agent conversation run by Pi through pi-host, distinct from local model chat |
| Factory task / wave | A task is a Pi coding session that the agent-factory package drives through gates, review and merge in its own git worktree, with an optional preview. A wave is a set of tasks built on one branch, reviewed once and merged as a whole |
| Workbench | One `.md` file or one LaTeX project as Neovim beside its rendered view (Markdown preview or built PDF) with a Pi chat drawer. The files live on disk and the agent edits them there |
| Learn session | A LearnHarness teaching session (probe, plan, teach, recall) shown in learn mode, host-only |
| Memory | Durable facts as plain files in a memory-store root, offered to chats as tools and an index in the prompt; not vector search |
| Share link | A revocable secret URL that grants one stored resource (today a canvas, edited together) to whoever holds it |

Use [architecture](architecture.md) for implementation, persistence, access boundaries, integration contracts, and failure semantics. Use the [module map](module-map.md) to locate source owners. Find operational instructions through the [repository documentation index](../AGENTS.md#documentation).

## Runtime and prerequisites

- Run the development workspace on Linux with Bash, Python 3.12 or 3.13, `uv`, and running Docker with Compose accessible to the user. The launcher installs dependencies, including Node.js; the default local MinerU OCR stack makes first installation large.
- Start the workspace with `./run.sh` or the individual launchers. Agents start it as [AGENTS.md](../AGENTS.md#running-the-dev-server) describes.
- Open the development UI at `localhost:2999`; the API runs at `127.0.0.1:8001`. LikeC4, pi-host, and the preview gate are supporting services, not alternate user workspaces.
- Startup reuses PostgreSQL credentials and data. If `POSTGRES_PASSWORD_DEV` is missing, setup asks before generating credentials and creating the database; declining stops startup without changes. Restore an existing database's original password rather than replacing it.
- Keep provider credentials in private `~/.secrets` or the environment. Optional nonsecret configuration belongs in `${XDG_CONFIG_HOME:-$HOME/.config}/gigachad-bot/env`.
- Pi tabs require `pi` at the version `pi-host/package.json` pins on the backend's `PATH`; `run.sh` starts pi-host (set `GIGACHAD_PI_HOST=0` to skip it).
- `run.sh` also starts the factory preview gate. Set `GIGACHAD_FACTORY=0` to skip it.
- Keep production backend services private. Development launchers are not production deployment. Host-access users can trigger a production redeploy from settings. Remote inference, identity, production setup, and service updates belong to the maintained runbooks indexed in [AGENTS.md](../AGENTS.md#documentation).
- Failed startup reads show a shared warning after an error or a 15-second timeout. Retry reloads only failed data without losing drafts or already-loaded content.

## Model chat, context, and projects

- Select models from the providers Pi has a login or API key for, or local Ollama models. Configure model defaults and editable system prompts for the task; supported models expose reasoning controls.
- Distinguish Pi models in local chat from Pi coding-session tabs. Every local-chat model, Ollama included, answers through pi-host's completion gateway, the tabs run Pi sessions through pi-host.
- Stream replies, revisit saved histories, branch from a message, and merge branches back into a conversation.
- Collapsed answers render their rich text when nearby or focused, reducing work in long chats. Hidden tabs wait until shown, and expanding a turn reveals its full answer.
- Replies stream as they arrive, paced evenly. A chat request on a dead connection is sent again (at worst a reply is generated twice), and a transport failure shows in the reply instead of an empty bubble.
- Organize related chats and artifacts in a project. Use its task board for lightweight planning; factory boards run agent tasks and are separate.
- Space outside text fields opens the tab picker: search over open tabs with a live preview of the highlighted one. Delete (at the end of the search) or Alt+W closes the highlighted tab, Alt+T opens a new background tab, both keep the picker open.
- Rename a project to move all its chats, documents, canvases, architecture, memory and attachments to the new slug, with internal links updated. A name whose slug is taken is refused. Archive projects (and, per device, Code directories) to hide them from the sidebar, restore from the section's archive list.
- Keep each chat's project membership independent of the currently selected project elsewhere. Its containing project determines project memory, even when another tab is active.
- Attach uploads, mounted files, global/project documents, architecture sources, or drawings. Toggle attachments out of model context without deleting their source; open them to preview or edit.
- Create project documents in their intended project. Reuse generated tool artifacts as documents and retain source references from web research.
- Remove chat turns or attachments from context with undo available; resource deletion also offers a deferred undo window. Canvas editing has its own history.
- Deleting a project, chat (with its branches), canvas, project file or chat upload also deletes what was derived from it, such as uploads, the library PDF copy and its annotation canvas, once no other project, chat or canvas uses them. Annotations of a mounted PDF opened from the file tree stay while that file keeps its bytes. Extracted Markdown and images are kept, and mounted disk files are never deleted. Deleting a workbench document removes its whole folder from disk (undo only until the deferred delete fires).
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

- Mount an existing backend directory, including an Obsidian vault, to work on its files in place. Browse its filtered tree and open the type-aware text, PDF, canvas, architecture, plot, or mind-map view. Tool artifacts (plots, mind maps, Mermaid diagram documents) open rendered and fill the pane, their text is the secondary view.
- Click an image file (png, jpg, gif, webp, svg) in any mount, project or LaTeX folder to see it on its own. From a LaTeX project tab it opens in a new tab, so the editor and its unsaved text stay.
- Add a mounted file to a project as a reference. Editing changes the original disk file; removing the project reference does not delete that source.
- Upload files into a project's files folder and drag them to another project. Chats, canvases and other projects using the file follow it. Drag an uploaded file onto a Code folder to move it to disk, where it then behaves exactly like a mounted file. Uploaded and moved PDFs get MinerU extraction like mounted ones.
- Rename a project file or a mounted file from the sidebar: click opens it, double-click renames it in place (Enter or leaving saves, Esc cancels). Projects and learn sessions rename the same way. The file itself is renamed and every reference follows (project lists, chat and canvas attachments, notes, the workbench chat link), open tabs on every device move with it, and an open Neovim session keeps its unsaved text. A name in use is refused under the row. A LaTeX project rebuilds after a rename.
- Open a `.ipynb` file from the sidebar to read its outputs and edit and run cells on a kernel of the chosen environment (a `.venv` near the notebook or the app's own). Edits by agents or other devices on disk merge into the open notebook live, and renaming or deleting the file moves or ends its kernel (needs host access).
- With Neovim enabled, the notebook opens split like the Markdown workbench: Neovim holds the whole notebook as one text with a marker line before each cell, the notebook shows outputs and stays editable, and the top bar switches between Neovim only, split and notebook only. `:w` applies the text to the notebook and saves it, cells that stay keep their outputs. Shift+Enter in Neovim saves, runs the cell under the cursor and moves to the next cell (adding one after the last), Ctrl+Enter saves and runs in place. Edits in the notebook, by agents or other devices reach Neovim (merged into unsaved edits). A `:w` that would undo a notebook change Neovim has not shown yet is refused with "The notebook changed, write again", the change arrives in Neovim and the next `:w` saves. The language server sees the text as a file beside the notebook (`<notebook>.ipynb.py` for Python, finding its `.venv`), a file that is never created. `:q` leaves the notebook alone with a "Start Neovim" button, which reopens Neovim on the notebook's current text.
- Upload PDFs or attach mounted PDFs, view their pages, and use MinerU extraction to supply readable Markdown to chat. Image OCR can transcribe text and LaTeX with the configured vision model.
- Annotate PDFs with pen and highlighter. Annotations belong to the PDF's content, so every copy shares them and a renamed file keeps them, while changed bytes start empty.
- Ask about a PDF in a side chat. With host access it is a Pi session per PDF (session menu, New session, kept like workbench chats) that reads the PDF itself. Other logins get an unsaved local chat grounded in the extracted text. Ask about a selected passage either way.
- A PDF with a finished extraction gets a PDF/Text switch in its top bar, Text shows the extracted Markdown rendered with its images. That Markdown has no sidebar row of its own and is deleted together with its PDF.
- Keep PDF library names unique within your account. Uploading or syncing another PDF with the same filename replaces the shared library entry, rather than creating a separate library item. Extraction is also keyed by filename stem.
- For study, attach extracted lecture/document material, choose a tutoring prompt, and ask for explanations, active-recall material, diagrams, or mind maps.
- Enable tools for the current conversation rather than assuming every reply uses them. Use web search for current or source-dependent facts and deep research for broad, multi-part investigations; deep research can take minutes.
- Web search uses Brave source evidence and the selected answering model. Deep research uses GPT-Researcher with a DuckDuckGo retriever and configurable research models, depth, and breadth.
- Generate Mermaid diagrams and mind maps in chat. Use the sandbox plot tool for interactive charts or opt into the workspace agent for file/code work in a persistent chat-scoped sandbox.
- Save the conversation to checkpoint its sandbox workspace. Rich outputs are stored during execution, but the latest workspace remains staged in backend memory until the chat save succeeds.
- Optional external services and credentials are real prerequisites. An unavailable integration does not imply a transparent local replacement. See [architecture](architecture.md) for service boundaries and failures.

## Workbench (Markdown and LaTeX)

- With Neovim enabled and host access, `.md` project files and mounted files, and `.tex`, `.bib`, `.sty` and `.cls` files, open in the workbench: Neovim left, rendered view middle, Pi chat in a drawer. Without either, they open in the regular document editor. Notes always use the regular editor.
- A stored project document moves to disk on first open (`GIGACHAD_BASE_DIR/workbench/<kind>/<project>/<name>/`) and stays a disk file from then on. Mounted files are edited in place. New LaTeX documents are created on disk directly.
- Markdown renders live from the saved text, including Mermaid and mind-map documents. A failed LaTeX build keeps the last good PDF, and the PDF swaps to each build without blanking.
- Visual-mode `<leader>a` quotes the selected lines into the chat draft. Each chat keeps its own list of sessions, shared by all devices: the chevron beside the session title switches between them, `+` starts a new one. The first message of a new session names what the chat is about (`File: <path>` for Markdown, `Project: <folder>, root: <file>` for LaTeX), and after a rename or a move the next message of each earlier session names it again.
- Agent edits appear in Neovim while the agent runs. Unsaved edits on other lines survive. Unsaved edits on lines the incoming change also touches, or on adjacent lines, are replaced by the incoming version without notice. Saving before an agent edits those lines is the user's responsibility. A save never overwrites an agent edit it has not seen: `:w` merges first and reports "Saved, merged with the agent's edits".
- Every text editor follows its file live: edits by an agent, a terminal or another device appear in Neovim (merged into unsaved edits) or reload a clean editor within a moment, and the PDF of a LaTeX project rebuilds for any change in its folder.
- Workbench chats use a short Markdown or LaTeX prompt kept in this repository, without coding rules or `AGENTS.md`. Switch between editor only, split and preview only (remembered per device).

## LaTeX projects

- A LaTeX project is a folder with a root `.tex` file, the one with `\documentclass`. Opening any file of it (`.tex`, `.bib`, `.sty`, `.cls`) while a tab shows that project switches the tab to the file instead of opening another: Neovim keeps every file as a buffer, and `:bnext` switches between them too. One chat belongs to the project folder, not to a file.
- The PDF is the root's. The root of a file is the `.tex` its `% !TEX root = <relative path>` line in the first 5 lines names, else the file itself when it has `\documentclass`, else `main.tex` of its folder, else the only root document there. Without a root the PDF pane lists the root documents in the folder and the 2 above it, and choosing one writes the `% !TEX root` line into the open file. Without any it says "No root document".
- The PDF rebuilds on open, on every `:w` of any open file, on every agent edit as it arrives and when any file of the folder changes, also one you have not opened. Build output (`.latex-build/`, `*.view.pdf`, `.aux`, `.log` and the like beside a `.tex`) is hidden from the sidebar and triggers no rebuild. A build error names its file when it is not the open one.
- Renaming a file of a LaTeX project (sidebar) also rewrites the references to it in the project's `.tex` and `.bib` files (`\input`, `\include`, `\includegraphics`, `\bibliography`, `\addbibresource`, the `% !TEX root` line). Open buffers pick the new text up, and a failed rewrite leaves the rename in place.
- Turn a project's workbench `.tex` into a project folder with the folder button on its row ("Make LaTeX project", one way). The folder then lists in the project like a mounted folder: files open and rename as there. Deleting its row only unlists it.
- A Code folder that holds a `.tex`, and a project folder, shows a create action on its row: type a name (a name without extension gets `.tex`) and Enter. A new `.tex` starts with a `% !TEX root = <relative path>` line when the folder has a root, else with an empty article. New `.bib`, `.sty` and `.cls` files start empty.
- Paste a PNG from the clipboard into Neovim (`p`, `"+p` or Ctrl+Shift+V) to store it as `imgs/img-<timestamp>.png` in the project folder (next to the file for Markdown) and insert `\includegraphics[width=\linewidth]{imgs/img-<timestamp>.png}` (Markdown `![](imgs/img-<timestamp>.png)`) below the cursor. This works in every Neovim editor (workbench, Code file panel, document editor) for a `.tex` or `.md` file on disk. A root without `graphicx` gets `\usepackage{graphicx}`. In any other file Neovim shows an error, other image formats paste nothing, and text pastes stay unchanged.
- Double-click a spot in the PDF to jump Neovim to its source line (inverse search). `<leader>v` in a `.tex` buffer scrolls the PDF to the cursor line and marks it (forward search). Download the built PDF from the top bar.

## Presentations

- Create a LaTeX document from the files folder's `+`: pick LaTeX, then "Blank" or a template set (a Beamer presentation, a presentation draft, a KOMA-Script letter). The project gets its own copy of the set's files, so it builds anywhere (e.g. Overleaf) and may diverge.
- A root that uses a presentation set's theme is a presentation. With Neovim enabled the top bar offers Add slide: a searchable picker of the set's slide primitives with previews, inserted as a new frame after the one around Neovim's cursor. Clicking the gap between two PDF pages inserts after the slide above. Ctrl+Shift+V of an image fills the image placeholder on the cursor line.
- Present opens a presenter popup (current and next slide, `\note{}` notes, timer) and on request an audience popup, synchronized slide by slide.
- Workbench chats for LaTeX follow the shared slide rules of the agent prompt and the set's `STYLE.md`: in a presentation the agent builds slides from the primitives, in a draft it brainstorms storyline and content with plain frames and `\visual{}` placeholders, polished later by switching the theme. Editing a template set follows `templates/latex/AGENTS.md`.

## Canvas

- Start a scratch canvas, then save it as a named global or project `.canvas` document. Scratch content is device/browser-local until saved; named canvases can synchronize edits across open clients.
- Draw and erase with a pen, pan/zoom, select and transform ink, insert Markdown notes, and arrange page frames and images.
- Keep PDFs, documents, architecture diagrams, and nested canvases alongside the artwork as interactive windows. Open a window for focused work, then return to its containing canvas.
- Capture a canvas region for sharing and attach a drawing snapshot to a model conversation.
- Add page frames to export an A4 PDF. Exports and model-facing raster artwork include ink, text, and image frames, not the contents of live document/PDF/architecture/nested-canvas windows.
- Use independent undo/redo for the canvas being edited. Nested canvas history is separate; file-window navigation is not an artwork undo step.
- Canvas nesting is limited to prevent recursive file cycles. A failed named-canvas load leaves stored content untouched; interrupted synchronization exposes retry and retries changes automatically. See [architecture](architecture.md) for collaboration and persistence guarantees.
- Share a stored canvas from its top bar: the link (`/share#<secret>`) opens it full page for anyone holding it, edited live together with the owner. Creating a new link replaces the old one, revoking ends it at once (an open guest stream stops before the next change). The link follows renames and dies with the canvas.

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
- Copy diff notes without clearing drafts.
- When anchors cannot be resolved uniquely, preserve original locations and mark the note outdated. Failed sends retain notes; successful sends clear only submitted notes that have not changed.
- Optionally edit a diff in Neovim with read-only `HEAD` beside the working file. Save with `:w` and return with `:qa`. Deleted, binary, oversized, and symlinked diff files cannot use this path.
- Enable Neovim per device when available: host-access identity, `nvim` on backend `PATH`, and an existing `$XDG_CONFIG_HOME/nvim` configuration directory (default `~/.config/nvim`). Otherwise use the built-in editor.
- Neovim `:w` uses the document's normal save path, including architecture validation; failed saves leave the buffer modified. Yanks are copied to the browser clipboard. Preview/fullscreen switches preserve the session, but closing/quitting/disconnecting terminates it and loses unsaved edits. While focused, its terminal owns all keys, including Escape.
- The browser is Neovim's clipboard, in every Neovim editor: `p` and `"+p` paste what is on the clipboard of the browser, never the clipboard of the machine Neovim runs on (on the deployed server that is not yours). Chrome asks once to allow reading the clipboard, Firefox and Zen show a paste button on every read. When the browser does not answer (permission denied, nothing within 15 seconds), Neovim shows the reason and pastes what you last yanked in Neovim, so `yy` then `p` always works.

## Worktrees

- Paste an https or ssh git link into the Code `+` dialog to clone it into `~/git/<repo>` and mount it. An existing folder is refused, a failed clone leaves nothing behind.
- Switch a mounted git clone between worktrees without renaming its mount. The file tree, diff, and editor follow the selection, which persists per mount.
- Find the remote default through `origin/HEAD`, falling back to the main checkout's branch. Reuse or create that checkout without pulling or switching the main checkout merely to find it.
- Create a worktree from `dev` by default, otherwise the remote default. Creation fetches and fast-forwards the checked-out base before creating `~/git/worktrees/<repo>/<name>`; existing branches or folders are refused.
- Require explicit force to remove a dirty worktree. Branch deletion is separate, unmerged deletion requires confirmation, and remote branches are never deleted by this workflow.

## Pi tabs

- Open Pi coding sessions as workspace tabs, rendered natively in the shell. The sidebar lists sessions per repository and mount, worktrees grouped under their main repository, each with its running, waiting-on-a-dialog and unread state. A session can start in any folder: Git checkouts and `~/git`, `~/dev` folders are listed, and a mounted folder is offered too.
- Branch from any turn: it forks into a new session that opens in the tab. Open returned file links under the shell's file-access policy; an extension command can open a file the same way, e.g. `/files` opens a file the session changed in the file panel.
- Use the shell composer for prompts, images, abort, model/thinking selection, and `/` or `@` completion. While the agent runs, Enter steers and Alt+Enter queues a follow-up; Esc or Stop aborts and takes queued messages back into the draft.
- The transcript folds what is meant for the agent: a loaded skill (`/skill:<name>`) is one collapsible line, a compaction shows its full summary, a background shell session that ended well leaves no line, and factory notices, subagent results and intercom messages are one line that opens to their text. Above the composer sit the session's subagents and the factory waves it declared.
- Start a Pi session from an empty chat with the project selector in its composer (`None` by default) (host access only). No folder is a plain chat, a folder turns the first message into a new Pi session in it, the tab becomes a Pi tab.
- The session picker in the top bar (Alt+A) lists running, recent and older Pi sessions with full-text search. Picking opens it in the current tab (host access only).
- Pi features require host access and a running pi-host. Runs continue while no tab shows them and across backend reloads. Old omp sessions are not shown, they stay reachable through the `omp` CLI. Integration ownership and deployment compatibility belong in [architecture](architecture.md); operational steps are indexed in [AGENTS.md](../AGENTS.md#documentation).

## Agent factory

The factory is the agent-factory package, outside this repository: it runs each task's coder, the repository's gates and a reviewer, feeds problems back as the next round, and merges when a round passes. Waves group tasks a supervising Pi session (or the CLI) declares. GigaChad is a thin client: it shows boards, timelines, diffs and previews, keeps per-project settings, and reverts a merged task's code on delete. How tasks and waves run: [agent factory](agent-factory.md).

1. Open a repository's factory board (sidebar, Pi section). A repository without a `factory.yaml` shows an editable draft to save into its checkout, uncommitted; New task stays disabled until it exists.
2. Set the project's defaults: branch, merge `auto` or `review`, max rounds, review on or off, preview, and a small and a big model (easy wave tasks run on small, everything else and the reviewer on big).
3. Create a task with a title, a request and optional images (pasted or attached), optionally overriding branch, merge and models. A task started here is tagged with your login.
4. Follow it on the board: Working, Merging and Finished, with whatever waits on you (a question, approval, a blocked or interrupted task or wave) in a Waiting block above Working. Waves sit in the columns as containers with their tasks, reviews, merge and a link to the declaring session. The task view shows the diff, the round timeline with gate output and review findings, and the agent's Pi session. Comment on lines of the diff and send the comments as one review message, which starts a new round.
5. Approve a task waiting for approval, answer a question, retry a blocked or interrupted one, cancel a running one, or write to it: a message to a resting task starts new rounds on the same branch.
6. Start a preview from the task view when the repository's `factory.yaml` defines one. Preview data starts empty.
7. Delete a task: its session, worktree and branch go. A merged task's commit is reverted in the repository's checkout first; a revert that would conflict refuses the delete.

- `factory.yaml` and `.factory/agents/coder.md` and `reviewer.md` come from the target's commit; the coder cannot change them.
- Previews run on host loopback ports 8100–8199 with the development identity, reachable only through the preview gate.
- Factory features require host access, a running pi-host and the installed `factory` CLI. Without the package the board names where to get it.

## Learn mode

- Study with LearnHarness inside gigachad: it probes what you know, plans concepts toward a goal, teaches with checks and keeps a global knowledge graph. Sessions appear in the sidebar and open as tabs.
- Host-only, single learner. LearnHarness owns its knowledge, gigachad chats never read it. Everything else about learn mode: [integration-learnharness](integration-learnharness.md).

## Memory system

- Memory is the memory-store package: JSON fact lists ("memories", each with a description) and Markdown documents in named memory_stores under one root. The host login shares the machine's root (`~/.pi/memory_store`) with Pi; every other login gets its own folder in the checkout.
- `user` holds facts about you across chats, each project has its own memory_store (named after its linked folder, else its slug). Renaming an unlinked project moves its memory_store.
- In a chat the Memory tool toggle (on by default) gives the model tools to list, read and change memories, and puts your `profile` and `preferences` in full plus an index of `user` and the project's memory_store into the system prompt. The model reads a memory before answering from it and saves only durable facts. There is no extraction step, review screen or slash command.
- Browse and edit everything in the memory window (sidebar brain button): memory_stores, folders, memories and documents. Facts are edited, added and retired in place, descriptions are editable, documents are read-only.
- Memories stored before memory-store stay hidden until `scripts/migrate_memory.py` has run.

## User-visible access and failure limits

- Require host access for Pi sessions, factory, Neovim, the workbench, learn mode, production redeploy, and files outside ordinary mounts. Other identities remain restricted to mounted roots. Outside those roots, credential-store protection limits host file access. Do not mount secret directories expecting a blanket denylist to protect their contents. A share link is the one way past identity: whoever holds it reaches its one resource.
- Mounted trees omit ignored/internal directories. Text editing refuses non-UTF-8 data rather than risking a lossy binary overwrite.
- A reference is not a backup of the original file; missing references retain only their last available snapshot. PDF extraction may lag a changed file, and visual attachments do not update a previously sent snapshot automatically.
- Model/provider, OCR, research, sandbox, architecture, Pi, and factory services have independent prerequisites. Service unavailability, invalid upstream responses, rejected edits, and failed saves must remain visible rather than being represented as successful empty results.
- Keep operational recovery and exact status/error/security contracts in [architecture](architecture.md) and the maintained runbooks linked from [AGENTS.md](../AGENTS.md#documentation).
