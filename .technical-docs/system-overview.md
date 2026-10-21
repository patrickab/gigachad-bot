# System Overview

## What runs where

GigaChad Bot is a local-first, single-user LLM workspace: a Next.js/React
client (`:2999`) talks to a FastAPI backend (`:8001`) which owns all
persistence and filesystem access. The backend is the sole authority for
paths — the browser only ever sees paths the server already issued.

`./run.sh` starts both. On Linux desktop, the frontend is statically exported
into a Tauri webview and the backend is frozen into a PyInstaller sidecar that
Tauri launches on an ephemeral loopback port; the browser-facing API surface
(FastAPI) is unchanged, Tauri just supplies the base URL after a readiness
handshake.

## External services it integrates with

The backend is a coordinator over several optional local/cloud services —
none of them are reimplemented in-repo:

- **LiteLLM** — the model-provider abstraction. All chat/completions go
  through it, so Gemini, DeepSeek, OpenRouter, and local providers are
  swappable behind one interface (`src/config.py` holds defaults).
- **Brave LLM Context** — the `web_search` chat tool retrieves labelled source
  passages using `BRAVE_API_KEY`, then has the selected model produce a
  citation-grounded answer. No local search service is required.
- **GPT-Researcher** powers the `deep_research` chat tool with its built-in
  DuckDuckGo retriever.
- **MinerU** — PDF parsing/OCR. `routes/mineru.py` + `ExtractQueue`
  (`extract_queue.py`) manage the parse lifecycle; parsed Markdown backs both
  the "Study" mode (mind map/overview/article generation) and the document
  library's PDF preview. The desktop sidecar bundles only MinerU's HTTP
  *client* — the torch/CUDA OCR server is excluded, so desktop OCR requires an
  external `mineru.cli.fast_api` instance via `MINERU_SERVER_URL`
  (`remote-inference.md`).
- **Nextcloud** (or any configured cloud-synced directory): persists PDFs and
  MinerU inputs/outputs only. PostgreSQL owns every other artifact under
  database-native namespaces.
- **Tauri** — the desktop shell only; it has no filesystem authority of its
  own and exists purely to host the exported UI + sidecar.

All of the above are optional/user-configured — the app degrades gracefully
(e.g. web search and OCR just aren't available) when they're not running.

## Theme palette

The role colors shared with the omp frame (ink, paper, surfaces, danger, dividers,
small shadow, inner highlight) live once in
`src/frontend/packages/chat-ui/src/palette.css` as `--gc-<mode>-<role>`.
`src/frontend/app/globals.css` maps them onto the shell's role tokens (`:root`
dark, `:root.light`) and owns the shell-only ones, ompweb's
`custom/theme/palette.css` maps the same values in embed mode. The warm dark
surface stack: paper at 13% OKLCH lightness, surface at 16.5%, elevated panels
at 19%, low-chroma yellow warmth (hue 70), not red-brown. Light mode is independent.

`features/settings/ThemeToggle.tsx` switches palettes with a constant-speed 450 ms
View Transition expanding from the viewport center. A narrow 11-27 px mask edge
softens the reveal without blurring the content. `app/globals.css` owns the
animation instead of globally fading every
element's colors. Reduced motion and browsers without View Transitions switch
instantly. The theme remains persisted locally and reaches the omp frame through
the existing theme bridge.

## Chat sizing

Q&A columns use the collapsed-sidebar width derived from the viewport, not the
remaining pane width. Opening navigation preserves that width until the available
space cannot fit it, then the column shrinks without horizontal overflow. Wide mode
continues to fill the available pane.

Q&A pairs use Soft styling: borderless rounded tonal panels, with answers and tool
outputs on a slightly raised surface. Collapsed previews share the same treatment.
The entire expanded question area collapses its pair, including the avatar and
padding. Links, attachments, action buttons and text selection retain their own
behavior. The chevron remains keyboard-accessible. Surface colors derive from the
shared theme tokens in both dark and light modes.
Collapsed pairs use tighter padding and gaps, omit role headings, and always show a compact answer preview and
each called tool's icon and name, without hovering. Long previews scroll within
12.5% of viewport height. Expanded messages use equal 20px padding on every side
and no role headings, with the first text line centered beside the avatar.
Actions share the final paragraph line with reserved space so text cannot overlap
them, appearing on hover or keyboard focus. Other terminal blocks keep actions
below their content. Collapsed preview panels use equal 14px padding.
`packages/chat-ui` (`Turn.tsx`, `turn.css`) owns the cards, surfaces, actions and
final-line action placement, and the embedded omp chat renders the same package,
so a visual change to either chat lands in both.
Chat uses plain panels rather than nesting the shared elevation context, which
remains available to memory and project views. Message action eligibility is
shared between toolbar rendering and its reserved text space.

## Chat context

The chat has no side panel. Context lives in the composer:

- **Chip row** above the input: one chip per attachment of earlier messages, then
  the draft's own. A chip's icon switches it in or out of context (faded when
  out), its name opens the full preview, the x removes it. Hovering a chip or a
  message attachment shows a preview card centered above it after 300 ms: the
  image, every PDF page at its own aspect ratio, or the rendered text. The card
  stays open while the pointer is on it and scrolls
  (`features/chat/preview/AttachmentHoverCard.tsx`)
- **Full preview** opens in a floating window (`AttachmentPreview`). A header
  switch picks the view: PDF or its rendered MinerU markdown, and for text
  attachments Neovim (whenever the backend runs it, first when the device's Neovim
  setting is on, `:w` stores the
  edit, `:q` closes) or rendered: markdown files as themselves (task-list
  checkboxes edit them), other text as a code block highlighted by extension
- **"+" menu**: mounted file, project document (the picker, Alt+X, attaches
  project or global documents), architecture, drawing. There is no file upload:
  files come from file mounts
- **Sources** of a web search answer sit in a collapsed "N sources" row under it

New project documents come from the project row's hover "New document" icon in the
sidebar, which opens the created file. Documents are added through file mounts.

**Documents are references.** A mounted file or project document in a chat is
stored by its path (`Attachment.vaultPath`), never copied:

- **Re-read on send, chat open and preview open.** Every send re-reads every
  active reference in the chat, earlier messages and the draft included, and
  rebuilds the text the model reads (`readReferences` + `applyReferenceReads`
  in `features/chat/attachments.ts`), so a file edited in a terminal or by an
  agent reaches the next answer. Each source is read once per pass, and the
  result is patched into whatever the chat holds once the reads finish.
  Opening a chat does the same in the background. Opening a preview re-reads
  that one file: a text preview waits for it, so Neovim never edits a stale
  copy, PDFs and images open at once. Inactive references are skipped until
  switched on again
- **Moved files are followed**: a mount file whose path is gone is looked up by
  name in the mount that held it (`FileRoots.relocate`). Exactly one match
  relinks the reference. None or several (a 404) leave it `missing`: red file
  icon on its chip, last known text kept for the model and the preview as a
  read-only snapshot, a PDF shown from its PDF-library copy. Sends skip missing
  references, so the mount is searched once. Opening the preview retries. Other
  read failures keep the reference as it was. Project listings rewrite moved
  references the same way
- **Edits write through**: Neovim `:w` and checkbox ticks on a reference save
  its source (`saveAttachmentEdit` → `writeReference`: mount file, project
  document, note or architecture source) before the chat updates. PDFs and
  images are never written
- **PDFs** stay referenced by their mount path. Their bytes are synced into the
  PDF library, and a changed PDF drops its cached MinerU extraction so it is
  re-extracted (`sync_library_pdf`). Until that finishes the previous
  extraction is used, so the first send after a change still answers from the
  old text. The extraction queue runs the same bytes once however often they
  are attached, and drops an extraction whose PDF changed meanwhile. Older
  chats that stored `PDFs/<name>` still resolve
- **Not yet references**: pasted images, drawings, a canvas attached as an
  image and the architecture attachment are one-time snapshots

## Canvas background

The dot grid uses small, screen-sized dots and wider spacing at reduced zoom.
Intermediate dots fade with an eased curve between density levels, while the
grid stays anchored to canvas coordinates during pan and zoom. Base dot opacity stays
between 85% and 95% for visibility. Light mode uses the darker `--ink-muted` color,
while dark mode keeps `--ink-faint`.

## Architecture workspaces

Each project holds one LikeC4 model, either under `docs/architecture/` of its linked
file mount on disk or, without one, under `graph/<slug>/` in PostgreSQL:
every `.c4` file from that root downward (`backend/backend.c4`,
`views/checkout.c4`, ...), merged by LikeC4 into one model. The `.c4` text is
the source of truth, written the LikeC4 way:

- **Systems and modules.** A system is a folder whose own file declares one
  system and the view of it (`backend/backend.c4`: `backend = system 'Backend'`,
  `view backend of backend { include * }`). The first system also declares the
  element kinds. A module is a file beside it adding one container
  (`backend/api.c4`: `extend backend { api = container 'API' }`), so it shows in
  the system's view. The service classifies every file by what it declares
  (`model.tree`: `system`, `module`, `view`, or `file`), never by its name, so
  hand-written files are listed as they really are.
- **Connections.** Each is stored once, in the file declaring its source
  element. Every view showing both ends draws it, so a file's own view also
  shows connections coming in from other files.
- **Saved views.** Each lives in its own file, `views/<id>.c4`, which holds
  exactly one `view <id> { }` built from `include`/`exclude` lines.
- **Positions.** These are LikeC4 manual layouts: `.likec4/<viewId>.likec4.snap`,
  the whole laid-out view as JSON5, in the same format LikeC4's own editor
  writes. The connection style (angled or curved) is not part of LikeC4. It is
  a per-window canvas preference.

`lib/architecture_workspace.py` reads a workspace from any location:
`ArchitectureStore(files, root)` takes a database prefix (`PostgresDataStore`)
or a directory on disk (`DirectoryFiles`). Inside a git work tree a directory
lists what `git ls-files --cached --others --exclude-standard` lists, so
gitignored files are skipped; outside one it lists every file. Symlinks are
never followed. A project is linked to the file root whose folder name equals
its slug: when `<root>/docs/architecture/` exists, that directory is its
architecture (routes resolve it through `get_architecture_store`, and
`/api/fileviewer/text` reads `graph/<slug>/…` paths from it); otherwise the
database copy under `graph/<slug>/` is used.

- **`src/c4/`** — a stateless loopback Node service (LikeC4 + Langium).
  `POST /model` parses sources, applies the saved layouts, and returns views,
  elements, kinds, the file tree, and errors.
  `POST /apply` turns operations into surgical text edits at syntax-tree
  offsets, re-parses, and rejects an edit that introduces errors:
  - Model operations: `addElement`, `addRelation`, `delete`, `setTitle`,
    `setDescription`, `setKind`, `setLabel`, `rename`, `reparent`.
  - File and view operations: `createSystem`, `createModule`, `createView`,
    `addFileView`, `includeInView`, `removeFromView`.
  - `layout`, which moves nodes and bends connections.
  - `pruneDangling`, run on every hand-edited text save and every file
    deletion: drops connections, `include`/`exclude` entries, `view x of gone`
    views and `extend gone` blocks naming an element that no longer exists.

  `delete` cascades the same way: it removes the elements together with every
  connection, view entry, view and `extend` block naming them. A delete or
  save is refused only for other broken references, e.g. element kinds that
  a deleted `specification` declared. Edits to an element go
  into the file declaring it. An element drawn in `view x of x` becomes a child
  of x. An element drawn in any other view goes into the request's `home` file
  or the chosen parent, and is included by name when the view's rules would not
  show it. "Remove from view" drops the element's own `include` entry, or adds
  an `exclude` when a wider rule still draws it. A view is editable only when
  it is built from `*` and element references (`x`, `x.*`, `x.**`, `x._`) in
  `include`/`exclude` rules. Pinned views are re-saved after every edit. In
  `pinView` (`layout.ts`), explicit pins win, unpinned children move with
  their parent, elements added since the view was arranged move as one block
  to the right of it, and compounds grow (never shrink) to contain their
  children. A snapshot that cannot be read is ignored, so its view falls back
  to auto layout. Renames carry their saved positions along.
- **`routes/architecture.py`** (`/api/architecture/{slug}`) owns storage and
  proxies to the service (`lib/c4_service.py`). Every write goes through one
  `_write` helper. It writes changed files in one transaction
  (`DataStore.write_many`), each checked against the revision it was read
  at. On a conflict it rereads and re-validates the operations before
  retrying. Deleted sources and layouts of views that no longer exist are
  removed afterwards, unchecked. New files and deletions are not
  revision-checked, which is accepted for a single user's workspace. Systems,
  modules, and saved views are created through `POST .../ops`
  (`createSystem`, `createModule`, `createView`), raw files through
  `POST .../source`. Files are deleted together
  (`DELETE .../source?path=a&path=b`) only when the merged model still
  parses, after pruning what named them. A refusal names the remaining file
  and line that still refer to them, and a hand edit that leaves
  something unresolved is refused the
  same way. Responses are `{model, created}`. Reading a project without an
  architecture returns an empty one. The project's document list shows
  sources by their path inside the workspace. Layout files are not listed.
- **Frontend.** Each project row in the sidebar has a hover layers icon: it opens
  the project's architecture window in place of a canvas, or creates an empty
  database architecture when the project has none. The canvas "+" menu has one "Architecture" entry on any canvas
  of a project, including the unsaved scratch canvas while a project is open.
  Without a project the menu says to open a project canvas instead. Its window
  (`features/editor/architecture/ArchitectureWindow.tsx`) holds a collapsible tree (systems
  with their modules, views, other files) built from the shared `FileTree`, with
  hover "+" for a system, a module (per system) and a view, and Delete on any
  row (a system goes with its modules). Full-screen in canvas mode, the window
  follows the app's chrome: the tree is a second sidebar card, and the tree
  toggle, drawing tools, Diagram/Text switch and connection menu portal into
  the shared `TopBar` pills like the canvas toolbar. Inside a canvas attachment
  it keeps a flat tree and an inline header strip instead. The window's
  path is the file it is on; tree clicks change it without an undo step. A
  system or view file draws its view, a module draws its system's view with
  the module selected (`ArchitectureDiagramSurface`, React Flow + rough.js),
  and the Diagram/Text switch shows the file in the `.c4` text editor. A file
  without a view offers "Add a view of this file".
  - **Cards are paper until activated.** A click activates (selects) a card;
    only then does it move, and do its title, bullets, and kind respond. On a
    card that is not activated, a drag pans and the pen draws, like on the
    bare canvas.
  - **Nesting is drawn.** A shape drawn inside an element (any element, a leaf
    then becomes a parent) creates a child of it. On the bare canvas of a
    system view it joins the system; a saved view creates nothing there,
    since it belongs to no element (it has no "+" either).
  - **Saved views.** The toolbar adds "Add to view" (a whole file's systems,
    or single elements grouped by file).
  - **Fitted before drawing.** `fitGraph` (in `features/editor/architecture/diagramGraph.ts`) sizes
    the view once per change, before anything renders: each card is at least
    as wide as its heading plus kind label (measured in the real font,
    `nodeMinWidth`), overlapping siblings are pushed right, and parents grow
    around their children, innermost first. The fit is display only; a node's
    fitted geometry is saved once it is moved or resized.
  - **Removing nodes.** A node's corner X (on hover or selection) and the
    Delete key do the same thing: in a saved view they only remove the node
    from the view, then a dialog "Remove reference from <file>?" offers to delete
    it from its `.c4` file (Enter deletes it from the model with everything
    naming it, Esc or a click outside keeps it). In a
    system or module view they delete the element from the model.
    Connections are always model edits.
  - **Auto layout** (toolbar, once positions are saved) clears them and lets
    LikeC4 lay the view out again. Elements added to an arranged view land as
    one block to the right of it.
  - **Connections are routed** with libavoid (`lib/domain/edgeRouting.ts`, WebAssembly
    from `public/libavoid.wasm`, copied from `libavoid-js` on build). Routes are
    right-angled and keep clear of every card a connection does not join. Frames
    are not obstacles, so all connections share one router and parallel stretches
    are nudged apart; each end takes its own pin, on whichever side libavoid finds
    cheapest. Routing reruns on every node move (about 15 ms for a 15-card
    view). A connection's bend is a via point off the midpoint between its
    cards' centres, the same offset the snapshot stores; dragging the handle
    reroutes live. Curved style rounds the routed corners. Until the WASM has
    loaded, or if it cannot, connections fall back to the unrouted Z/L and
    arc shapes.
  - **Views using filters or styles** are read-only on the canvas.
  - **Compounds** show only their title. Their bullets (the element's
    description) stay hidden on the canvas and are edited in Text.
  - **Errors.** One area under the window header shows a refused edit or a
    failed read, and every parse diagnostic as `file:line: message`, in both
    Diagram and Text mode.

  `features/editor/architecture/diagramGraph.ts` diffs each surface edit into operations. The window
  sends writes and rereads through one queue, so an older read never replaces
  a newer write's answer. A change from another device while a write is in
  flight triggers one reread after it. The project row's "New document >
  Architecture" in the sidebar creates an empty source to write by hand and
  opens the window. Saves go through `PUT /api/architecture/{slug}/source`.
  When the project has architecture sources, the composer's "+" menu offers
  "Architecture". It uploads one text snapshot with a fenced
  LikeC4 block for each `graph/<slug>/**/*.c4` source, labeled by its path
  inside the workspace. The existing attachment pipeline includes that
  snapshot in the next chat request; edits made afterwards need a new
  attachment. There is no separate stored chat architecture context. The
  document picker still attaches individual sources. `DirectoryFiles`
  remains available for a future git-backed workspace, but the current
  project routes store sources in PostgreSQL only.
  Deletes of history files and folders, projects, file roots, sidebar
  canvases, project documents and architecture sources, prompts, dashboard
  cards, and canvas-menu documents show a six-second Undo notice
  (`UndoDeleteContext`). The row is hidden at once and the deletion request
  runs when the window closes. Only one deletion is undoable at a time: a
  second delete commits the first immediately, and a row stays hidden until
  its request settles. A refused request restores the row and shows the
  error. Each surface hides only its own rows. Chat message pairs and sent
  attachments are removed from the chat at once instead, so the model no
  longer sees them; an attachment's upload is deleted when the window closes,
  and a pair has nothing to delete. Undo puts them back (`schedule`'s
  `onUndo`, also run when the request is refused), unless another chat has
  been loaded into the tab meanwhile. A committed project delete closes the
  project only if it is still the open one. Canvas node editing keeps
  its own editor undo history. Every icon-only button has an `aria-label`;
  the UI uses no hover tooltips.
  Highlighting uses LikeC4's own TextMate grammar, vendored at
  `src/frontend/components/content/grammars/likec4.tmLanguage.json`.

## Provider usage

`GET /api/provider-usage[?refresh=true]` (`routes/models.py`, `lib/provider_usage.py`) runs
`omp usage --json` and returns one row per OMP login: its worst limit per
5h/7d/30d window with the minutes until reset, or `no_limits`. Output is cached
for 5 minutes. `refresh=true` runs `omp usage invalidate` first. A missing or
failing `omp` answers 502. The sidebar's `ProviderUsage` (above Settings) shows a
"Usage" line with the lowest available 5-hour usage across subscriptions. Provider
rows show only their 5-hour percentage and a muted bar. Missing 5-hour limits show
`—`, never a weekly fallback. Clicking a provider reveals all windows (including
weekly usage) and reset times. Account emails stay hidden. Accounts without limits
are hidden. Only percentage labels at 100% or above turn red, never the bars.


## File mounts

File roots are mounted backend directories, listed in the sidebar's
"File mounts" section below "General" and above "OMP". "Add file mount" opens a directory
picker (`features/navigation/FileMountPicker.tsx`) that starts at the backend
user's home: clicking a directory opens it, the up button opens its parent, the
list follows a typed path, and Enter mounts the path in the input. It lists
directories through `GET /api/fileroots/browse?path=`, which browses the whole
host (directory names only), since any directory can be mounted. Mounting goes
through `POST /api/fileroots/roots` as before.

Reads and writes by path stay inside the mounts, except for logins with host
access (`RequestIdentity.has_host_access`: the local dev login, or a tailnet
login granted `gigachad.local/cap/host`, see "Host access" in the deployment runbook).
They reach every path the backend user can, except credential stores
(`lib/file_roots.py` `is_secret_path`: `.ssh`, `.gnupg`, `.secrets`, `.env*`,
gigachad-bot's `env`, omp's `agent.db` and `models.yml`). Git, worktree and
tree routes stay mount-only.

The tree lists every file a root holds by its full name, skipping only
`.gitignore` matches and `.git`, `node_modules`, `.obsidian` and `.trash`.
Opening a file picks its viewer by type: PDFs open in the PDF viewer, `.canvas`,
`.c4`, `.plot.json` and markmap documents keep their renderers, and everything
else opens as editable text (Neovim when enabled). Files that are not UTF-8 text
are refused (`GET /api/fileviewer/text` returns 415) rather than decoded lossily,
so a save can never overwrite a binary file.

Dragging files from a file mount onto a project's "files" adds them by reference: the
project stores the file's absolute path (`POST /api/fileroots/add-to-project`),
so it always shows the current file, and saving it in the editor writes the
mounted file. Deleting it from the project only removes the reference, the
file stays on disk. PDFs are referenced the same way, and also synced into the
shared PDF library for extraction.

## Diff viewer

Each file mount's **Open diff** action opens a continuous HEAD-to-working-tree
diff with green additions and red removals. The file tree jumps to each file.
`DiffViewer.tsx` uses `/api/fileroots/diff/files`, `/diff` and `/diff/content`,
served by `routes/file_roots.py` and `lib/git_diff.py` within mounted roots.

With Neovim enabled, **Edit diff** opens read-only HEAD beside editable working
content. `:w` saves the file, `:qa` returns to the refreshed diff. Deleted,
binary, oversized and symlinked files cannot be edited this way.

## Worktrees

A file mount holding a `.git` folder is a clone, the integration checkout its
worktrees branch off. Its "worktrees" folder in the sidebar lists every worktree
git knows, wherever it was created (`lib/git_worktree.py`, `/api/fileroots/worktrees`).
Each row shows uncommitted files and commits ahead of and behind the base branch.

- **Switch**: clicking a worktree makes the mount show it. The choice is kept
  per mount in `localStorage` (`active_worktrees`), `GET /api/fileroots/tree?worktree=`
  returns that worktree's files with `mount` holding the clone. The file tree,
  diff and Neovim follow it. `FileRoots` admits the worktrees of clone mounts.
- **Create**: the base defaults to `dev`, else origin's default branch. The base is
  fetched and fast-forwarded where it is checked out, then `git worktree add -b`
  creates `~/git/worktrees/<repo>/<name>`, the layout of the `worktree` agent skill.
  An existing branch or folder is refused.
- **Remove**: uncommitted changes need a confirmed force remove. Then "Delete
  branch?" is opt-in, an unmerged branch needs a second confirmation. The remote
  branch is never touched.

The project linked to a clone keeps reading the clone's `docs/architecture`.

## omp tabs (omp-web)

omp tabs show [omp](https://github.com/can1357/oh-my-pi) coding sessions through
omp-web, a separate Next.js server from the ompweb fork (branch `custom`).
`run-omp-web.sh` serves it on `127.0.0.1:30177` from a detached worktree at
`~/.local/share/gigachad-bot/ompweb` that follows the latest commit of `custom`
on every start, reusing an instance already on the port.
omp-web refuses frames unless `OMP_WEB_FRAME_ANCESTORS` lists the frontend origin,
the launcher sets it from `GIGACHAD_OMP_WEB_FRAME_ANCESTORS`
(default the `:2999` origins). Remote access needs that origin listed too.

- **Data**: the backend reads omp-web server to server (`lib/omp_web.py`,
  `/api/omp/*`), the browser never calls omp-web's API. Pydantic contract models
  fail with the field name when omp-web's JSON changes (502). omp-web down: 503.
  `GIGACHAD_OMP_WEB_URL` points it elsewhere, `GIGACHAD_OMP_WEB_AUTH` is sent as
  `x-omp-web-auth` for a password-protected omp-web
- **Frame access**: `/api/config` hands host-access logins `GIGACHAD_OMP_WEB_FRAME_URL`
  (default `GIGACHAD_OMP_WEB_URL`), others get none and `/api/omp/*` refuses them.
  In production that URL is Tailscale Serve `:8443` in front of `backend/omp_web_gate.py`,
  which forwards only logins granted the host capability, see the deployment runbook
- **Frame**: `features/omp/OmpProvider.tsx` renders ONE iframe for all omp tabs,
  fixed over the active omp tab's slot and hidden otherwise. Tabs switch it by
  postMessage (`packages/chat-ui/src/protocol.ts`, the definition the fork
  installs as `@gigachad/chat-ui/protocol`), never by reload. The frame loads with the first
  session as `?session=` deep link. In a frame omp-web hides its own sidebar,
  topbar, file panel and input box
- **Tab item**: `WorkspaceItem` kind `omp` holds an `OmpView` (session, new session
  in a cwd, settings). A session the frame opens itself (fork, first message of a
  new session) becomes the tab's view
- **Composer**: omp tabs use gigachad's `ChatInput` (tools menu off), `ModelDropdown`
  (only the OMP provider rows, `litellm_proxy/<provider>/<model>`) and
  `ReasoningSelector` (omp-web's thinking levels), `features/omp/OmpComposer.tsx`.
  They drive omp-web's hidden composer over the bridge (`composer-send`,
  `composer-abort`, `composer-set-model`, `composer-set-thinking`, state back as
  `composer`), so sending, queueing, `!cmd` shell commands and new sessions stay
  omp-web's. Images go as base64, documents as text the way chats send them.
  Uploads land under the tab's `chatId`
  - Autocomplete: `composer-complete` asks the frame, `composer-completions`
    answers. `/word` lists omp's extension, prompt and skill commands, `@` lists
    files of the session cwd (directories drill down). The menu is
    `features/chat/composer/CompletionMenu.tsx`, ChatInput's `complete` prop
  - Restore: text and images omp hands back (Stop withdrew queued messages, the
    queue panel's Edit) arrive as `composer-restore` and merge into the draft
  - Not available: word prediction, steering or queueing from gigachad's composer
    while a run is busy
- **Conversation presentation**: embedded omp turns use the same soft Q&A layout as local chats, with equal padding, icon-only copy/fork actions and always-visible collapsed answer previews capped at 12.5vh. Tool names remain visible in collapsed cards. Header toggles and preview scrolling are independent. Older turns collapse initially, the newest stays expanded, and per-session choices survive history paging. Plain text-only turns are kept together before history windowing. Earlier available history, including pre-compaction messages, loads without an extra click. Standalone omp-web keeps its original flat presentation.
  - Copy uses original markdown, not thinking/tool output. Persisted per-message deletion is deferred: OMP has no supported entry-deletion mutation, so there is no delete button.
  - One implementation, `src/frontend/packages/chat-ui/` (`@gigachad/chat-ui`): Q&A turn components and `turn.css`, the Markdown/LaTeX renderer, `markdown.css` and source-preserving copy. Local chat and the fork compose it with host content. The fork consumes a reproducibly packed, versioned tarball and builds without a sibling checkout. Fork internals (typed seam, token mapping, history handling) are in the fork's `custom/AGENTS.md`, "Q&A cards".
  - Package updates: bump its version, run `npm run sync:ompweb` from `src/frontend` (packs into the fork, installs it there), and commit the fork's tarball, manifest and lock together. `npm run check:ompweb` verifies the tarball, the fork's version and its lock integrity.
- **Sidebar**: a project whose slug names a file mount's folder gets a "code"
  category with the sessions whose `project_root ?? cwd` lies inside that mount.
  The "OMP" section lists every other omp-web workspace, its label opens omp
  settings. Forks nest under their parent, the session in view lists its
  in-session branches. Rows archive, the "archived" folder restores.
  The overview refreshes every 15s and on window focus
- **Keys**: Alt+T/W/D pressed inside the frame come back to the shell
- **File links**: a local file link in omp's chat (markdown, Q&A cards, tool
  output) arrives as `open-file` and opens in the shell's floating file view
  (`features/editor/FileHoverView.tsx`, `useOpenFileHoverView` from anywhere under
  `FileHoverViewProvider`): Neovim when the Neovim editor setting is on (`:w`
  saves in place, `:q` closes), else a read-only render, markdown as itself and
  other text as a highlighted code block. Without host access only files inside
  a file root load, others show why
- **Limits**: DOCX/PDF previews stay blank inside the frame (`/api/files` keeps
  `SAMEORIGIN`)

## Agent factory

The factory turns a written request into a reviewed change: an engine
(`agent-sandbox factory serve`, from the Agents-in-a-box repo, `127.0.0.1:8090`)
runs agents in a git worktree per task and gigachad shows the board, the diff
and a running preview. The engine knows nothing about gigachad and a repo needs
no files to use it: the engine ships built-in settings, agents
(`scout`, `planner`, `coder`, `reviewer`) and workflows (`code`, `scout-code`,
`scout-plan-code`, `scout-plan-code-review`). Settings are layered, later wins:
built-in defaults, the user's defaults (edited in the board's settings dialog
and kept by the engine), the repo's optional `factory.yaml`, the workflow file,
the request's overrides. An agent is the built-in one, the user's default or the
repo's `.omp/agents/<name>.md`. This repo only describes what is specific to it:

- `factory.yaml`: the sandbox `profile` (`gigachad`), hooks, gates and the
  `preview:` processes. The preview backend is checked on `/healthz` and runs
  as the development login with an empty database, the frontend is pointed at it
  through `NEXT_PUBLIC_API_BASE`, and the scenario gets the backend's URL from
  `preview.api_url`. Gates and the preview backend run on the main checkout's
  `.venv`, which hooks, gates and preview fields name through the engine
  placeholder `{repo_path}` (the checkout's path as registered). Worktrees
  install frontend packages only
- `.omp/agents/coder.md`: overrides the generic coder with gigachad's conventions
  and the preview scenario example. Agents answer with a JSON envelope in
  `.factory/out/<phase>.json`, which is git-ignored (`.factory/.gitignore`)
- `factory.yaml`, `.factory/**` and `.omp/agents/**` are protected: agent edits
  to them are reverted, so a task cannot rewrite its own rules

`.omp/agents/` is the one tracked part of the otherwise ignored `.omp/`.

`run-factory.sh` (started by `run.sh`, opt out with `GIGACHAD_FACTORY=0`) serves
the engine from the Agents-in-a-box checkout (`GIGACHAD_FACTORY_REPO_DIR`, default
`~/git/Agents-in-a-box`) with its state in the dev PostgreSQL's `factory`
database, reusing an engine already on `127.0.0.1:8090`. On first startup it
clones a missing checkout, installs the locked factory extra, creates the default
engine config, and creates the `factory` database if absent. Existing checkouts,
config and data are preserved. It waits up to 60 s for the PostgreSQL that
`run-backend.sh` starts. `run.sh` first loads `~/.secrets` and runs the guided
`scripts/setup-postgres.sh` when `POSTGRES_PASSWORD_DEV` is missing, stopping if
setup is declined.

**Adapter and routes.** `lib/factory.py` is the only code that knows the engine's
URLs, like `lib/omp_web.py` for omp-web: `GIGACHAD_FACTORY_URL`
(default `http://127.0.0.1:8090`), Pydantic contract models validate
every response. `routes/factory.py` (`/api/factory`) forwards one to one in the
engine's snake_case shapes and requires the host capability, because tasks edit
and run code on the machine: `GET /repos` (name and main checkout of each
registered repo), `POST /repos` (`{path}`, registers a git checkout, repeating a
path returns the same repo), `GET /defaults` and `PUT /defaults` (the user's
settings plus each agent's markdown, the engine refuses an invalid agent with
422), `GET /workflows?repo=`, `GET /tasks?repo=`, `POST /tasks`
(the body names its repo), `GET /tasks/{id}`, `GET /events?repo=&after=`,
`GET /events/latest` (the newest `seq`, where a poller starts; engine-wide on
purpose, `GET /events` is filtered by repo so a cursor past other repos' events
is harmless), `POST /tasks/{id}/{start|approve|cancel|retry}`,
`POST /tasks/{id}/messages`, `POST /tasks/{id}/preview/restart`. Engine down is
503, an unknown response shape 502, a refusal (404 unknown task, 409 wrong state,
422 invalid) keeps the engine's status and message.

Both adapters (`lib/omp_web.py`, `lib/factory.py`) send their requests through
`lib/upstream.py` and share its three exceptions, and both routers use the
host-access dependency and the error mapping of `backend/routes/upstream.py`.

**Changes view.** A task's worktree stays uncommitted until it merges, so the diff
viewer shows it as working tree against `HEAD`. `FileRoots.root_of` (what
`/api/fileroots/diff*` checks) admits any directory inside a git linked worktree
(its `.git` is a file starting with `gitdir:`) for host-access logins, whether or
not the repo is a mounted file root, so it does not depend on where the engine
keeps worktrees. Only `root_of` does: the file viewer still refuses secrets
inside a worktree.

**Frontend.** `GET /api/config` reports `host_access`. For those logins every omp
workspace row (an OMP-section workspace or a project's code category) carries a
factory icon. It opens that repo's board (a `factory` workspace item, path `<repo>`
or `<repo>/<task id>`, `features/factory/`), after registering the directory with
`POST /factory/repos` when `GET /factory/repos` does not list it yet (a failure,
such as "not a git checkout", shows in the OMP section's error line). The board
header's gear opens "Factory defaults": merge mode, maximum rounds, sandbox,
preview, a model per agent and each agent's markdown, saved through
`PUT /factory/defaults` (the engine's 422 detail shows inline). The board has the
columns Backlog, Todo, Working (`in_progress` and
`rework`), Needs you (`human_review` and `blocked`), Merging and Done, cancelled
tasks are hidden, and "New task" picks a workflow and the request. It polls
`/events` every 3 seconds while its tab is in front, starting at `/events/latest`
and moving the cursor on only after the refresh succeeded. A task opens in a
view with Preview (an iframe on `preview.open_url`), Changes (`DiffViewer` on the
task's worktree) and an agent side panel with phases, gate output, findings,
messages, a composer and Approve, Cancel and Retry.

**Preview scenario.** A preview database starts empty. The coder may write
`.factory/out/scenario.py` (never committed), which the engine runs against the
ready preview. Its last stdout line, `{"open": "<path and query>"}`, is where the
preview opens. Gigachad's deep link is `?open=<base64url JSON>`, parsed once on
startup: `{"kind": "chat", "history_file": ...}` or
`{"kind": "canvas" | "document" | "architecture", "path": ..., "project_slug": ...}`,
then removed from the address bar.

Previews are the one exception to "never start the dev server": the engine starts
them on ports 8100-8199 (`AGENTS.md`).

## Neovim editor

Text documents can be edited in a real Neovim with the backend user's own
config, rendered in the browser by xterm.js. The "Neovim editor" switch in
Settings is a per-device preference (`localStorage`) and appears only when
`GET /api/config` reports `neovim_available`: the login has host access,
`nvim` is on the backend's `PATH`, and
`$XDG_CONFIG_HOME/nvim` (default `~/.config/nvim`) exists. Otherwise the
built-in editor stays in use.

- **One process per open document.** `WS /api/neovim/session`
  (`routes/neovim.py`) spawns `nvim -n` in a PTY (`lib/neovim_session.py`) on a
  private temp copy of the text the browser loaded. It starts in the innermost
  mounted file root holding the document (so pickers and `:e` see that tree),
  in the home directory for database documents. Binary frames carry terminal
  bytes, text frames carry JSON control messages.
- **Saving is explicit.** A buffer-local `BufWriteCmd` sends the buffer over
  msgpack-RPC to the backend, which forwards it to the browser. The browser
  persists it through the document's normal write path (project document, file
  root, or validated `.c4` source) and returns the stored text, which replaces
  the buffer if the server rewrote it. A failed save shows in Neovim and keeps
  the buffer modified. The temp copy itself is never written.
- **Closing discards.** Closing the tab or document, or losing the connection,
  terminates `nvim` and deletes the temp copy, with no recovery of unsaved
  edits. `:q` closes the document. Switching to preview or fullscreen keeps the
  same session.
- **Rendering.** xterm.js uses its WebGL renderer, which draws box-drawing
  glyphs itself so borders join, and falls back to the DOM renderer without
  WebGL2. Fonts come from the browser's machine: a local `JetBrainsMono Nerd
  Font Mono` or `Symbols Nerd Font Mono` shows config icons, otherwise the app's
  mono font is used and icons are missing.
- **Keys.** While the terminal has focus every key goes to Neovim, Escape
  included: the terminal stops propagation, and capture-phase app handlers
  (shortcuts, `useEscapeLayer` windows, the command menu) skip events from it
  (`isNeovimKeyEvent` in `lib/utils.ts`). `:q` or the close button leaves a
  window holding Neovim. Outside the terminal the app keys work as before.
- **Security.** The terminal is code execution as the backend user. Besides the
  allow-list, the WebSocket requires an `Origin` from `GIGACHAD_CORS_ORIGINS`,
  or a loopback origin while that list is the `*` default, since CORS does not
  apply to WebSockets.

## Memory system

Memory is a **document-backed, LLM-mediated** store (`src/lib/memory_store.py`,
`routes/memory.py`) — no vector DB. It has two independent scopes:

- **global** — durable facts about the user, valid across all projects
  (identity, communication/learning/engineering preferences, goals).
- **project** — facts about the chat's containing project (purpose, current focus, key
  concepts, decisions, constraints, resources, open questions).

Each scope has its own configurable set of **categories**
(`global-categories.json` / a project's `memory/categories.json`, falling back
to built-in defaults); every stored memory belongs to exactly one category.

**Pipeline**, all user-gated:

1. **Extract** — a small model (`MEMORY_MODEL`) reads the conversation tail
   since that chat's last memorization and emits only *new or corrected*
   atomic facts as JSON, given the categories and what's already on record (it
   never re-proposes known facts). Candidates are buffered to
   `memory/pending/<review_id>.json`. Global and project extraction run
   concurrently and independently.
2. **Review** — the frontend shows the candidates; the user accepts, rejects,
   or edits them. Nothing is written to the canonical store yet.
3. **Reconcile** — accepted candidates are merged into the canonical list
   *per category only* (a category with no new candidates is untouched); an
   LLM call merges overlapping/contradictory text within that one category
   and returns the deduplicated canonical list, tagging each result
   `pre-existing` / `new` / `combined` for the diff view (`preview`/`commit`
   share this path).
4. **Commit** — the canonical JSON (`global-profile.json` or a project's
   `memory/memory.json`) and a rendered Markdown doc
   (`global-profile.md` / `memory/memory.md`) are both written, and a
   per-`(chat_id, scope)` **watermark** advances so the same messages aren't
   re-extracted later. Cancelling discards the pending buffer without moving
   the watermark.

At chat time, `MemoryStore.augment_system_prompt` reads both Markdown docs
(global profile + the chat's containing project's memory, if any) and appends them to
the system prompt under a `# Persistent Memory Context` header — this is how
the LLM actually "remembers" across chats. Users can also list, move a memory
between scopes, or re-bucket memories orphaned by a category rename/deletion
(`remap_orphaned`), all without touching the raw JSON by hand.

Project membership belongs to each workspace tab, not a global active-project setting.
Switching tabs or selecting another project does not change another chat's memory scope.
The sidebar highlights the selected item and its containing project. Selection backgrounds
fit the item label and badge, excluding indentation and unused row space.

Workspace navigation reveals a centered breadcrumb with a borderless new-chat `+`
immediately to its right and a separate, centered next-tab button beneath it.
All controls share the hover and keyboard-focus reveal. The `+` creates a tab in
the current project scope without opening the menu. The tab menu only lists,
searches, and closes open workspaces.
