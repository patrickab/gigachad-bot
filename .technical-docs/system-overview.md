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

## Canvas background

The dot grid uses small, screen-sized dots and wider spacing at reduced zoom.
Intermediate dots fade with an eased curve between density levels, while the
grid stays anchored to canvas coordinates during pan and zoom. Base dot opacity stays
between 85% and 95% for visibility. Light mode uses the darker `--ink-muted` color,
while dark mode keeps `--ink-faint`.

## Architecture workspaces

Each project holds one LikeC4 model, either under `docs/architecture/` of its linked
workspace on disk or, without one, under `graph/<slug>/` in PostgreSQL:
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
  flight triggers one reread after it. Each `.c4`
  file also opens in the ordinary text editor (with its assistant sidebar and
  inline edit) from the chat's Documents panel, where "New Document >
  Architecture" creates an empty one to write by hand. Saves go through
  `PUT /api/architecture/{slug}/source`.
  When the project has architecture sources, the chat Documents panel offers
  "Attach architecture to chat". It uploads one text snapshot with a fenced
  LikeC4 block for each `graph/<slug>/**/*.c4` source, labeled by its path
  inside the workspace. The existing attachment pipeline includes that
  snapshot in the next chat request; edits made afterwards need a new
  attachment. There is no separate stored chat architecture context. The
  ordinary document list still attaches individual sources. `DirectoryFiles`
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
"Usage" line with the highest percentage, and opens into one meter per limited
account (click for each window). Accounts without limits are hidden.


## Workspaces

File roots are mounted backend directories, listed in the sidebar's
"Workspaces" section below "General". "Add workspace" opens a directory
picker (`features/navigation/WorkspacePicker.tsx`) that starts at the backend
user's home: clicking a directory opens it, the up button opens its parent, the
list follows a typed path, and Enter mounts the path in the input. It lists
directories through `GET /api/fileroots/browse?path=`, which browses the whole
host (directory names only), since any directory can be mounted. Mounting goes
through `POST /api/fileroots/roots` as before.

The tree lists every file a root holds by its full name, skipping only
`.gitignore` matches and `.git`, `node_modules`, `.obsidian` and `.trash`.
Opening a file picks its viewer by type: PDFs open in the PDF viewer, `.canvas`,
`.c4`, `.plot.json` and markmap documents keep their renderers, and everything
else opens as editable text (Neovim when enabled). Files that are not UTF-8 text
are refused (`GET /api/fileviewer/text` returns 415) rather than decoded lossily,
so a save can never overwrite a binary file.

Dragging workspace files onto a project's "files" adds them by reference: the
project stores the file's absolute path (`POST /api/fileroots/add-to-project`),
so it always shows the current file, and saving it in the editor writes the
workspace file. Deleting it from the project only removes the reference, the
file stays on disk. PDFs still go into the shared PDF library as before.

## Neovim editor

Text documents can be edited in a real Neovim with the backend user's own
config, rendered in the browser by xterm.js. The "Neovim editor" switch in
Settings is a per-device preference (`localStorage`) and appears only when
`GET /api/config` reports `neovim_available`: the login is listed in
`GIGACHAD_NVIM_USERS`, `nvim` is on the backend's `PATH`, and
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
