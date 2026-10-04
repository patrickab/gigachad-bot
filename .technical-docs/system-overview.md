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
- **Brave LLM Context** — backend web search retrieves labelled source
  passages using `BRAVE_API_KEY`, then has the selected model produce a
  citation-grounded answer. No local search service is required.
- **GPT-Researcher** powers `/api/research` with its built-in DuckDuckGo
  retriever and streams progress and results over SSE.
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

## Architecture workspaces

Each project can hold one LikeC4 model under `graph/<slug>/` in PostgreSQL:
every `.c4` file from that root downward (`backend/backend.c4`,
`views/checkout.c4`, ...), merged by LikeC4 into one model. The `.c4` text is
the source of truth, written the LikeC4 way:

- **Packages and modules.** A package is a folder whose own file declares one
  system and the view of it (`backend/backend.c4`: `backend = system 'Backend'`,
  `view backend of backend { include * }`). The first package also declares the
  element kinds. A module is a file beside it adding one container
  (`backend/api.c4`: `extend backend { api = container 'API' }`), so it shows in
  the package's view. The service classifies every file by what it declares
  (`model.tree`: `package`, `module`, `view`, or `file`), never by its name, so
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
never followed. Only the database location is wired to routes today.

- **`src/c4/`** — a stateless loopback Node service (LikeC4 + Langium).
  `POST /model` parses sources, applies the saved layouts, and returns views,
  elements, kinds, the file tree, and errors.
  `POST /apply` turns operations into surgical text edits at syntax-tree
  offsets, re-parses, and rejects an edit that introduces errors:
  - Model operations: `addElement`, `addRelation`, `delete`, `setTitle`,
    `setDescription`, `setKind`, `setLabel`, `rename`, `reparent`.
  - File and view operations: `createPackage`, `createModule`, `createView`,
    `addFileView`, `includeInView`, `removeFromView`.
  - `layout`, which moves nodes and bends connections.

  Delete never cascades: a delete that leaves a reference behind (a
  connection, or a view naming the element) is refused. Edits to an element go
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
  revision-checked, which is accepted for a single user's workspace. Packages,
  modules, and saved views are created through `POST .../ops`
  (`createPackage`, `createModule`, `createView`), raw files through
  `POST .../source`. Files are deleted together
  (`DELETE .../source?path=a&path=b`) only when the merged model still
  parses. A refusal names the remaining file and line that still refer to
  them. Responses are `{model, created}`. Reading a project without an
  architecture returns an empty one. The project's document list shows
  sources by their path inside the workspace. Layout files are not listed.
- **Frontend.** The canvas "+" menu has one "Architecture" entry on any canvas
  of a project, including the unsaved scratch canvas while a project is open.
  Without a project the menu says to open a project canvas instead. Its window
  (`components/ArchitectureWindow.tsx`) holds a collapsible tree (packages
  with their modules, views, other files)
  with "+ Package", "+ Module" (under the current package), "+ View", and
  Delete on the current row (a package goes with its modules). The window's
  path is the file it is on; tree clicks change it without an undo step. A
  package or view file draws its view, a module draws its package's view with
  the module selected (`ArchitectureDiagramSurface`, React Flow + rough.js),
  and the Diagram/Text switch shows the file in the `.c4` text editor. A file
  without a view offers "Add a view of this file".
  - **Cards are paper until activated.** A click activates (selects) a card;
    only then does it move, and do its title, bullets, and kind respond. On a
    card that is not activated, a drag pans and the pen draws, like on the
    bare canvas.
  - **Nesting is drawn.** A shape drawn inside an element (any element, a leaf
    then becomes a parent) creates a child of it. On the bare canvas of a
    package view it joins the package; a saved view creates nothing there,
    since it belongs to no element (it has no "+" either).
  - **Saved views.** The toolbar adds "Add to view" (a whole file's systems,
    or single elements grouped by file).
  - **Fitted before drawing.** `fitGraph` (in `lib/architecture.ts`) sizes
    the view once per change, before anything renders: each card is at least
    as wide as its heading plus kind label (measured in the real font,
    `nodeMinWidth`), overlapping siblings are pushed right, and parents grow
    around their children, innermost first. The fit is display only; a node's
    fitted geometry is saved once it is moved or resized.
  - **Removing nodes.** A node's corner X (on hover or selection) and the
    Delete key do the same thing: in a saved view they only remove the node
    from the view, in a package or module view they delete the element from
    the model. Connections are always model edits.
  - **Auto layout** (toolbar, once positions are saved) clears them and lets
    LikeC4 lay the view out again. Elements added to an arranged view land as
    one block to the right of it.
  - **Views using filters or styles** are read-only on the canvas.
  - **Compounds** show only their title. Their bullets (the element's
    description) stay hidden on the canvas and are edited in Text.
  - **Errors.** One area under the window header shows a refused edit or a
    failed read, and every parse diagnostic as `file:line: message`, in both
    Diagram and Text mode.

  `lib/architecture.ts` diffs each surface edit into operations. The window
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
  Deletes of history files and folders, projects, file-vault roots, sidebar
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
  `src/frontend/lib/grammars/likec4.tmLanguage.json`.

## Memory system

Memory is a **document-backed, LLM-mediated** store (`src/lib/memory_store.py`,
`routes/memory.py`) — no vector DB. It has two independent scopes:

- **global** — durable facts about the user, valid across all projects
  (identity, communication/learning/engineering preferences, goals).
- **project** — facts about one active project (purpose, current focus, key
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
(global profile + the active project's memory, if any) and appends them to
the system prompt under a `# Persistent Memory Context` header — this is how
the LLM actually "remembers" across chats. Users can also list, move a memory
between scopes, or re-bucket memories orphaned by a category rename/deletion
(`remap_orphaned`), all without touching the raw JSON by hand.
