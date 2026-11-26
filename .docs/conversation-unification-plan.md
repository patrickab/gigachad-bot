# Conversation unification plan

Working plan: one conversation UI for chats and Pi sessions. Delete this file once its steps are merged and the lasting parts are in `.docs/architecture.md`.

## Goal

- One `ConversationColumn` and one turn card for the chat and Pi, a flag (`kind: "chat" | "pi"`) steers what differs
- Same actions on every turn: copy, branch, edit, delete
- Removed: regenerate (chat), read aloud (Pi), Pi's arrow "Edit in a new branch"
- Chat tool calls look like Pi's tool rows
- Pi's composer uses the chat's attachment chips
- Long chats and sessions load a page of turns, not all of them
- A chat picker on Alt+C, like the session picker on Alt+A: every chat grouped by project, full-text search, jump to the hit
- What only one side has stays: web search results, research trace and project documents in the chat; session search, queue, held messages, roster and compaction cards in Pi

## Decisions (from the user)

- **Delete** means deleted entirely: the turn is gone from the Pi session file on disk and from the session's context, and from the chat's message list. No `context_edit`, no fork, no hidden copy, no backup kept for undo
- A session restart for a Pi delete is fine
- **Edit** happens in the card: Enter confirms, the old turn is deleted, the edited prompt is sent as a new turn (Esc cancels, Shift+Enter is a newline). It never creates a session
- **Branch** is the only action that creates a new session (Pi) or history file (chat)
- **Edit of an earlier turn** deletes that turn and every turn after it, entirely, then sends the edited prompt: the conversation continues from the edit. Delete alone removes one turn and keeps the later ones

## State when this plan was written

- `master` holds six perf commits (`824cdd2f` .. `f984f8d1`): markdown block memo, no `AnimatePresence`, collapsed-preview head (`packages/chat-ui/src/previewHead.ts`), Pi live turn (stable rows, fold beyond `LIVE_FLAT_PIECES`), 12 ms render cap, incremental code highlighting
- `perf` worktree (`~/git/worktrees/gigachad-bot/perf`) has the same six on top of the React Compiler and pi-host perf commits, which are not on `master`
- `master`'s working tree has someone else's uncommitted work (`features/pi/PiTab.tsx`, `sessionStore.ts`, `usePiSidebar.tsx`, `pi-host/src/catalog`, `pi-host/src/host`). It makes 3 tests of `PiTab.test.tsx` fail. Do not touch or stage it
- Work on a new branch and worktree off `master` (`/worktree` skill), rebase when that uncommitted work lands
- Checks: `factory/gates.sh lint|types|frontend|pi-host|python`. Read `.docs/pi-migration.md` before pi-host work

## Facts the plan rests on

- **Chat**: messages are React state (`useChatStream`), `removePair` already deletes a pair and returns a restore function (undo toast via `UndoDeleteContext`). Regenerate is `regenerateAt`
- **Pi session file**: JSONL, a header line, then entries linked by `id` / `parentId` into a tree, the leaf is the current position (pi docs `session-format.md`)
- **Entries that point at others**: `context_edit.targetId`, `label` targets, `compaction.firstKeptEntryId`
- **One writer** (M1): pi-host holds `flock` on the file for the life of the `pi --mode rpc` child, and the child keeps the session in memory. A file changed under a live child is overwritten or forks silently
- **Pi RPC** has no command to remove entries. `fork` and `clone` create a new session file, pi-host blocks them on a live child (`SESSION_SWITCHING_COMMANDS`)
- **Resync** (M2) already answers `mode: replace` when the prefix changed: clients rebuild the transcript
- **Pi TUI** is a writer pi-host cannot see (open risk in `.docs/pi-migration.md`)
- **Turn UI today**: `features/chat/ChatPair.tsx`, `ChatContainer.tsx`, `useStickToBottom.ts`, `usePairExpansion.ts`, `usePairFocus.ts` against `features/pi/PiTranscript.tsx` (`PiTurn`, own pinning, paging with `TURN_PAGE_SIZE`, search jump). Shared already: `packages/chat-ui/src/Turn.tsx`, `components/ui/columnResize`, `WideModeToggle`, `ComposerDock`

## Steps

Each step is one reviewable change that leaves both surfaces working. Run the gates after each.

### 1. pi-host: delete a turn from a session file

- New pure function in `pi-host/src/host/local/sessions.ts`: `removeTurn(entries, promptEntryId, { through })` returns the new entry list
  - `through: "turn"` (delete) removes that one turn, `through: "end"` (edit) removes it and everything after it on the active path, the entry before the prompt becomes the leaf
  - The turn is the prompt entry and every entry after it on the active path up to, not including, the next user prompt (or a `compaction` entry)
  - The first surviving entry after the turn takes the turn's parent as `parentId`
  - Side branches rooted inside the removed range are removed with it
  - `model_change`, `thinking_level_change` and `session_info` entries inside the range stay, re-linked into the chain
  - `context_edit` and `label` entries whose target is gone are removed
  - A `compaction.firstKeptEntryId` inside the range moves to the next surviving entry
  - Refused: a prompt entry that is not a user message, an unknown id, and a turn a later compaction has summarized (its content lives on inside the summary, which cannot be taken apart: the answer says so)
  - With `through: "end"` a compaction after the prompt is removed with the rest, so only a compaction that still survives can refuse
- New `Host.deleteTurn(sessionPath, entryId)`: read, `removeTurn`, write to a temp file in the same folder, rename over the session file
- New supervisor route `POST /sessions/:id/turns/:entryId/delete`, body `{ through: "turn" | "end" }`
  - 409 while a run is streaming or a dialog waits
  - Stop the child, rewrite the file while pi-host still owns the session, start the child again on the same path, re-take the `flock` on the new inode
  - The restarted child builds its context from the rewritten file, so the turn is out of the model's context from the next prompt on
  - Broadcast a resync with `mode: replace`, emit `sessions_changed`
- **Nothing of the turn stays behind.** After the rename:
  - No backup: the temp file is the new file, the old inode is unlinked by the rename, nothing is copied aside
  - pi-host's derived data is rebuilt from the new file: session summary index (the persisted one too), search cache, preview tail, the replay buffer of the current turn
  - Tool images of the turn are removed from the media store (`/api/pi/media/<sha256>`) unless another entry of any session still references the hash
  - Files the turn's prompt stored in the session folder's `.pi/pastes/` are removed unless a surviving entry mentions them
  - A test asserts it: after a delete, no byte of the turn's prompt, answer or tool output is found in the session file, the summary index or the search cache
- Tests in `pi-host/test`: `removeTurn` cases above on fixture files, the route against a live fake child, a held lock, a running turn
- Backend: `src/lib/pi_host.py` and `routes/pi.py` pass the route through (host access, trusted request)
- Update `.docs/pi-migration.md`: a new decision row (delete rewrites the file), the M1 note, the feature table

### 2. Extract `ConversationColumn` with Pi as its only user

- New `features/conversation/ConversationColumn.tsx`, no behaviour change for Pi
- Owns: scroll container, follow-the-end, paging (newest page mounted, earlier on scroll or "Show earlier turns", reading window held when away from the end), open/collapsed state, column width, wide mode, composer padding
- Takes: `turns: { key, label, icon?, live }[]`, `renderPreview(turn)`, `renderBody(turn)`, optional `reveal` (a turn to mount, open and scroll to: Pi's search hit, the chat's `focus`), `footer`
- `PiTranscript.tsx` keeps what is Pi's: piece derivation, `Answer`, renderers, search highlights, held messages
- Tests: move Pi's scroll and paging tests to the column, keep Pi's rendering tests

### 3. Move the chat onto the column

- `ChatContainer` renders `ConversationColumn`, `useStickToBottom`, `usePairExpansion` and the paging-free list go
- The column gains what only the chat has, for both: keyboard focus on turns (`usePairFocus`), lazy previews near the viewport, the "settling" pin until the reader first interacts
- The chat gains paging: a long history mounts its newest page
- Remove the offscreen special case in `app/globals.css` on `perf` (`[data-chat-column]:first-child`) once both use the same column
- Manual pass in the running app: open at the end, follow a stream, scroll away and back, jump to a search hit, resize, wide mode

### 4. One turn card and one set of actions

- New `features/conversation/TurnCard` (or grow `packages/chat-ui/src/Turn.tsx`): header, preview, question region, answer surface, actions
- Actions come from one list: copy, branch, edit, delete. `kind` decides the handlers, not the buttons
- Remove: `regenerateAt` and its button, `ReadAloudAction`, Pi's "Edit in a new branch" arrow and the `insertion` prefill in `PiTab.tsx`
- **Edit in the card**: the question turns into a textarea with the prompt, Enter confirms, Esc cancels, Shift+Enter newline. Disabled while a run streams
  - Chat: cut the message list at the edited pair (it and every later pair go, the history file is saved), then `send` with the edited text and the turn's attachments
  - Pi: delete route of step 1 with `through: "end"`, then `prompt` with the edited text and the turn's images
  - Editing a turn that has later turns asks for confirmation first, naming how many turns go. The last turn needs none
  - If the send fails after the delete, the edited text stays in the composer: the old turns are gone either way
- **Delete**
  - Pi: delete route of step 1, after a confirmation. No undo: an undo would need the removed entries kept somewhere
  - Chat: the same confirmation, no undo toast, so both behave alike. The pair leaves the message list and the saved history file in the same step (`removePair`, then the save)
  - The browser drops the turn too: transcript state, the collapsed-preview cache, a draft of an edit
- `ChatPair` and `PiTurn` shrink to `renderPreview` and `renderBody`
- Tests: edit and delete for both kinds, edit of an earlier turn removes the later ones, actions hidden on a live turn

### 5. Chat tool calls as Pi rows

- Move `features/pi/tools/` (row, process details, diff view, format) and `features/pi/renderers.ts` to `features/conversation/tools/`
- Adapter: a chat `ToolCallRecord` becomes `{ call, run }` (`status`, `error`, `summary` map to the run)
- The chat's rich results register as renderers, like Pi's: sandbox output, web sources, research trace, mindmap, plots
- Chat answers fold activity into "Process details" like Pi's, `standalone` placement for results that must stay visible
- `ToolCallElement.tsx` goes

### 6. Pi composer attachments

- `PiComposer` shows the chat's chips row above the input (the row in the chat's composer), replacing its own chips
- Sending stays Pi's: images as image blocks, files and videos as `@path` mentions, pasted videos in `.pi/pastes/`
- A turn's images show on its question like the chat's attachments (already `MessageAttachments`)

### 7. Chat picker with search (Alt+C)

- Free today: no `Alt+C` binding exists (taken: `Alt+A` sessions, `Alt+S`, `Alt+R`, `Alt+O`, `Alt+T`, `Alt+W`, `Alt+X`)
- UI: the session picker's shape (`features/pi/PiSessionPicker.tsx` on `components/ui/CommandPicker`'s `FloatingPicker`), not a copy
  - Pull the parts both need out of `PiSessionPicker` into a shared picker: grouped rows, debounced search, hit rows with a snippet, preview pane, keyboard stepping
  - Groups are projects, chats without a project last under "General", newest first inside a group
  - Without a query it lists every chat (`/api/chat-histories`), with one it lists hits: the chat's title row, then one row per hit with its snippet
  - Enter opens the chat in the current tab, on a hit it opens at that turn. Esc closes
- Search: new `GET /api/chat-histories/search?q=&limit=` in `routes/histories.py`
  - Searches question and answer text of every history of the login, returns `{ historyFile, projectSlug, title, qaIndex, snippet }`
  - PostgreSQL holds the histories: use its full-text search (a `tsvector` over the message text, a migration in `db/schema/`) rather than scanning JSON per request
  - Bounded like the Pi search: a result limit and a snippet length
- Jump to the hit: the picker opens the chat with `focus: { qaIndex }`, which is the column's `reveal` of step 2: the turn is mounted, opened and scrolled to
  - Paint the hits in the opened chat with the same CSS Custom Highlight code Pi uses (`useSearchHighlights`), moved into the column, so Enter and Shift+Enter step through hits in both
- Deleted and edited turns leave the search with them: the index follows the saved history
- Tests: the route (hits, project grouping, another login's chats never returned), the picker (grouping, search rows, Enter opens at the turn)

### 8. Close out

- Regenerate `.docs/architecture.md` and `.docs/module-map.md` (`/repomix`), move this plan's lasting parts there, delete this file
- Re-run the streaming probe (20 turns, then an 8,300-character stream, production build) and the live-turn benchmark (300 tool calls) and compare with the numbers in the perf commits

## Risks

- **File rewrite under a live child**: the child must be stopped before the write and started after, a crash in between must leave either the old or the new file (temp file plus rename)
- **The Pi TUI** open on the same session writes without the lock: a delete can lose its appended entries. Refuse when the file changed between read and rename (size and mtime)
- **Prompt cache**: deleting a middle turn invalidates the cache from that point, the next prompt costs a full read
- **What a delete cannot reach**: the provider's own prompt cache and logs until they expire, sessions branched from this one before the delete (each holds its own copy of the turn), a compaction summary written while the turn existed (refused, see step 1), and files the turn's tools changed in the checkout
- **Two pinning behaviours**: the chat settles late layout until the reader interacts, Pi holds a reading window. The column must keep both
- **jsdom has no layout**: scroll behaviour needs the manual pass of step 3
- **Shared package**: `packages/chat-ui` is also used by the LearnHarness overlay, card changes must not break it

## Cleanliness refactors (separate from the steps above)

### Split `PiTranscript.tsx`

- Piece derivation (`answerPieces`, `assistantPieces`, `headPieces`) moves beside `transcript.ts`
  - Pure, testable without rendering
  - Can be cached per message instead of rebuilt for the whole live turn each frame
- Rendering, search jump and held messages stay as small files

### Explicit renderer registration

- One `renderers/index.ts` lists every tool and message renderer
  - Replaces the side-effect imports in `PiTranscript.tsx` (`import "./FactoryTool"`)
  - Shows what is registered in one place
  - Lets heavy renderers (video preview, diff view) load lazily

### `PiTab.tsx` as a state machine

- States: no session, starting, attached, closed
  - Replaces the `launched` and `firstSent` refs that guard effects
  - Start, resume, branch and first-submit hand-off become transitions
- Roster, dock and header move out as components that read the state

### Structured subagent data

- pi-host forwards the subagent roster as a typed event
  - Replaces parsing the `PI_SUBAGENT_ASYNC_JSON:` text widget
  - Replaces the inspect round trip through a prompt
- Cost: touches the contract with the unmodified pi-subagents extension

### Drop the session-list poll

- `sessionStore.ts` refreshes on `sessions_changed`, socket reopen, focus and visibility
  - The 30 s timer is redundant if the host event is reliable
  - Verify with pi-host tests first that every list change emits the event

### Ambient pause in `deepfocus-ambient`

- A `paused` flag in `NightOceanFrame`: no draw, video paused, last frame held
  - Lets the shell pause for an unfocused window without the pop `still` causes
  - Own repository, a battery measure, not a smoothness one
