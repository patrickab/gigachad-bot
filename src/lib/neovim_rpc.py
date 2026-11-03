"""Msgpack-RPC with a running Neovim, plus the Lua that bridges its `:w`, yanks and selections back to the caller.

- `SETUP_LUA` tracks the first document's buffer, `_G.gigachad_open` any further one (a private copy each). Documents
  are told apart by an id, the document's path. Every tracked buffer stays loaded (`bufhidden=hide`) when another is shown
- Each tracked buffer's `:w` becomes a `save` request carrying the id instead of writing the disk, and echoes
  `Saved, merged with the agent's edits` when the answer says edits made elsewhere changed the text
- A `BufEnter` autocmd sends a `switched` notification with the id when the window shows another tracked buffer, except for
  the switch `_G.gigachad_open` and `_G.gigachad_show` made. A `BufDelete` autocmd forgets the buffer
- A `TextYankPost` autocmd sends a `yank` notification with the yanked text
- Normal-mode `<Esc>` sends a `leave` notification (the browser gives focus back to the app), after running the user's own
  normal-mode `<Esc>` mapping (`:noh`). Buffer-local `<Esc>` mappings (markdown cells) win over it
- With `ask_keymap`, visual-mode `<leader>a` sends a `selection` notification with the id and the selected lines
- `_G.gigachad_buffer_state` reports a buffer's text, modified flag and change tick
- `_G.gigachad_set_text` rewrites only the changed hunks, for edits made to the file elsewhere, keeping the modified flag
  (or setting it when the text differs from the stored file)
- `_G.gigachad_rename` gives the buffer the file's new name and id, keeping its text, cursor and modified flag
- `_G.gigachad_insert_lines` inserts lines below the cursor of the shown document
- `_G.gigachad_show_error` shows a message as an error
- With `own_clipboard`, `+` and `*` are the caller's clipboard (the browser's, not the machine's Neovim runs on): a paste
  is a `paste` request with the path of the current buffer, answered `{ text }` or `{ error }`. An error is shown and the
  last copy is pasted instead (`yy` then `p` works without the browser's clipboard), a copy stays in Neovim's registers.
  It replaces the provider the user's config chose (`vim.g.clipboard`, even one already in use), the `clipboard` option
  decides which registers reach it
- With `cell_session` the first document is an empty placeholder. `_G.gigachad_set_cells` replaces it with the notebook's code
  buffer and, once markdown cells come, adds a markdown buffer: each holds all cells of its kind in notebook order, each cell led by
  the buffer's marker line (`# %%`, `<!-- %% -->`), the same code for both. Their names are virtual paths beside the notebook when
  the caller gives them (never written: `BufWriteCmd` and `FileWriteCmd` are intercepted, no swap or undo file, the buffers keep an
  empty `buftype` or `vim.fs.root` would not use their names), else private copies. Cells are tracked by extmarks on their marker
  lines, keyed by cell id (unique across both buffers). Changes that alter, remove or add a marker line are undone (the snapshot
  of the last intact text is put back) with a short echo, a cell that lost all its lines gets an empty one. Lines put above the
  first marker (LSP auto-import) move into the first cell. The cursor never rests on a marker line, it goes on into the neighbor
  cell. The window shows one buffer, pinned to its active cell (`scrolloff=0`, top line the cell's first line, the lines outside
  it hidden by `conceal_lines` extmarks, so `conceallevel` is raised to 2, Neovim 0.11+) on `CursorMoved*` and `WinScrolled`, and
  a cursor that leaves the active cell activates the cell it lands in. `_G.gigachad_focus_cell` activates a cell from outside and
  puts its buffer in the window. A rebuild (cells added, removed or moved) drops the undo history, since undoing a step from before
  it would remove a marker line. `_G.gigachad_rename_notebook` gives the buffers the virtual names beside a renamed notebook
- `more` is off and `messagesopt` makes messages time out after the user's config ran, a hit-enter prompt would stop every request
- The notebook buffers send notifications: `cell_edit` (id, text) per changed cell 40 ms after typing (only when its text
  moved past what the caller knows), `cell_focus` (id) when the cursor crossed into another cell of its buffer, `cell_exit` (id,
  `up` or `down`) from `j`/`k`/`<Up>`/`<Down>` on the edge line of a cell whose neighbor in the notebook that way is of the other
  kind (the user's own `gj`-style mapping of the key still applies), `cell_run` (id, text, advance) from `<S-CR>` (true) and
  `<C-CR>` (false), `cell_leave` (id) from normal-mode `<Esc>` in the markdown buffer (after the user's own `<Esc>` mapping), and
  `save_notebook` from `:w`. All but `cell_focus` flush pending `cell_edit`s first; a `cell_exit` from a code cell comes after the
  left cell's render. When a markdown cell's editing ends (`cell_leave`, `cell_run`, `cell_exit`) the code buffer takes the window
  back, pinned to its active cell, so code renders go on. `cell_height` (id, rows) tells the screen rows the active cell takes in
  the window (wrapped lines, folds, virtual lines) when it is pinned and whenever they change. Every code cell's render comes as
  `cell_render` (id, html, gutter, base, rows, height, wrap), `cell_render_drop` (ids) takes renders back (see "Rendering" in the Lua)
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import msgpack

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_READ_CHUNK = 65536

# Receives (channel, first document's private path, its id, optional baseline path, ask_keymap, own_clipboard, cell_session). Mirrors
# Neovim's write rules: empty, otherwise a final newline unless both 'eol' and 'fixeol' are off.
SETUP_LUA = r"""
local channel, first_path, first_id, base_path, ask_keymap, own_clipboard, cell_session = ...
if base_path == vim.NIL then base_path = nil end
-- A message longer than the command line would stop Neovim at a hit-enter or "more" prompt, and a stopped Neovim answers no request
-- (an LSP client quitting is such a message). Set after the user's config, the messages wait and go away by themselves
vim.o.more = false
pcall(function() vim.o.messagesopt = "wait:2000,history:500" end)

-- Buffers of the documents by id, and back
local tracked = {}
local ids = {}
local current_id = nil

local function buffer_of(id)
  local buffer = tracked[id]
  if buffer == nil or not vim.api.nvim_buf_is_loaded(buffer) then error("No open buffer for " .. id, 0) end
  return buffer
end

local function untrack(id)
  local buffer = tracked[id]
  tracked[id] = nil
  if buffer ~= nil then ids[buffer] = nil end
  if current_id == id then current_id = nil end
end

local function buffer_text(buffer)
  local lines = vim.api.nvim_buf_get_lines(buffer, 0, -1, false)
  if #lines == 1 and lines[1] == "" then return "" end
  local text = table.concat(lines, "\n")
  if vim.bo[buffer].eol or vim.bo[buffer].fixeol then text = text .. "\n" end
  return text
end

local function stored_lines(buffer, text)
  local lines = vim.split(text, "\n", { plain = true })
  if #lines > 1 and lines[#lines] == "" then table.remove(lines) else vim.bo[buffer].eol = false end
  return lines
end

local diff_lines = (vim.text and vim.text.diff) or vim.diff
local remote_edit_namespace = vim.api.nvim_create_namespace("gigachad_remote_edit")
local remote_edit_generations = {}
local REMOTE_EDIT_HIGHLIGHT_MS = 1500

local function highlight_remote_edit(buffer, first, count)
  local generation = (remote_edit_generations[buffer] or 0) + 1
  remote_edit_generations[buffer] = generation
  for line = first, first + count - 1 do
    vim.api.nvim_buf_set_extmark(buffer, remote_edit_namespace, line, 0, { line_hl_group = "DiffText" })
  end
  vim.defer_fn(function()
    if generation == remote_edit_generations[buffer] and vim.api.nvim_buf_is_valid(buffer) then
      vim.api.nvim_buf_clear_namespace(buffer, remote_edit_namespace, 0, -1)
    end
  end, REMOTE_EDIT_HIGHLIGHT_MS)
end

-- Rewrites only the changed line ranges, bottom-up, so cursor, view and marks stay on their text. `from`, `to` limit it to
-- the rows [from, to) the new lines replace, the whole buffer by default.
local function apply_lines(buffer, new_lines, highlight, from, to)
  from = from or 0
  local old_lines = vim.api.nvim_buf_get_lines(buffer, from, to or -1, false)
  local hunks = diff_lines(table.concat(old_lines, "\n") .. "\n", table.concat(new_lines, "\n") .. "\n", { result_type = "indices" })
  vim.api.nvim_buf_clear_namespace(buffer, remote_edit_namespace, 0, -1)
  for index = #hunks, 1, -1 do
    local start_old, count_old, start_new, count_new = unpack(hunks[index])
    local first = (count_old > 0 and start_old - 1 or start_old) + from
    local replacement = vim.list_slice(new_lines, start_new, start_new + count_new - 1)
    vim.api.nvim_buf_set_lines(buffer, first, first + count_old, false, replacement)
    if highlight then highlight_remote_edit(buffer, first, count_new) end
  end
end

local function apply_text(buffer, text) apply_lines(buffer, stored_lines(buffer, text), true) end

-- The text `:w` would store, whether it has unsaved edits, and the change tick `gigachad_set_text` checks.
function _G.gigachad_buffer_state(id)
  local buffer = buffer_of(id)
  return { text = buffer_text(buffer), modified = vim.bo[buffer].modified, tick = vim.api.nvim_buf_get_changedtick(buffer) }
end

-- Shows `text` as one undo step, false without a change when the buffer moved past `tick`.
-- An unmodified buffer stays unmodified (the text is on disk) unless `dirty`, a modified one keeps its unsaved state.
function _G.gigachad_set_text(id, text, tick, dirty)
  local buffer = buffer_of(id)
  if tick ~= vim.api.nvim_buf_get_changedtick(buffer) then return false end
  local modified = vim.bo[buffer].modified or dirty == true
  if text ~= buffer_text(buffer) then apply_text(buffer, text) end
  vim.bo[buffer].modified = modified
  return true
end

-- Names `buffer` `new_path`, keeping its modified flag. A buffer left under that name by an earlier rename would refuse it (E95).
local function rename_buffer(buffer, new_path)
  local modified = vim.bo[buffer].modified
  local named
  for _, other in ipairs(vim.api.nvim_list_bufs()) do
    if vim.api.nvim_buf_get_name(other) == new_path then named = other end
  end
  if named == buffer then return end
  if named ~= nil then vim.api.nvim_buf_delete(named, { force = true }) end
  vim.api.nvim_buf_set_name(buffer, new_path)
  vim.bo[buffer].modified = modified
end

-- Renames the buffer for a file that was renamed, `:w` and the saved message follow the new name.
function _G.gigachad_rename(id, new_id, new_path)
  local buffer = buffer_of(id)
  rename_buffer(buffer, new_path)
  tracked[id] = nil
  tracked[new_id] = buffer
  ids[buffer] = new_id
  if current_id == id then current_id = new_id end
end

-- The window the documents show in: the current one, else the first window showing a document.
local function work_window()
  local current = vim.api.nvim_get_current_win()
  if ids[vim.api.nvim_win_get_buf(current)] ~= nil then return current end
  for _, window in ipairs(vim.api.nvim_tabpage_list_wins(0)) do
    if ids[vim.api.nvim_win_get_buf(window)] ~= nil then return window end
  end
  return current
end

-- Makes the work window show the document, which stays loaded behind other documents.
local function display(id)
  local buffer = tracked[id]
  local window = work_window()
  current_id = id
  vim.api.nvim_set_current_win(window)
  vim.api.nvim_win_set_buf(window, buffer)
  vim.bo[buffer].bufhidden = "hide"
end

-- Inserts the lines of `text` below the cursor of the shown document, the cursor lands on the first one.
function _G.gigachad_insert_lines(text)
  local window = work_window()
  local lines = vim.split(text, "\n", { plain = true })
  if #lines > 1 and lines[#lines] == "" then table.remove(lines) end
  local row = vim.api.nvim_win_get_cursor(window)[1]
  vim.api.nvim_buf_set_lines(vim.api.nvim_win_get_buf(window), row, row, false, lines)
  vim.api.nvim_win_set_cursor(window, { row + 1, 0 })
end

-- Shows `text` as an error message.
function _G.gigachad_show_error(text)
  vim.api.nvim_echo({ { text, "ErrorMsg" } }, true, {})
end

-- With `ask_keymap`, visual-mode `<leader>a` in `buffer` tells the caller the selected lines (`selection`). `where(first, last)`
-- gives for the buffer lines `first` .. `last` the path to name, the lines to quote and the line before the first one the caller
-- counts from.
local function install_ask(buffer, where)
  if ask_keymap ~= true then return end
  vim.keymap.set("x", "<leader>a", function()
    local first, last = vim.fn.line("v"), vim.fn.line(".")
    if first > last then first, last = last, first end
    vim.cmd("normal! \27")
    local path, base
    path, first, last, base = where(first, last)
    local lines = vim.api.nvim_buf_get_lines(buffer, first - 1, last, false)
    local selection = { path = path, start_line = first - base, end_line = last - base, text = table.concat(lines, "\n") }
    vim.rpcnotify(channel, "selection", selection)
  end, { buffer = buffer })
end

-- `save` answers { text = stored text, merged = whether edits made elsewhere changed the buffer's text }.
local function track(private_path, id)
  if tracked[id] ~= nil then untrack(id) end
  local buffer = vim.fn.bufadd(private_path)
  vim.bo[buffer].buflisted = true -- `bufadd` leaves it unlisted, `:bnext` would skip it
  tracked[id] = buffer
  ids[buffer] = id
  vim.api.nvim_create_autocmd("BufWriteCmd", {
    buffer = buffer,
    callback = function()
      local text = buffer_text(buffer)
      local ok, stored = pcall(vim.rpcrequest, channel, "save", ids[buffer], text)
      if not ok then error("Save failed: " .. tostring(stored), 0) end
      if stored.text ~= buffer_text(buffer) then apply_text(buffer, stored.text) end
      vim.bo[buffer].modified = false
      local name = vim.fn.fnamemodify(vim.api.nvim_buf_get_name(buffer), ":t")
      local message = stored.merged and "Saved, merged with the agent's edits" or "Saved " .. name
      vim.api.nvim_echo({ { message } }, false, {})
    end,
  })
  install_ask(buffer, function(first, last) return ids[buffer], first, last, 0 end)
end

-- Opens a private copy as a new document and shows it.
function _G.gigachad_open(private_path, id)
  track(private_path, id)
  display(id)
end

-- Shows an open document, false when its buffer is gone (unloaded or deleted) and the document has to be opened again.
function _G.gigachad_show(id)
  local buffer = tracked[id]
  if buffer == nil then return false end
  if not vim.api.nvim_buf_is_loaded(buffer) then
    untrack(id)
    vim.api.nvim_buf_delete(buffer, { force = true })
    return false
  end
  display(id)
  return true
end

-- The notebook: one buffer per kind of cell, a "book" each (`books.code`, `books.markdown`). The code buffer holds all code cells of
-- the notebook as one script, the markdown buffer all markdown cells as one document, each cell led by the book's marker line. A
-- book's buffer is unlisted, on a file name (virtual beside the notebook, or a private copy: filetype detection, the user's plugins
-- and LSP see one file) and never written. One window shows one book, pinned to that book's active cell. The caller holds the
-- notebook, a book `nb` is `{ kind, buffer, marker, order, index, marks, known, lookalikes, snapshot, tick, active, timer, busy,
-- shown }`:
--   order      the ids of the book's cells in notebook order, `index` maps an id to its position
--   marks      the extmark (`cells_namespace`) of every cell's marker line, how a cell is found: never by counting marker text
--   known      the text of every cell as the caller last knew it, what `cell_edit` is measured against
--   lookalikes lines inside the cells' own text that read like a marker and are none
--   snapshot   `{ lines, rows }` the last text whose marker lines were intact, put back when a change breaks them
--   tick       the change tick `snapshot` belongs to, `active` the cell the window is pinned to, `busy` while this code edits
--   shown      whether a window showed the buffer yet
-- `notebook` holds the ids of all cells in notebook order: a cell of the other kind between two cells of one book keeps the cursor
-- from moving straight from one into the other (`cell_exit`). Changes of the cells and moves into another cell leave as `cell_edit`,
-- `cell_focus`, `cell_exit`, `cell_run`, `cell_leave` and `save_notebook`. `nb.render` is the state of `cell_render` (every code
-- cell as styled text, see "Rendering" below), the markdown book has none.
local uv = vim.uv or vim.loop
local EDIT_DEBOUNCE_MS = 40
local cells_namespace = vim.api.nvim_create_namespace("gigachad_cells")
local hidden_namespace = vim.api.nvim_create_namespace("gigachad_cells_hidden")
-- With `cell_session` the first document is an empty placeholder that the code buffer replaces
local placeholder_id = cell_session == true and first_id or nil
local KINDS = { "code", "markdown" }
local books = {}
local notebook = { order = {}, position = {} }

local function book_of(buffer)
  for _, nb in pairs(books) do
    if nb.buffer == buffer then return nb end
  end
  return nil
end

local function book_with(id)
  for _, nb in pairs(books) do
    if nb.index[id] ~= nil then return nb end
  end
  return nil
end

local function forget_book(nb)
  if books[nb.kind] ~= nb then return end
  nb.timer:stop()
  nb.timer:close()
  if nb.render ~= nil then
    nb.render.timer:stop()
    nb.render.timer:close()
  end
  books[nb.kind] = nil
end

-- `flush_edits` and the cell changes call these, they are defined with the rendering below: a render waits for a pause in the typing,
-- and a cell that is left is rendered at once so that the caller never shows its old text
local render_later, render_now = function() end, function() end

-- The window showing the book's buffer: the current one, else any, nil when none does.
local function notebook_window(nb)
  local current = vim.api.nvim_get_current_win()
  if vim.api.nvim_win_get_buf(current) == nb.buffer then return current end
  local window = vim.fn.bufwinid(nb.buffer)
  return window ~= -1 and window or nil
end

-- The row (0-based) of a cell's marker line.
local function mark_row(nb, id)
  return vim.api.nvim_buf_get_extmark_by_id(nb.buffer, cells_namespace, nb.marks[id], {})[1]
end

-- The rows [first content row, row after the last) of the cell at `index`, 0-based.
local function cell_range(nb, index)
  local next_id = nb.order[index + 1]
  local stop = next_id ~= nil and mark_row(nb, next_id) or vim.api.nvim_buf_line_count(nb.buffer)
  return mark_row(nb, nb.order[index]) + 1, stop
end

-- The cell holding the 0-based `row`, and whether the row is its marker line.
local function locate(nb, row)
  local low, high, found = 1, #nb.order, nil
  while low <= high do
    local middle = math.floor((low + high) / 2)
    if mark_row(nb, nb.order[middle]) <= row then
      found, low = middle, middle + 1
    else
      high = middle - 1
    end
  end
  return found, found ~= nil and mark_row(nb, nb.order[found]) == row
end

local function cell_lines(text) return vim.split(text, "\n", { plain = true }) end

-- The lines of the script for `cells` and the row of every marker.
local function script_of(cells, marker)
  local lines, rows = {}, {}
  for index, cell in ipairs(cells) do
    rows[index] = #lines
    lines[#lines + 1] = marker
    for _, line in ipairs(cell_lines(cell.text)) do lines[#lines + 1] = line end
  end
  if #lines == 0 then lines[1] = "" end
  return lines, rows
end

local function marker_count(nb, lines)
  local count = 0
  for _, line in ipairs(lines) do
    if line == nb.marker then count = count + 1 end
  end
  return count
end

-- The marker rows in cell order when every marker line is where its extmark says and no other line reads as one, else nil.
local function intact(nb, lines)
  local rows, previous = {}, -1
  for index, id in ipairs(nb.order) do
    local row = mark_row(nb, id)
    if row == nil or row <= previous or lines[row + 1] ~= nb.marker then return nil end
    rows[index], previous = row, row
  end
  if #rows > 0 and rows[1] ~= 0 then return nil end
  if marker_count(nb, lines) ~= #rows + nb.lookalikes then return nil end
  return rows
end

-- A cell has at least one line, as an empty buffer does: gives one to every cell that lost all of its lines. A cursor the
-- deleted line left on the marker below goes onto the new line.
local function fill_empty_cells(nb, lines, rows)
  local filled = false
  local window = notebook_window(nb)
  for index = #rows, 1, -1 do
    local stop = rows[index + 1] or #lines
    if stop == rows[index] + 1 then
      local cursor = window ~= nil and vim.api.nvim_win_get_cursor(window)[1] or nil
      vim.api.nvim_buf_set_lines(nb.buffer, stop, stop, false, { "" })
      if cursor ~= nil and cursor == (rows[index + 1] ~= nil and stop + 1 or stop) then
        vim.api.nvim_win_set_cursor(window, { rows[index] + 2, 0 })
      end
      filled = true
    end
  end
  return filled
end

-- Edits made by this code, they are no change to report or to check. They run in the book's buffer (undo, `undojoin`).
local function quietly(nb, work)
  nb.busy = true
  local ok, failure = pcall(vim.api.nvim_buf_call, nb.buffer, work)
  nb.busy = false
  if not ok then error(failure, 0) end
end

-- Lines above the first marker (an LSP quick fix that adds an import, `:0put`) join the first cell as its first lines: the marker
-- line moves back to the top. False when something else is off. Joins the undo step of the change that put them there.
local function adopt_head(nb, lines)
  local first = nb.order[1] ~= nil and mark_row(nb, nb.order[1]) or nil
  if first == nil or first <= 0 or lines[first + 1] ~= nb.marker then return false end
  for row = 1, first do
    if lines[row] == nb.marker then return false end
  end
  local previous = first
  for index = 2, #nb.order do
    local row = mark_row(nb, nb.order[index])
    if row == nil or row <= previous or lines[row + 1] ~= nb.marker then return false end
    previous = row
  end
  local moved = vim.list_slice(lines, 1, first)
  table.insert(moved, 1, nb.marker)
  quietly(nb, function()
    pcall(vim.cmd, "undojoin")
    vim.api.nvim_buf_set_lines(nb.buffer, 0, first + 1, false, moved)
    vim.api.nvim_buf_set_extmark(nb.buffer, cells_namespace, 0, 0, { id = nb.marks[nb.order[1]] })
  end)
  return true
end

-- Takes the buffer as the new snapshot when its marker lines are intact. False when they are not.
local function settle(nb)
  local lines = vim.api.nvim_buf_get_lines(nb.buffer, 0, -1, false)
  local rows = intact(nb, lines)
  if rows == nil and adopt_head(nb, lines) then
    lines = vim.api.nvim_buf_get_lines(nb.buffer, 0, -1, false)
    rows = intact(nb, lines)
  end
  if rows == nil then return false end
  if fill_empty_cells(nb, lines, rows) then
    lines = vim.api.nvim_buf_get_lines(nb.buffer, 0, -1, false)
    rows = intact(nb, lines)
    if rows == nil then return false end
  end
  nb.snapshot = { lines = lines, rows = rows }
  nb.tick = vim.api.nvim_buf_get_changedtick(nb.buffer)
  vim.bo[nb.buffer].modified = false -- the caller holds the text, nothing is left unsaved here
  return true
end

local function same_lines(first, second)
  if #first ~= #second then return false end
  for index = 1, #first do
    if first[index] ~= second[index] then return false end
  end
  return true
end

-- Puts the last intact text back. A change made in normal mode is undone as such, so it leaves no step behind. Otherwise (insert
-- mode, or the text differs after the undo) the text is replaced in the undo step of the change that broke it.
local function restore(nb)
  local snapshot = nb.snapshot
  quietly(nb, function()
    local undone = false
    if vim.fn.mode() == "n" then
      pcall(vim.cmd, "silent undo")
      undone = same_lines(vim.api.nvim_buf_get_lines(nb.buffer, 0, -1, false), snapshot.lines)
    end
    if not undone then
      pcall(vim.cmd, "undojoin")
      apply_lines(nb.buffer, snapshot.lines, false)
    end
    for index, id in ipairs(nb.order) do
      vim.api.nvim_buf_set_extmark(nb.buffer, cells_namespace, snapshot.rows[index], 0, { id = nb.marks[id] })
    end
  end)
  nb.tick = vim.api.nvim_buf_get_changedtick(nb.buffer)
  vim.bo[nb.buffer].modified = false
  vim.api.nvim_echo({ { "Cell marker lines cannot be changed" } }, true, {})
  -- The cursor stays where it was before the change, the undo would have moved it to the change
  local window = notebook_window(nb)
  if window ~= nil and nb.cursor ~= nil then
    local row = math.min(nb.cursor[1], vim.api.nvim_buf_line_count(nb.buffer))
    local line = vim.api.nvim_buf_get_lines(nb.buffer, row - 1, row, false)[1]
    vim.api.nvim_win_set_cursor(window, { row, math.min(nb.cursor[2], #line) })
  end
end

-- Checks a change made since the snapshot, false when it was idle or fine, true when it broke a marker line and was undone.
local function sync(nb)
  if nb == nil or nb.busy or vim.api.nvim_buf_get_changedtick(nb.buffer) == nb.tick then return false end
  if settle(nb) then return false end
  restore(nb)
  return true
end

-- Tells the caller the cells whose text moved past what it knows, in notebook order.
local function flush_edits(nb)
  if nb == nil or books[nb.kind] ~= nb then return end
  nb.timer:stop()
  sync(nb)
  local lines, rows = nb.snapshot.lines, nb.snapshot.rows
  for index, id in ipairs(nb.order) do
    local text = table.concat(lines, "\n", rows[index] + 2, rows[index + 1] or #lines)
    if text ~= nb.known[id] then
      nb.known[id] = text
      vim.rpcnotify(channel, "cell_edit", id, text)
    end
  end
  render_later(nb)
end

-- The screen rows the active cell took when the caller was last told (`cell_height`), as "id rows".
local told_height = nil

-- Pins the window to the active cell: the top line is its first line (when it fits), everything outside it is hidden. The caller
-- hears how many screen rows the cell takes (`cell_height`: wrapped lines, folds, virtual lines) whenever that or the cell changes,
-- it sizes the terminal to them. The window's height plays no part in them, so the caller's resize never changes them.
local function pin(nb)
  local window = notebook_window(nb)
  local index = window ~= nil and nb.index[nb.active] or nil
  if index == nil then return end
  local start, stop = cell_range(nb, index)
  local total = vim.api.nvim_buf_line_count(nb.buffer)
  -- Setting an option, even to its value, makes the cursor forget the column it wants
  if vim.wo[window].scrolloff ~= 0 then vim.wo[window].scrolloff = 0 end
  if vim.wo[window].conceallevel < 2 then vim.wo[window].conceallevel = 2 end -- lines are only hidden from level 2
  -- Both ranges hold whole lines, the end row included whatever the end column. Moving only the window leaves them as they are
  local key = table.concat({ start, stop, total, vim.api.nvim_buf_get_changedtick(nb.buffer) }, ":")
  if key ~= nb.hidden then
    nb.hidden = key
    vim.api.nvim_buf_clear_namespace(nb.buffer, hidden_namespace, 0, -1)
    if start > 0 then
      vim.api.nvim_buf_set_extmark(nb.buffer, hidden_namespace, 0, 0, { end_row = start - 1, end_col = 0, conceal_lines = "" })
    end
    if stop < total then
      local last = vim.api.nvim_buf_get_lines(nb.buffer, total - 1, total, false)[1]
      vim.api.nvim_buf_set_extmark(nb.buffer, hidden_namespace, stop, 0, { end_row = total - 1, end_col = #last, conceal_lines = "" })
    end
  end
  -- Measured once the cell's own lines are not hidden: a hidden line takes no row
  local height = vim.api.nvim_win_text_height(window, { start_row = start, end_row = stop - 1 }).all
  vim.api.nvim_win_call(window, function()
    local view = vim.fn.winsaveview()
    local fits = height <= vim.api.nvim_win_get_height(window)
    -- A cell taller than the window scrolls with its cursor, but never above its first line
    if view.topline < start + 1 or (fits and view.topline ~= start + 1) then vim.fn.winrestview({ topline = start + 1 }) end
  end)
  local told = nb.active .. " " .. height
  if told ~= told_height then
    told_height = told
    vim.rpcnotify(channel, "cell_height", nb.active, height)
  end
end

-- Makes `id` the active cell and puts the cursor in it: "first" or "last" line, "keep" it where it is when that is inside,
-- or `{ offset, col }` the line offset (0-based) and column to land on, clamped to the cell. Pins the window.
local function show_cell(nb, id, cursor)
  local start, stop = cell_range(nb, nb.index[id])
  nb.active = id
  local window = notebook_window(nb)
  if window ~= nil then
    local row, col = start + 1, 0
    if cursor == "last" then
      row = stop
    elseif cursor == "keep" then
      local here = vim.api.nvim_win_get_cursor(window)
      if here[1] > start and here[1] <= stop then row, col = here[1], here[2] end
    elseif type(cursor) == "table" then
      row, col = math.min(start + 1 + cursor.offset, stop), cursor.col
    end
    local line = vim.api.nvim_buf_get_lines(nb.buffer, row - 1, row, false)[1] or ""
    vim.api.nvim_win_set_cursor(window, { row, math.min(col, #line) })
  end
  pin(nb)
  if window ~= nil then nb.cursor = vim.api.nvim_win_get_cursor(window) end
end

-- Keeps the cursor off marker lines, activates the cell it is in and pins the window to it. Moving onto a marker goes on
-- into the neighbor: down into the first line of the next cell, up into the last line of the previous one (the first
-- cell's own first line when there is none). A jump onto a marker lands on the first line of that cell.
local function follow_cursor()
  local window = vim.api.nvim_get_current_win()
  local nb = book_of(vim.api.nvim_win_get_buf(window))
  if nb == nil or nb.busy or #nb.order == 0 then return end
  sync(nb)
  local view = vim.fn.winsaveview()
  local index, on_marker = locate(nb, view.lnum - 1)
  if index == nil then return end
  if on_marker then
    local leaving_up = index == nb.index[nb.active] and index > 1
    local target = leaving_up and index - 1 or index
    local start, stop = cell_range(nb, target)
    local row = leaving_up and stop or start + 1
    local line = vim.api.nvim_buf_get_lines(nb.buffer, row - 1, row, false)[1]
    local insert = vim.fn.mode():sub(1, 1) == "i"
    local col = math.min(view.curswant, math.max(#line - (insert and 0 or 1), 0))
    vim.fn.winrestview({ lnum = row, col = col, curswant = view.curswant })
    index = target
  end
  local id = nb.order[index]
  if id ~= nb.active then
    flush_edits(nb)
    local previous = nb.active
    nb.active = id
    render_now(nb, previous) -- the caller gets the left cell's new look before it shows the cell as a snapshot
    vim.rpcnotify(channel, "cell_focus", id)
  end
  pin(nb)
  nb.cursor = vim.api.nvim_win_get_cursor(window)
end

local function on_change(nb)
  if books[nb.kind] ~= nb then return end
  if sync(nb) then follow_cursor() end
  if not nb.timer:is_active() then nb.timer:start(EDIT_DEBOUNCE_MS, 0, vim.schedule_wrap(function() flush_edits(nb) end)) end
end

-- The keys the user's own mapping of `key` in `mode` gives (an expression such as `v:count == 0 ? 'gj' : 'j'`, or a plain
-- non-recursive one), `key` itself without one.
local function own_keys(mode, key)
  for _, map in ipairs(vim.api.nvim_get_keymap(mode)) do
    if map.lhs == key and map.expr == 1 then
      local ok, keys = pcall(function() return map.callback ~= nil and map.callback() or vim.api.nvim_eval(map.rhs) end)
      if ok and type(keys) == "string" then return keys end
    elseif map.lhs == key and map.noremap == 1 and map.callback == nil and (map.rhs or "") ~= "" then
      return map.rhs
    end
  end
  return key
end

-- A markdown cell's editing ended (`from` the markdown book): the code buffer goes back into the window, pinned to its active cell,
-- so that code cell renders follow the disk, diagnostics and the theme while no markdown cell is edited. Called after the frame
-- that ends the editing.
local function show_code(from)
  local code = books.code
  if from.kind ~= "markdown" or books[from.kind] ~= from or code == nil or code.index[code.active] == nil then return end
  if not vim.api.nvim_buf_is_loaded(code.buffer) or notebook_window(code) ~= nil then return end
  local window = notebook_window(from)
  if window == nil then return end
  if vim.fn.mode():sub(1, 1) == "i" then vim.cmd.stopinsert() end -- the editing ended, a run key may come from insert mode
  vim.api.nvim_win_set_buf(window, code.buffer)
  show_cell(code, code.active, "keep")
  render_later(code)
end

-- An up or down key (`step` -1 or 1) in a book's buffer. On the first or last line of a cell (the first or last screen line of it
-- for a screen-line motion such as `gj`) whose neighbor in the notebook in that direction is a cell of the other kind, the cursor
-- leaves the buffer: the caller gets `cell_exit` (after the edits) and activates that neighbor. Otherwise the key moves as the
-- user's own mapping of it does, into the next cell of this buffer too (`cell_focus`). A count (`5j`) that does not start on the
-- edge line moves through the markers like `G` or a search does, so it can land in a later cell of this buffer past a cell of the
-- other kind.
local function vertical(nb, mode, key, step)
  local keys = own_keys(mode, key)
  local window = vim.api.nvim_get_current_win()
  if #nb.order == 0 or vim.api.nvim_win_get_buf(window) ~= nb.buffer then return keys end
  local row = vim.api.nvim_win_get_cursor(window)[1]
  local index, on_marker = locate(nb, row - 1)
  if index == nil or on_marker then return keys end
  local id = nb.order[index]
  local position = notebook.position[id]
  local neighbor = position ~= nil and notebook.order[position + step] or nil
  if neighbor == nil or nb.index[neighbor] ~= nil then return keys end
  local start, stop = cell_range(nb, index)
  if row ~= (step > 0 and stop or start + 1) then return keys end
  if keys:sub(1, 1) == "g" and vim.wo[window].wrap then
    local column = vim.fn.virtcol(".")
    local range = step > 0 and { start_row = row - 1, end_row = row - 1, start_vcol = column - 1 }
      or { start_row = row - 1, end_row = row - 1, end_vcol = column }
    local height = vim.api.nvim_win_text_height(window, range)
    if height.all - height.fill > 1 then return keys end
  end
  vim.schedule(function() -- an expression mapping may not change text, the flush can
    if books[nb.kind] ~= nb then return end
    flush_edits(nb)
    render_now(nb, id) -- the caller shows the left code cell as its render from now on
    vim.rpcnotify(channel, "cell_exit", id, step > 0 and "down" or "up")
    show_code(nb)
  end)
  -- A count typed before the key would stay pending (and Neovim answers no request meanwhile): `<Esc>` drops it
  return mode == "n" and vim.v.count > 0 and "<Esc>" or ""
end

-- The user's own normal-mode `<Esc>` (assigned below, after the user's config ran)
local run_user_escape

local function install_book(nb)
  local buffer = nb.buffer
  local changed = function() on_change(nb) end
  vim.api.nvim_create_autocmd({ "TextChanged", "TextChangedI", "TextChangedP" }, { buffer = buffer, callback = changed })
  -- `<leader>a` quotes lines of the active cell, counted from the cell's first line
  install_ask(buffer, function(first, last)
    local index = nb.index[nb.active]
    if index == nil then return vim.api.nvim_buf_get_name(buffer), first, last, 0 end
    local start, stop = cell_range(nb, index)
    first = math.min(math.max(first, start + 1), stop)
    return vim.api.nvim_buf_get_name(buffer), first, math.max(first, math.min(last, stop)), start
  end)
  -- `:w` asks the caller to save the notebook, after the last edit
  vim.api.nvim_create_autocmd("BufWriteCmd", {
    buffer = buffer,
    callback = function()
      flush_edits(nb)
      vim.rpcnotify(channel, "save_notebook")
      vim.bo[buffer].modified = false
      vim.api.nvim_echo({ { "Saving notebook" } }, false, {})
    end,
  })
  -- `:1,2w!` has no `BufWriteCmd`, it writes the buffer's file as `FileWriteCmd` (`acwrite` would stop it too, but a buffer with a
  -- `buftype` makes `vim.fs.root` ignore its name, and the LSP would find no project)
  vim.api.nvim_create_autocmd({ "FileWriteCmd", "FileAppendCmd" }, {
    buffer = buffer,
    callback = function() vim.api.nvim_echo({ { "The notebook is only written whole, with :w" } }, true, {}) end,
  })
  -- `<S-CR>` runs the cell and the caller moves on to the next one, `<C-CR>` runs it in place
  for key, advance in pairs({ ["<S-CR>"] = true, ["<C-CR>"] = false }) do
    vim.keymap.set({ "n", "i" }, key, function()
      flush_edits(nb)
      if nb.active ~= nil then vim.rpcnotify(channel, "cell_run", nb.active, nb.known[nb.active], advance) end
      vim.schedule(function() show_code(nb) end)
    end, { buffer = buffer })
  end
  local motions = {
    { "n", "j", 1 }, { "n", "<Down>", 1 }, { "n", "k", -1 }, { "n", "<Up>", -1 }, { "i", "<Down>", 1 }, { "i", "<Up>", -1 },
  }
  for _, motion in ipairs(motions) do
    local mode, key, step = motion[1], motion[2], motion[3]
    vim.keymap.set(mode, key, function() return vertical(nb, mode, key, step) end, { buffer = buffer, expr = true })
  end
  -- `<Esc>` in normal mode leaves a markdown cell: the user's own `<Esc>` runs, then the caller shows the cell rendered again
  if nb.kind == "markdown" then
    vim.keymap.set("n", "<Esc>", function()
      run_user_escape()
      flush_edits(nb)
      if nb.active ~= nil then vim.rpcnotify(channel, "cell_leave", nb.active) end
      vim.schedule(function() show_code(nb) end)
    end, { buffer = buffer })
  end
end

vim.api.nvim_create_autocmd({ "CursorMoved", "CursorMovedI", "WinScrolled" }, { callback = follow_cursor })
-- A scroll that `WinScrolled` did not report (a second one right after the pin) is caught when Neovim goes idle
vim.api.nvim_create_autocmd("SafeState", {
  callback = function()
    follow_cursor()
  end,
})

-- Whether the book `kind` has its buffer, forgetting one that Neovim unloaded.
function _G.gigachad_has_book(kind)
  local nb = books[kind]
  if nb == nil then return false end
  if vim.api.nvim_buf_is_loaded(nb.buffer) then return true end
  forget_book(nb)
  if vim.api.nvim_buf_is_valid(nb.buffer) then vim.api.nvim_buf_delete(nb.buffer, { force = true }) end
  return false
end

-- Names the buffer of every book in `names` (kind to path, the virtual names beside a renamed notebook), keeping its text, cells and
-- modified flag. Writes stay intercepted: they are set up on the buffer, not on its name.
function _G.gigachad_rename_notebook(names)
  for kind, path in pairs(names) do
    if _G.gigachad_has_book(kind) then rename_buffer(books[kind].buffer, path) end
  end
end

-- Makes `private_path` the buffer of the book `kind` with the text `script` (nil: the file holds it). The code buffer takes the
-- placeholder's place in the window, the markdown buffer is only loaded until a `focus_cell` shows it.
local function create_book(kind, private_path, marker, script)
  local buffer = vim.fn.bufadd(private_path)
  local virtual = script ~= nil and script ~= vim.NIL
  if virtual then
    -- The name only exists for the LSP (root markers, `.venv`): nothing is ever written there, no swap or undo file either
    vim.bo[buffer].swapfile = false
    vim.bo[buffer].undofile = false
  end
  if kind == "code" then
    vim.api.nvim_win_set_buf(0, buffer)
  else
    vim.fn.bufload(buffer)
  end
  if virtual then
    local lines = vim.split(script, "\n", { plain = true })
    if lines[#lines] == "" then table.remove(lines) end
    local levels = vim.bo[buffer].undolevels -- the global one stays in charge unless the buffer had its own
    vim.bo[buffer].undolevels = -1 -- the first text is no undo step
    vim.api.nvim_buf_set_lines(buffer, 0, -1, false, lines)
    vim.bo[buffer].undolevels = levels
    vim.bo[buffer].modified = false
  end
  vim.bo[buffer].buflisted = false -- showing a buffer lists it
  vim.bo[buffer].bufhidden = "hide"
  local nb = {
    kind = kind,
    buffer = buffer,
    marker = marker,
    order = {},
    index = {},
    marks = {},
    known = {},
    lookalikes = 0,
    snapshot = { lines = {}, rows = {} },
    tick = vim.api.nvim_buf_get_changedtick(buffer), -- the loaded private copy is no change
    timer = uv.new_timer(),
    busy = false,
    shown = kind == "code",
  }
  if kind == "code" then
    nb.render = {
      timer = uv.new_timer(),
      sigs = {}, -- by cell id: what the last render of that cell was made from
      queue = {},
      running = false,
      generation = 0, -- raised by what changes every cell (colorscheme, size)
      rendering = false,
      semantic = {}, -- by cell id: the semantic tokens of the last valid result
    }
  end
  books[kind] = nb
  install_book(nb)
  if kind == "code" and placeholder_id ~= nil then
    local placeholder = tracked[placeholder_id]
    placeholder_id = nil
    if placeholder ~= nil and vim.api.nvim_buf_is_valid(placeholder) then vim.api.nvim_buf_delete(placeholder, { force = true }) end
  end
end

-- Builds the script of `cells` from scratch (a changed list of cells). The cursor stays on its line of its cell when that
-- cell is still there, and so does the active cell. A cell the caller still sends as it was told it (`known`) while its
-- text moved on in Neovim keeps the newer text, `cell_edit` brings it to the caller.
local function rebuild(nb, cells)
  local old = nb.snapshot
  local texts = {}
  for index, id in ipairs(nb.order) do
    texts[id] = table.concat(old.lines, "\n", old.rows[index] + 2, old.rows[index + 1] or #old.lines)
  end
  local previous = nb.index[nb.active]
  local window = notebook_window(nb)
  local cursor
  if window ~= nil and #nb.order > 0 then
    local position = vim.api.nvim_win_get_cursor(window)
    local index = locate(nb, position[1] - 1)
    if index ~= nil then
      cursor = { id = nb.order[index], offset = math.max(position[1] - 2 - old.rows[index], 0), col = position[2] }
    end
  end
  local shown = {}
  for index, cell in ipairs(cells) do
    local text = nb.known[cell.id] == cell.text and texts[cell.id] or cell.text
    shown[index] = { id = cell.id, text = text }
  end
  local lines, rows = script_of(shown, nb.marker)
  quietly(nb, function()
    -- The undo steps before a rebuild describe a structure that is gone: undoing one would remove a marker line, and the
    -- protection would then refuse every later `u`. A rebuild that changes text drops the history, the typing before it
    -- stays untouched but can no longer be undone
    local levels = vim.bo[nb.buffer].undolevels
    vim.bo[nb.buffer].undolevels = -1
    local ok, failure = pcall(apply_lines, nb.buffer, lines, false)
    vim.bo[nb.buffer].undolevels = levels
    if not ok then error(failure, 0) end
    vim.api.nvim_buf_clear_namespace(nb.buffer, cells_namespace, 0, -1)
    nb.marks = {}
    for index, cell in ipairs(shown) do
      nb.marks[cell.id] = vim.api.nvim_buf_set_extmark(nb.buffer, cells_namespace, rows[index], 0, {})
    end
  end)
  local order, index, known = {}, {}, {}
  for position, cell in ipairs(cells) do
    order[position], index[cell.id], known[cell.id] = cell.id, position, cell.text
  end
  nb.order, nb.index, nb.known = order, index, known
  nb.lookalikes = marker_count(nb, lines) - #cells
  if not settle(nb) then
    nb.snapshot = { lines = lines, rows = rows }
    nb.tick = vim.api.nvim_buf_get_changedtick(nb.buffer)
  end
  local active = nb.active
  if active == nil or index[active] == nil then
    local fallback = cells[math.min(previous or 1, #cells)]
    active = fallback ~= nil and fallback.id or nil
  end
  nb.active = active
  if active ~= nil then
    show_cell(nb, active, cursor ~= nil and cursor.id == active and cursor or "first")
  end
end

-- Makes the book `kind` hold `cells`, a list of `{ id, text }` in notebook order, creating its buffer first when `private_path` is
-- given. The same ids in the same order only replace the text of cells that differ (minimal hunks, the cursor stays).
local function set_book_cells(kind, private_path, marker, cells, script)
  if private_path ~= nil and private_path ~= vim.NIL then create_book(kind, private_path, marker, script) end
  local nb = books[kind]
  if nb == nil then error("No " .. kind .. " buffer", 0) end
  sync(nb)
  local same = #cells == #nb.order
  for index, cell in ipairs(cells) do
    same = same and cell.id == nb.order[index]
  end
  if same then
    for index, cell in ipairs(cells) do
      local start, stop = cell_range(nb, index)
      local current = table.concat(vim.api.nvim_buf_get_lines(nb.buffer, start, stop, false), "\n")
      -- Text the caller got from this buffer earlier is no news: what was typed since stays
      if cell.text ~= current and cell.text ~= nb.known[cell.id] then
        quietly(nb, function() apply_lines(nb.buffer, cell_lines(cell.text), false, start, stop) end)
      end
      nb.known[cell.id] = cell.text
    end
    nb.lookalikes = marker_count(nb, vim.api.nvim_buf_get_lines(nb.buffer, 0, -1, false)) - #cells
    if settle(nb) then
      if nb.active == nil and #cells > 0 then show_cell(nb, cells[1].id, "first") end
      pin(nb)
      render_later(nb)
      return
    end
  end
  rebuild(nb, cells)
  render_later(nb)
end

-- Makes the books hold their cells: `given` maps a kind to `{ path, marker, cells, script }` (see `set_book_cells`), `order` lists
-- the ids of all cells of the notebook in order. No `cell_edit` comes back for what the caller sent.
function _G.gigachad_set_cells(given, order)
  notebook.order, notebook.position = order, {}
  for position, id in ipairs(order) do notebook.position[id] = position end
  for _, kind in ipairs(KINDS) do
    local book = given[kind]
    if book ~= nil then set_book_cells(kind, book.path, book.marker, book.cells, book.script) end
  end
  return true
end

-- Activates the cell `id`, `cursor` is "first", "last" or "keep". False for a cell that is not there. No `cell_focus` comes back.
-- A cell of the book that the window does not show puts that book's buffer in the window: the cell of the book shown before is
-- left first (its edits, its render). Renders wait while the code buffer is not shown.
function _G.gigachad_focus_cell(id, cursor)
  local nb = book_with(id)
  if nb == nil then return false end
  sync(nb)
  local switched = notebook_window(nb) == nil
  if switched then
    local window = vim.api.nvim_get_current_win()
    for _, other in pairs(books) do
      local shows = other ~= nb and notebook_window(other) or nil
      if shows ~= nil then
        flush_edits(other)
        render_now(other, other.active)
        window = shows
      end
    end
    vim.api.nvim_win_set_buf(window, nb.buffer)
    if not nb.shown then
      -- Its filetype was set while it was loaded out of sight: set again, the user's settings for it reach the window
      nb.shown = true
      vim.api.nvim_win_call(window, function() vim.bo[nb.buffer].filetype = vim.bo[nb.buffer].filetype end)
    end
  end
  local previous = nb.active
  if previous ~= id then flush_edits(nb) end
  show_cell(nb, id, cursor)
  if previous ~= id then render_now(nb, previous) end
  if switched then render_later(nb) end
  return true
end

-- Rendering: every code cell as styled text (`cell_render`, id, html, gutter, base, rows, height, wrap), what the caller shows for
-- a cell that is not being edited, so that all of them look as in this Neovim: colorscheme, treesitter, the language server's
-- semantic tokens, number column and signs, diagnostics, conceal. Neovim's own `tohtml` (the Lua behind `:TOhtml`) does it on a
-- hidden float that shows the notebook buffer with the notebook window's options and the cursor on the cell's first line (so the
-- numbers read as they do when the cell is entered). The screen rows the cell takes in the notebook window (`height`) and its
-- 'wrap' come along, so the caller gives the render the room the cell takes when it is entered.
-- Its HTML goes to the caller, which turns it into plain data (`lib/neovim_render.py`). Three things
-- `tohtml` does not do for a range are done here: the gutter (its number column of a range lands on the wrong rows), the sign column
-- of an 'auto' 'signcolumn' (a float that was never drawn has none) and the semantic tokens (it skips the extmarks Neovim makes for
-- them, which exist for the lines on screen only): the cell's tokens are marked in a namespace of ours for the render.
-- What a cell was last rendered from is its signature (text, diagnostics, semantic tokens, colorscheme and gutter options, sign
-- column, width, and its first line when the numbers are absolute): a cell whose signature did not change is not rendered again. A
-- render waits for 150 ms without typing or other changes, runs one cell per event loop turn (nearest the active cell first, the
-- active cell itself only for its first render) and gives way to typing. A cell that was just left is rendered at once. A cell whose
-- render failed, and every cell once the buffer grows past `RENDER_MAX_LINES`, is dropped (`cell_render_drop`, ids): the caller
-- shows its own editor there.
-- The triggers are the changes themselves (edits, `set_cells`, diagnostics, tokens, colorscheme, options, size), not a decoration
-- provider's `on_win`: that reports redraws of a window, and the notebook window only ever draws the active cell, never the cells
-- that are rendered.
local RENDER_DEBOUNCE_MS = 150
-- `tohtml` reads the whole buffer for every cell, a notebook of this size is left to the caller's own editor (see `over_limit`)
local RENDER_MAX_LINES = 4000

local function hex(color) return color ~= nil and string.format("#%06x", color) or nil end

-- The style of a highlight group as `neovim_render.py` takes it.
local function group_style(group)
  local hl = vim.api.nvim_get_hl(0, { name = group, link = false })
  local underline = nil
  if hl.undercurl then
    underline = "curl"
  elseif hl.underline or hl.underdouble or hl.underdotted or hl.underdashed then
    underline = "line"
  end
  return {
    fg = hex(hl.fg),
    bg = hex(hl.bg),
    sp = hex(hl.sp),
    bold = hl.bold or nil,
    italic = hl.italic or nil,
    strike = hl.strikethrough or nil,
    underline = underline,
  }
end

-- The sign columns of an 'auto' 'signcolumn' (at most `most`): as many as the line of the buffer with the most signs has, as Neovim
-- counts them for a window. A float that was never drawn shows none, the render's float gets them as a fixed 'signcolumn'.
local function auto_signs(buffer, most)
  local count, widest = {}, 0
  for _, mark in ipairs(vim.api.nvim_buf_get_extmarks(buffer, -1, 0, -1, { type = "sign" })) do
    count[mark[2]] = (count[mark[2]] or 0) + 1
    widest = math.max(widest, count[mark[2]])
  end
  return math.min(widest, most)
end

-- The columns `tohtml` pads the gutter to: the number column, the signs and the fold column of `window`, whose 'signcolumn' is fixed.
local function gutter_width(window)
  local wo = vim.wo[window]
  local width = (wo.number or wo.relativenumber) and wo.numberwidth or 0
  if wo.signcolumn:match("^yes") then width = width + (tonumber(wo.signcolumn:match("(%d)$")) or 1) * 2 end
  return width + (tonumber(wo.foldcolumn) or 0)
end

-- The lines that extmarks (`virt_lines`) put before the row, by row (1-based), within `first` .. `last`.
local function virtual_lines(buffer, first, last)
  local extra = {}
  local options = { type = "virt_lines", details = true }
  local marks = vim.api.nvim_buf_get_extmarks(buffer, -1, { math.max(first - 2, 0), 0 }, { last - 1, -1 }, options)
  for _, mark in ipairs(marks) do
    if mark[4].virt_lines ~= nil then
      local row = mark[2] + (mark[4].virt_lines_above and 1 or 2)
      if row >= first and row <= last then extra[row] = (extra[row] or 0) + #mark[4].virt_lines end
    end
  end
  return extra
end

-- The leading runs `{ text, style }` of every output line of the rows `first` .. `last` (1-based): what the window's
-- 'statuscolumn' shows (signs, fold column, line numbers), blank for a line that `virt_lines` adds (`extra`, see `virtual_lines`).
-- Nil when the window has no such column. `width` is how wide the notebook window draws these columns: Neovim widens a
-- 'statuscolumn' to the widest it drew, which a float that was never drawn did not. Runs inside `window`.
local function gutter_of(window, first, last, width, extra)
  local wo = vim.wo[window]
  local statuscolumn = wo.statuscolumn
  if statuscolumn == "" then
    if wo.relativenumber then
      statuscolumn = wo.number and '%C%s%{%v:lnum!=line(".")?"%=".v:relnum." ":v:lnum%}' or '%C%s%{%"%=".v:relnum." "%}'
    elseif wo.number then
      statuscolumn = '%C%s%{%"%=".v:lnum." "%}'
    elseif wo.signcolumn ~= "no" or wo.foldcolumn ~= "0" then
      statuscolumn = "%C%s"
    else
      return nil
    end
  end
  local statuses, widest = {}, math.max(width, gutter_width(window))
  for row = first, last do
    local options = { winid = window, use_statuscol_lnum = row, highlights = true, maxwidth = widest }
    local status = vim.api.nvim_eval_statusline(statuscolumn, options)
    statuses[row] = status
    widest = math.max(widest, vim.api.nvim_strwidth(status.str))
  end
  local styles, out = {}, {}
  local function style(group)
    if styles[group] == nil then styles[group] = group_style(group) end
    return styles[group]
  end
  for row = first, last do
    for _ = 1, extra[row] or 0 do out[#out + 1] = { { (" "):rep(widest), {} } } end
    local status = statuses[row]
    local pad = (" "):rep(widest - vim.api.nvim_strwidth(status.str))
    local runs, highlights = {}, status.highlights
    if #highlights == 0 then
      runs[1] = { status.str .. pad, {} }
    else
      if highlights[1].start > 0 then runs[1] = { status.str:sub(1, highlights[1].start), {} } end
      for index, highlight in ipairs(highlights) do
        local text = status.str:sub(highlight.start + 1, highlights[index + 1] and highlights[index + 1].start or #status.str)
        if index == #highlights then text = text .. pad end
        runs[#runs + 1] = { text, style(highlight.group) }
      end
    end
    out[#out + 1] = runs
  end
  return out
end

-- The cell holding the 0-based `row`, among the content start rows `starts` (ascending). Nil above the first.
local function find_cell(starts, row)
  local low, high, found = 1, #starts, nil
  while low <= high do
    local middle = math.floor((low + high) / 2)
    if starts[middle] <= row then
      found, low = middle, middle + 1
    else
      high = middle - 1
    end
  end
  return found
end

-- What the render of every cell would be made of, by cell id. Semantic tokens are taken from the last valid result: while a newer
-- text waits for its tokens the cells keep the ones they have.
local function signatures(nb, window)
  local rd = nb.render
  local lines = vim.api.nvim_buf_get_lines(nb.buffer, 0, -1, false)
  local starts, texts = {}, {}
  for index = 1, #nb.order do
    local start, stop = cell_range(nb, index)
    starts[index], texts[index] = start, table.concat(lines, "\n", start + 1, stop)
  end
  local function bucket(buckets, row, piece)
    local index = find_cell(starts, row)
    if index == nil then return end
    local list = buckets[index] or {}
    list[#list + 1] = (row - starts[index]) .. ":" .. piece
    buckets[index] = list
  end
  local diagnostics = {}
  for _, diagnostic in ipairs(vim.diagnostic.get(nb.buffer)) do
    bucket(diagnostics, diagnostic.lnum, table.concat({
      diagnostic.col, diagnostic.end_lnum or "", diagnostic.end_col or "", diagnostic.severity or "",
      tostring(diagnostic.source or ""),
      tostring(diagnostic.code or ""), tostring(diagnostic.message),
    }, ":"))
  end
  pcall(function()
    local highlighter = vim.lsp.semantic_tokens.__STHighlighter.active[nb.buffer]
    local fresh, valid = {}, false
    for _, state in pairs(highlighter and highlighter.client_state or {}) do
      local result = state.current_result
      if result ~= nil and result.highlights ~= nil and result.version == vim.lsp.util.buf_versions[nb.buffer] then
        valid = true
        for _, token in ipairs(result.highlights) do
          local modifiers = {}
          for modifier in pairs(token.modifiers) do modifiers[#modifiers + 1] = modifier end
          table.sort(modifiers)
          local piece = table.concat({ token.start_col, token.end_col, token.type, table.concat(modifiers, ",") }, ":")
          bucket(fresh, token.line, piece)
        end
      end
    end
    if valid then
      rd.semantic = {}
      for index, id in ipairs(nb.order) do
        table.sort(fresh[index] or {})
        rd.semantic[id] = table.concat(fresh[index] or {}, ";")
      end
    end
  end)
  -- The sign column of every cell is as wide as the buffer's most signs on one line need
  local wo = vim.wo[window]
  local auto = wo.signcolumn:match("^auto") and auto_signs(nb.buffer, 9) or wo.signcolumn
  local layout = table.concat({ vim.api.nvim_win_get_width(window), vim.fn.getwininfo(window)[1].textoff, auto }, ":")
  -- An absolute number column ('number', or a 'statuscolumn', which may show `v:lnum`) shows where the cell starts in the buffer:
  -- a line added or removed above it changes its render
  local absolute = wo.number or wo.statuscolumn ~= ""
  local sigs = {}
  for index, id in ipairs(nb.order) do
    table.sort(diagnostics[index] or {})
    local found = table.concat(diagnostics[index] or {}, ";")
    local start = absolute and starts[index] or ""
    sigs[id] = table.concat({ texts[index], found, rd.semantic[id] or "", rd.generation, layout, start }, "\0")
  end
  return sigs
end

-- Runs the Vimscript call `call` under `:noautocmd`, which ignores every autocommand for that one command and puts 'eventignore'
-- back by itself: what the user's autocommands do for windows (`OptionSet`, `WinClosed`) must not happen for the hidden float of a
-- render, and nothing else runs meanwhile with them ignored
local function without_autocmds(call)
  vim.cmd({ cmd = "call", args = { call }, mods = { noautocmd = true } })
end
local semantic_namespace = vim.api.nvim_create_namespace("gigachad_cells_semantic")

-- Marks the semantic tokens of the 0-based rows `first` .. `last` as Neovim does for the lines on screen (`@lsp.type.*` and the
-- modifier groups, the most specific one last) in `semantic_namespace`, from the language server's last result for the current
-- text. True when the buffer has a language server's tokens but none for the current text yet (the render lacks them).
local function mark_tokens(buffer, first, last)
  local ok, highlighter = pcall(function() return vim.lsp.semantic_tokens.__STHighlighter.active[buffer] end)
  if not ok or highlighter == nil then return false end
  local filetype, version, current = vim.bo[buffer].filetype, vim.lsp.util.buf_versions[buffer], false
  for _, state in pairs(highlighter.client_state) do
    local result = state.current_result
    if result ~= nil and result.highlights ~= nil and result.version == version then
      current = true
      for _, token in ipairs(result.highlights) do
        if token.line >= first and token.line <= last then
          local function mark(group)
            local options = { hl_group = group, end_col = token.end_col, strict = false }
            vim.api.nvim_buf_set_extmark(buffer, semantic_namespace, token.line, token.start_col, options)
          end
          mark(("@lsp.type.%s.%s"):format(token.type, filetype))
          for modifier in pairs(token.modifiers) do
            mark(("@lsp.mod.%s.%s"):format(modifier, filetype))
            mark(("@lsp.typemod.%s.%s.%s"):format(token.type, modifier, filetype))
          end
        end
      end
    end
  end
  return not current
end

-- The cell `id` as `html, gutter, base, rows, stale, height, wrap` (see `neovim_render.py`): `rows` holds the cell line (0-based) of
-- every output line, `stale` whether the language server's tokens for the current text were missing, `height` the screen rows the
-- cell takes in `window` (what `cell_height` says once it is the active cell), `wrap` the column a wrapped line goes on at when the
-- window wraps lines (nil when it does not). Raises what went wrong after the window is as it was: the float closed, the lines
-- outside the active cell hidden again, the tokens unmarked.
local function render_cell(nb, id, window)
  local rd = nb.render
  local first, last = cell_range(nb, nb.index[id])
  first = first + 1 -- the 1-based rows `first` .. `last`
  local buffer, width = nb.buffer, vim.api.nvim_win_get_width(window)
  local float = nil
  rd.rendering = true
  local ok, html, gutter, rows, stale, height, wrap = pcall(function()
    -- The lines outside the active cell are hidden in every window that shows the buffer, `pin` hides them again
    vim.api.nvim_buf_clear_namespace(buffer, hidden_namespace, 0, -1)
    nb.hidden = nil
    local marked, missing = pcall(mark_tokens, buffer, first - 1, last - 1)
    float = vim.api.nvim_win_call(window, function()
      return vim.api.nvim_open_win(buffer, false, {
        relative = "editor",
        row = 0,
        col = 0,
        width = width,
        height = math.max(1, math.min(last - first + 1, vim.o.lines - 1)),
        focusable = false,
        hide = true,
        noautocmd = true,
      })
    end)
    local signs = vim.wo[window].signcolumn
    if signs:match("^auto") then
      local count = auto_signs(buffer, tonumber(signs:match("(%d)$")) or 1)
      without_autocmds(("setwinvar(%d, '&signcolumn', '%s')"):format(float, count > 0 and ("yes:" .. count) or "no"))
    end
    vim.api.nvim_win_set_cursor(float, { first, 0 })
    local lines = rd.tohtml.tohtml(float, { number_lines = false, width = width, range = { first, last } })
    local textoff = vim.fn.getwininfo(window)[1].textoff
    local extra, rows = virtual_lines(buffer, first, last), {}
    for row = first, last do
      for _ = 1, (extra[row] or 0) + 1 do rows[#rows + 1] = row - first end
    end
    local gutter = vim.api.nvim_win_call(float, function() return gutter_of(float, first, last, textoff, extra) end)
    local height = vim.api.nvim_win_text_height(window, { start_row = first - 1, end_row = last - 1 }).all
    return table.concat(lines, "\n"), gutter, rows, marked and missing, height, vim.wo[window].wrap and textoff or nil
  end)
  if float ~= nil then pcall(without_autocmds, ("nvim_win_close(%d, v:true)"):format(float)) end
  vim.api.nvim_buf_clear_namespace(buffer, semantic_namespace, 0, -1)
  rd.rendering = false
  pin(nb)
  if not ok then error(html, 0) end
  return html, gutter, group_style("Normal"), rows, stale, height, wrap
end

-- The caller drops its render of the cells `ids` and shows its own editor there.
local function drop(ids)
  vim.rpcnotify(channel, "cell_render_drop", ids)
end

-- Renders the cell `id` and tells the caller, `sig` is what it was made from. A render that failed is not tried again before the cell
-- changes, the caller drops the render it has. A render without the language server's tokens for the current text is made again
-- once they arrive (even when they equal the ones the signature holds). Without a window nothing happens and it is tried again.
local function render_one(nb, id, sig)
  local rd = nb.render
  local window = notebook_window(nb)
  if window == nil then return end
  local ok, html, gutter, base, rows, stale, height, wrap = pcall(render_cell, nb, id, window)
  rd.sigs[id] = ok and stale and sig .. "\0stale-tokens" or sig
  if ok then
    rd.failure = nil
    vim.rpcnotify(channel, "cell_render", id, html, gutter or {}, base, rows, height, wrap or vim.NIL)
  else
    rd.failure = tostring(html)
    drop({ id })
  end
end

-- Whether the cell `id` needs a render: its signature moved, and it is not the active cell unless that has none yet.
local function wanted(nb, id)
  local rd = nb.render
  return nb.index[id] ~= nil and rd.sigs[id] ~= rd.wanted[id] and (id ~= nb.active or rd.sigs[id] == nil)
end

-- Background work on the book `nb` never shows an error in the user's Neovim: what went wrong is kept in `nb.render.failure`.
local function guarded(work)
  return function(nb)
    local ok, failure = pcall(work, nb)
    if not ok and books[nb.kind] == nb then nb.render.running, nb.render.failure = false, tostring(failure) end
  end
end

local run_step
local function render_step(nb)
  if books[nb.kind] ~= nb then return end
  local rd = nb.render
  local id = table.remove(rd.queue, 1)
  if id == nil then
    rd.running = false
    return
  end
  -- Typing went on: what is done stays, the rest waits for the next pause
  if vim.api.nvim_buf_get_changedtick(nb.buffer) ~= rd.tick then
    rd.running, rd.queue = false, {}
    render_later(nb)
    return
  end
  if wanted(nb, id) then render_one(nb, id, rd.wanted[id]) end
  if #rd.queue > 0 then
    vim.defer_fn(function() run_step(nb) end, 1)
  else
    rd.running = false
  end
end
run_step = guarded(render_step)

-- Neovim's `tohtml`, false when this Neovim has none (rendering is off then, the caller keeps its own editor for every cell). A copy
-- of its own whose `vim.notify` says nothing: `tohtml` warns through it (scheduled, after the render) when the terminal cannot tell
-- its colors or a highlight is beyond it, nothing the user asked for. The user's `vim.notify` is never touched (a notifier such as
-- noice.nvim reports one that was put aside), and `:TOhtml` keeps the module `require` gives.
local function load_tohtml(rd)
  if rd.tohtml == nil then
    pcall(vim.cmd, "packadd nvim.tohtml") -- an optional package of Neovim's runtime
    local path = vim.api.nvim_get_runtime_file("lua/tohtml.lua", false)[1]
    local chunk = path ~= nil and loadfile(path) or nil
    local ok, module = false, nil
    if chunk ~= nil then
      local quiet = setmetatable({ notify = function() end }, { __index = vim })
      setfenv(chunk, setmetatable({ vim = quiet }, { __index = _G }))
      ok, module = pcall(chunk, "tohtml")
    end
    if not ok or type(module) ~= "table" then
      rd.disabled = true
      return false
    end
    rd.tohtml = module
  end
  return true
end

-- Whether the buffer is over `RENDER_MAX_LINES`. Going over it drops every render once (the caller's own editor follows the text,
-- a render would stay behind), coming back under it renders every cell again.
local function over_limit(nb)
  local rd = nb.render
  if vim.api.nvim_buf_line_count(nb.buffer) <= RENDER_MAX_LINES then
    rd.capped = false
    return false
  end
  if not rd.capped then
    rd.capped, rd.sigs, rd.queue = true, {}, {}
    drop(vim.list_slice(nb.order))
  end
  return true
end

-- Queues the cells whose signature moved, the ones nearest the active cell first. The active cell, which Neovim's window shows, only
-- gets its first render (last): so that the caller has one for every cell, and a cell that is left never shows its editor meanwhile.
-- Nothing happens while no window shows the buffer (the markdown buffer is in it): the renders follow the window's options.
local function refresh(nb)
  if books[nb.kind] ~= nb or nb.render.disabled or over_limit(nb) then return end
  local rd, window = nb.render, notebook_window(nb)
  if window == nil or not load_tohtml(rd) then return end
  rd.wanted, rd.tick = signatures(nb, window), vim.api.nvim_buf_get_changedtick(nb.buffer)
  local active, queue = nb.index[nb.active] or 0, {}
  for index, id in ipairs(nb.order) do
    if wanted(nb, id) then queue[#queue + 1] = { id = id, distance = id == nb.active and math.huge or math.abs(index - active) } end
  end
  table.sort(queue, function(a, b) return a.distance < b.distance end)
  rd.queue = {}
  for _, item in ipairs(queue) do rd.queue[#rd.queue + 1] = item.id end
  for id in pairs(rd.sigs) do
    if nb.index[id] == nil then rd.sigs[id] = nil end
  end
  if not rd.running and #rd.queue > 0 then
    rd.running = true
    vim.defer_fn(function() run_step(nb) end, 1)
  end
end
local refresh_guarded = guarded(refresh)

-- Only the code book renders, a call for the markdown book does nothing.
render_later = function(nb)
  if nb == nil or nb.render == nil or nb.render.disabled then return end
  nb.render.timer:stop()
  nb.render.timer:start(RENDER_DEBOUNCE_MS, 0, vim.schedule_wrap(function() refresh_guarded(nb) end))
end

-- The cell `id` was just left (or is not rendered yet): its render is brought up to date now, ahead of the frame that says it is left.
render_now = function(nb, id)
  if nb == nil or nb.render == nil or id == nil or nb.index[id] == nil or nb.render.disabled or over_limit(nb) then return end
  local rd, window = nb.render, notebook_window(nb)
  if window == nil or not load_tohtml(rd) then return end
  -- Called from the cursor and focus handling, which must go on whatever the render does
  local ok, failure = pcall(function()
    local sigs = signatures(nb, window)
    if sigs[id] ~= rd.sigs[id] then render_one(nb, id, sigs[id]) end
  end)
  if not ok then rd.failure = tostring(failure) end
end

-- Whether rendering works (a failure is kept here, never shown), for the caller's diagnosis and the tests.
function _G.gigachad_render_state()
  local nb = books.code
  if nb == nil then return nil end
  return { failure = nb.render.failure, disabled = nb.render.disabled == true, running = nb.render.running }
end

-- Something the renders of the code cells are made of changed (only in the code buffer when `buffer` is given)
local function render_changed(buffer)
  local nb = books.code
  if nb ~= nil and (buffer == nil or buffer == nb.buffer) and not nb.render.rendering then render_later(nb) end
end

vim.api.nvim_create_autocmd("DiagnosticChanged", { callback = function(event) render_changed(event.buf) end })
-- The tokens of a result arrive for the whole buffer, they are marked for the visible lines only (one `LspTokenUpdate` per token)
vim.api.nvim_create_autocmd("LspTokenUpdate", { callback = function(event) render_changed(event.buf) end })
vim.api.nvim_create_autocmd("LspRequest", {
  callback = function(event)
    local request = event.data ~= nil and event.data.request or nil
    local tokens = request ~= nil and request.type == "complete" and tostring(request.method):find("semanticTokens", 1, true)
    if tokens then render_changed(event.buf) end
  end,
})
-- What changes every cell: the colorscheme (the app's theme sets 'background', which reloads it) and the options of the number
-- column and signs, unless only another buffer took them (the markdown buffer's filetype settings)
local function restyled()
  if books.code == nil then return end
  books.code.render.generation = books.code.render.generation + 1
  render_changed()
end
vim.api.nvim_create_autocmd("ColorScheme", { callback = restyled })
vim.api.nvim_create_autocmd("OptionSet", {
  pattern = { "number", "relativenumber", "numberwidth", "signcolumn", "statuscolumn", "foldcolumn", "tabstop", "wrap" },
  callback = function()
    if vim.v.option_type == "local" and books.code ~= nil and vim.api.nvim_get_current_buf() ~= books.code.buffer then return end
    restyled()
  end,
})
vim.api.nvim_create_autocmd({ "VimResized", "WinResized" }, { callback = function() render_changed() end })

track(first_path, first_id)
display(first_id)

vim.api.nvim_create_autocmd("BufEnter", {
  callback = function(event)
    local id = ids[event.buf]
    if id == nil or id == current_id then return end
    current_id = id
    vim.rpcnotify(channel, "switched", id)
  end,
})

vim.api.nvim_create_autocmd("BufDelete", {
  callback = function(event)
    local id = ids[event.buf]
    local nb = book_of(event.buf)
    if nb ~= nil then forget_book(nb) end
    if id == nil then return end
    untrack(id)
  end,
})

vim.api.nvim_create_autocmd("TextYankPost", {
  callback = function()
    if vim.v.event.operator ~= "y" then return end
    local text = table.concat(vim.v.event.regcontents, "\n")
    if vim.v.event.regtype == "V" then text = text .. "\n" end
    vim.rpcnotify(channel, "yank", text)
  end,
})

-- `<Esc>` in normal mode hands focus back to the app. The user's mapping of it (read after their config ran) still runs first
local user_escape = vim.fn.maparg("<Esc>", "n", false, true)
run_user_escape = function()
  if vim.tbl_isempty(user_escape) then return end
  local keys = user_escape.rhs or ""
  if user_escape.callback ~= nil then
    local result = user_escape.callback()
    if user_escape.expr ~= 1 then return end
    keys = result or ""
  elseif user_escape.expr == 1 then
    keys = vim.api.nvim_eval(keys)
  end
  vim.api.nvim_feedkeys(vim.api.nvim_replace_termcodes(keys, true, true, true), user_escape.noremap == 1 and "n" or "m", false)
end
vim.keymap.set("n", "<Esc>", function()
  run_user_escape()
  vim.rpcnotify(channel, "leave")
end, { desc = "Give focus back to the app" })

if own_clipboard == true then
  -- What the last copy put in `+` or `*`, pasted when the browser cannot answer so `yy` then `p` works without its clipboard
  local yanked = { { "" }, "v" }

  local function copy(lines, regtype) yanked = { lines, regtype } end

  -- `{ lines, regtype }` for `text`: linewise when it ends with a newline. An empty register would be an error (E353) that
  -- replaces the message shown for a failed paste, one empty line pastes nothing
  local function register_of(text)
    local lines = vim.split(text, "\n", { plain = true })
    if #lines > 1 and lines[#lines] == "" then
      table.remove(lines)
      return { lines, "V" }
    end
    return { lines, "v" }
  end

  local function paste()
    local buffer = vim.api.nvim_get_current_buf()
    local ok, answer = pcall(vim.rpcrequest, channel, "paste", ids[buffer] or vim.api.nvim_buf_get_name(buffer))
    if not ok then answer = { error = tostring(answer) } end
    if answer.error ~= nil then
      _G.gigachad_show_error(answer.error)
      return yanked
    end
    return register_of(answer.text or "")
  end

  vim.g.clipboard = {
    name = "gigachad",
    copy = { ["+"] = copy, ["*"] = copy },
    paste = { ["+"] = paste, ["*"] = paste },
    cache_enabled = 0,
  }
  -- Neovim resolves the provider once, its script has to load again to take the new `vim.g.clipboard`
  vim.g.loaded_clipboard_provider = nil
  vim.cmd("runtime autoload/provider/clipboard.vim")
end

if base_path then
  local document_window = vim.api.nvim_get_current_win()
  vim.cmd("leftabove vertical split " .. vim.fn.fnameescape(base_path))
  local base_buffer = vim.api.nvim_get_current_buf()
  vim.bo[base_buffer].readonly = true
  vim.bo[base_buffer].modifiable = false
  vim.api.nvim_create_autocmd("BufWriteCmd", {
    buffer = base_buffer,
    callback = function() error("Baseline is read-only", 0) end,
  })
  vim.cmd("diffthis")
  vim.api.nvim_set_current_win(document_window)
  vim.cmd("diffthis")
end
"""


class NeovimRpcError(RuntimeError):
    """Neovim answered a request with an error."""


class NeovimRpc:
    """Minimal msgpack-RPC client over Neovim's `--listen` socket."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        handle_request: Callable[[str, list], Awaitable[object]],
        handle_notification: Callable[[str, list], None],
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._handle_request = handle_request
        self._handle_notification = handle_notification
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._answers: set[asyncio.Task] = set()
        self._read_task = asyncio.create_task(self._read_messages())

    async def request(self, method: str, *params: object) -> object:
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[self._next_id] = future
        self._send([0, self._next_id, method, list(params)])
        return await future

    async def close(self) -> None:
        self._read_task.cancel()
        for task in (self._read_task, *self._answers):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._writer.close()

    def _send(self, message: list) -> None:
        self._writer.write(msgpack.packb(message))

    async def _read_messages(self) -> None:
        unpacker = msgpack.Unpacker(raw=False, unicode_errors="replace")
        try:
            while data := await self._reader.read(_READ_CHUNK):
                unpacker.feed(data)
                for message in unpacker:
                    self._dispatch(message)
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ConnectionError("Neovim RPC connection closed"))

    def _dispatch(self, message: list) -> None:
        """Resolve responses, answer requests, forward notifications."""
        if message[0] == 1:
            _, message_id, error, result = message
            future = self._pending.pop(message_id, None)
            if future is None or future.done():
                return
            if error:
                future.set_exception(NeovimRpcError(str(error)))
            else:
                future.set_result(result)
        elif message[0] == 0:
            _, message_id, method, params = message
            task = asyncio.create_task(self._answer(message_id, method, params))
            self._answers.add(task)
            task.add_done_callback(self._answers.discard)
        elif message[0] == 2:
            _, method, params = message
            self._handle_notification(method, params)

    async def _answer(self, message_id: int, method: str, params: list) -> None:
        try:
            result = await self._handle_request(method, params)
        except Exception as exc:
            self._send([1, message_id, str(exc), None])
        else:
            self._send([1, message_id, None, result])
