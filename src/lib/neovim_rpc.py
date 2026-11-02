"""Msgpack-RPC with a running Neovim, plus the Lua that bridges its `:w`, yanks and selections back to the caller.

- `SETUP_LUA` tracks the first document's buffer, `_G.gigachad_open` any further one (a private copy each). Documents
  are told apart by an id, the document's path. Every tracked buffer stays loaded (`bufhidden=hide`) when another is shown
- Each tracked buffer's `:w` becomes a `save` request carrying the id instead of writing the disk, and echoes
  `Saved, merged with the agent's edits` when the answer says edits made elsewhere changed the text
- A `BufEnter` autocmd sends a `switched` notification with the id when the window shows another tracked buffer, except for
  the switch `_G.gigachad_open` and `_G.gigachad_show` made. A `BufDelete` autocmd forgets the buffer
- A `TextYankPost` autocmd sends a `yank` notification with the yanked text
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
- With `cell_session` the first document is an empty placeholder. `_G.gigachad_set_cells` replaces it with the notebook buffer:
  one buffer that holds all code cells as one script, each cell led by a marker line (`# %%`). Its name is a virtual path
  beside the notebook when the caller gives one (never written: `BufWriteCmd` and `FileWriteCmd` are intercepted, no swap or undo
  file, the buffer keeps an empty `buftype` or `vim.fs.root` would not use its name), else a private copy. It tracks the cells by
  extmarks on their marker lines, keyed by cell
  id. Changes that alter, remove or add a marker line are undone (the snapshot of the last intact text is put back) with a short
  echo, a cell that lost all its lines gets an empty one. Lines put above the first marker (LSP auto-import) move into the first
  cell. The cursor never rests on a marker line, it goes on into the neighbor cell. The window is pinned to the active cell
  (`scrolloff=0`, top line the cell's first line, the lines outside it hidden by `conceal_lines` extmarks, so `conceallevel`
  is raised to 2, Neovim 0.11+) on `CursorMoved*` and `WinScrolled`, and a cursor that leaves the active cell activates the cell
  it lands in. `_G.gigachad_focus_cell` activates a cell from outside. A rebuild (cells added, removed or moved) drops the undo
  history, since undoing a step from before it would remove a marker line
- `more` is off and `messagesopt` makes messages time out after the user's config ran, a hit-enter prompt would stop every request
- The notebook buffer sends notifications: `cell_edit` (id, text) per changed cell 40 ms after typing (only when its text
  moved past what the caller knows), `cell_focus` (id) when the cursor crossed into another cell, `cell_run` (id, text) from
  `<S-CR>` and `save_notebook` from `:w`. The last two flush pending `cell_edit`s first
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

-- Renames the buffer for a file that was renamed, `:w` and the saved message follow the new name.
function _G.gigachad_rename(id, new_id, new_path)
  local buffer = buffer_of(id)
  local modified = vim.bo[buffer].modified
  -- A buffer left under that name by an earlier rename would refuse it (E95)
  local named
  for _, other in ipairs(vim.api.nvim_list_bufs()) do
    if vim.api.nvim_buf_get_name(other) == new_path then named = other end
  end
  if named ~= buffer then
    if named ~= nil then vim.api.nvim_buf_delete(named, { force = true }) end
    vim.api.nvim_buf_set_name(buffer, new_path)
    vim.bo[buffer].modified = modified
  end
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
  if ask_keymap == true then
    vim.keymap.set("x", "<leader>a", function()
      local first, last = vim.fn.line("v"), vim.fn.line(".")
      if first > last then first, last = last, first end
      vim.cmd("normal! \27")
      local lines = vim.api.nvim_buf_get_lines(buffer, first - 1, last, false)
      local selection = { path = ids[buffer], start_line = first, end_line = last, text = table.concat(lines, "\n") }
      vim.rpcnotify(channel, "selection", selection)
    end, { buffer = buffer })
  end
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

-- The notebook buffer: all code cells of the notebook as one script, each led by a marker line. It is an unlisted buffer on a
-- file name (virtual beside the notebook, or a private copy: filetype detection, the user's plugins and LSP see one file) and
-- never writes it. The caller holds the
-- notebook, `nb` is `{ buffer, marker, order, index, marks, known, lookalikes, snapshot, tick, active, timer, busy }`:
--   order      the cell ids in notebook order, `index` maps an id to its position
--   marks      the extmark (`cells_namespace`) of every cell's marker line, how a cell is found: never by counting marker text
--   known      the text of every cell as the caller last knew it, what `cell_edit` is measured against
--   lookalikes lines inside the cells' own text that read like a marker and are none
--   snapshot   `{ lines, rows }` the last text whose marker lines were intact, put back when a change breaks them
--   tick       the change tick `snapshot` belongs to, `active` the cell the window is pinned to, `busy` while this code edits
-- Changes of the cells and moves into another cell leave as `cell_edit`, `cell_focus`, `cell_run` and `save_notebook`
local uv = vim.uv or vim.loop
local EDIT_DEBOUNCE_MS = 40
local cells_namespace = vim.api.nvim_create_namespace("gigachad_cells")
local hidden_namespace = vim.api.nvim_create_namespace("gigachad_cells_hidden")
-- With `cell_session` the first document is an empty placeholder that the notebook buffer replaces
local placeholder_id = cell_session == true and first_id or nil
local nb = nil

local function forget_notebook()
  if nb == nil then return end
  nb.timer:stop()
  nb.timer:close()
  nb = nil
end

local function notebook_window()
  local current = vim.api.nvim_get_current_win()
  if vim.api.nvim_win_get_buf(current) == nb.buffer then return current end
  local window = vim.fn.bufwinid(nb.buffer)
  return window ~= -1 and window or nil
end

-- The row (0-based) of a cell's marker line.
local function mark_row(id)
  return vim.api.nvim_buf_get_extmark_by_id(nb.buffer, cells_namespace, nb.marks[id], {})[1]
end

-- The rows [first content row, row after the last) of the cell at `index`, 0-based.
local function cell_range(index)
  local next_id = nb.order[index + 1]
  local stop = next_id ~= nil and mark_row(next_id) or vim.api.nvim_buf_line_count(nb.buffer)
  return mark_row(nb.order[index]) + 1, stop
end

-- The cell holding the 0-based `row`, and whether the row is its marker line.
local function locate(row)
  local low, high, found = 1, #nb.order, nil
  while low <= high do
    local middle = math.floor((low + high) / 2)
    if mark_row(nb.order[middle]) <= row then
      found, low = middle, middle + 1
    else
      high = middle - 1
    end
  end
  return found, found ~= nil and mark_row(nb.order[found]) == row
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

local function marker_count(lines)
  local count = 0
  for _, line in ipairs(lines) do
    if line == nb.marker then count = count + 1 end
  end
  return count
end

-- The marker rows in cell order when every marker line is where its extmark says and no other line reads as one, else nil.
local function intact(lines)
  local rows, previous = {}, -1
  for index, id in ipairs(nb.order) do
    local row = mark_row(id)
    if row == nil or row <= previous or lines[row + 1] ~= nb.marker then return nil end
    rows[index], previous = row, row
  end
  if #rows > 0 and rows[1] ~= 0 then return nil end
  if marker_count(lines) ~= #rows + nb.lookalikes then return nil end
  return rows
end

-- A cell has at least one line, as an empty buffer does: gives one to every cell that lost all of its lines. A cursor the
-- deleted line left on the marker below goes onto the new line.
local function fill_empty_cells(lines, rows)
  local filled = false
  local window = notebook_window()
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

-- Edits made by this code, they are no change to report or to check.
local function quietly(work)
  nb.busy = true
  local ok, failure = pcall(work)
  nb.busy = false
  if not ok then error(failure, 0) end
end

-- Lines above the first marker (an LSP quick fix that adds an import, `:0put`) join the first cell as its first lines: the marker
-- line moves back to the top. False when something else is off. Joins the undo step of the change that put them there.
local function adopt_head(lines)
  local first = nb.order[1] ~= nil and mark_row(nb.order[1]) or nil
  if first == nil or first <= 0 or lines[first + 1] ~= nb.marker then return false end
  for row = 1, first do
    if lines[row] == nb.marker then return false end
  end
  local previous = first
  for index = 2, #nb.order do
    local row = mark_row(nb.order[index])
    if row == nil or row <= previous or lines[row + 1] ~= nb.marker then return false end
    previous = row
  end
  local moved = vim.list_slice(lines, 1, first)
  table.insert(moved, 1, nb.marker)
  quietly(function()
    pcall(vim.cmd, "undojoin")
    vim.api.nvim_buf_set_lines(nb.buffer, 0, first + 1, false, moved)
    vim.api.nvim_buf_set_extmark(nb.buffer, cells_namespace, 0, 0, { id = nb.marks[nb.order[1]] })
  end)
  return true
end

-- Takes the buffer as the new snapshot when its marker lines are intact. False when they are not.
local function settle()
  local lines = vim.api.nvim_buf_get_lines(nb.buffer, 0, -1, false)
  local rows = intact(lines)
  if rows == nil and adopt_head(lines) then
    lines = vim.api.nvim_buf_get_lines(nb.buffer, 0, -1, false)
    rows = intact(lines)
  end
  if rows == nil then return false end
  if fill_empty_cells(lines, rows) then
    lines = vim.api.nvim_buf_get_lines(nb.buffer, 0, -1, false)
    rows = intact(lines)
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
local function restore()
  local snapshot = nb.snapshot
  quietly(function()
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
  local window = notebook_window()
  if window ~= nil and nb.cursor ~= nil then
    local row = math.min(nb.cursor[1], vim.api.nvim_buf_line_count(nb.buffer))
    local line = vim.api.nvim_buf_get_lines(nb.buffer, row - 1, row, false)[1]
    vim.api.nvim_win_set_cursor(window, { row, math.min(nb.cursor[2], #line) })
  end
end

-- Checks a change made since the snapshot, false when it was idle or fine, true when it broke a marker line and was undone.
local function sync()
  if nb == nil or nb.busy or vim.api.nvim_buf_get_changedtick(nb.buffer) == nb.tick then return false end
  if settle() then return false end
  restore()
  return true
end

-- Tells the caller the cells whose text moved past what it knows, in notebook order.
local function flush_edits()
  if nb == nil then return end
  nb.timer:stop()
  sync()
  local lines, rows = nb.snapshot.lines, nb.snapshot.rows
  for index, id in ipairs(nb.order) do
    local text = table.concat(lines, "\n", rows[index] + 2, rows[index + 1] or #lines)
    if text ~= nb.known[id] then
      nb.known[id] = text
      vim.rpcnotify(channel, "cell_edit", id, text)
    end
  end
end

-- Pins the window to the active cell: the top line is its first line (when it fits), everything outside it is hidden.
local function pin()
  local window = notebook_window()
  local index = window ~= nil and nb.index[nb.active] or nil
  if index == nil then return end
  local start, stop = cell_range(index)
  local total = vim.api.nvim_buf_line_count(nb.buffer)
  -- Setting an option, even to its value, makes the cursor forget the column it wants
  if vim.wo[window].scrolloff ~= 0 then vim.wo[window].scrolloff = 0 end
  if vim.wo[window].conceallevel < 2 then vim.wo[window].conceallevel = 2 end -- lines are only hidden from level 2
  vim.api.nvim_win_call(window, function()
    local view = vim.fn.winsaveview()
    local height = vim.api.nvim_win_text_height(window, { start_row = start, end_row = stop - 1 }).all
    local fits = height <= vim.api.nvim_win_get_height(window)
    -- A cell taller than the window scrolls with its cursor, but never above its first line
    if view.topline < start + 1 or (fits and view.topline ~= start + 1) then vim.fn.winrestview({ topline = start + 1 }) end
  end)
  -- Both ranges hold whole lines, the end row included whatever the end column. Moving only the window leaves them as they are
  local key = table.concat({ start, stop, total, vim.api.nvim_buf_get_changedtick(nb.buffer) }, ":")
  if key == nb.hidden then return end
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

-- Makes `id` the active cell and puts the cursor in it: "first" or "last" line, "keep" it where it is when that is inside,
-- or `{ offset, col }` the line offset (0-based) and column to land on, clamped to the cell. Pins the window.
local function show_cell(id, cursor)
  local start, stop = cell_range(nb.index[id])
  nb.active = id
  local window = notebook_window()
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
  pin()
  if window ~= nil then nb.cursor = vim.api.nvim_win_get_cursor(window) end
end

-- Keeps the cursor off marker lines, activates the cell it is in and pins the window to it. Moving onto a marker goes on
-- into the neighbor: down into the first line of the next cell, up into the last line of the previous one (the first
-- cell's own first line when there is none). A jump onto a marker lands on the first line of that cell.
local function follow_cursor()
  if nb == nil or nb.busy or #nb.order == 0 then return end
  local window = vim.api.nvim_get_current_win()
  if vim.api.nvim_win_get_buf(window) ~= nb.buffer then return end
  sync()
  local view = vim.fn.winsaveview()
  local index, on_marker = locate(view.lnum - 1)
  if index == nil then return end
  if on_marker then
    local leaving_up = index == nb.index[nb.active] and index > 1
    local target = leaving_up and index - 1 or index
    local start, stop = cell_range(target)
    local row = leaving_up and stop or start + 1
    local line = vim.api.nvim_buf_get_lines(nb.buffer, row - 1, row, false)[1]
    local insert = vim.fn.mode():sub(1, 1) == "i"
    local col = math.min(view.curswant, math.max(#line - (insert and 0 or 1), 0))
    vim.fn.winrestview({ lnum = row, col = col, curswant = view.curswant })
    index = target
  end
  local id = nb.order[index]
  if id ~= nb.active then
    flush_edits()
    nb.active = id
    vim.rpcnotify(channel, "cell_focus", id)
  end
  pin()
  nb.cursor = vim.api.nvim_win_get_cursor(window)
end

local function on_change()
  if nb == nil then return end
  if sync() then follow_cursor() end
  if not nb.timer:is_active() then nb.timer:start(EDIT_DEBOUNCE_MS, 0, vim.schedule_wrap(flush_edits)) end
end

local function install_notebook(buffer)
  vim.api.nvim_create_autocmd({ "TextChanged", "TextChangedI", "TextChangedP" }, { buffer = buffer, callback = on_change })
  -- `:w` asks the caller to save the notebook, after the last edit
  vim.api.nvim_create_autocmd("BufWriteCmd", {
    buffer = buffer,
    callback = function()
      flush_edits()
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
  vim.keymap.set({ "n", "i" }, "<S-CR>", function()
    flush_edits()
    if nb.active ~= nil then vim.rpcnotify(channel, "cell_run", nb.active, nb.known[nb.active]) end
  end, { buffer = buffer })
end

vim.api.nvim_create_autocmd({ "CursorMoved", "CursorMovedI", "WinScrolled" }, { callback = follow_cursor })
-- A scroll that `WinScrolled` did not report (a second one right after the pin) is caught when Neovim goes idle
vim.api.nvim_create_autocmd("SafeState", {
  callback = function()
    follow_cursor()
  end,
})

-- Whether the notebook buffer exists, forgetting one that Neovim unloaded.
function _G.gigachad_has_notebook()
  if nb == nil then return false end
  if vim.api.nvim_buf_is_loaded(nb.buffer) then return true end
  local buffer = nb.buffer
  forget_notebook()
  if vim.api.nvim_buf_is_valid(buffer) then vim.api.nvim_buf_delete(buffer, { force = true }) end
  return false
end

-- Shows the private copy as the notebook buffer, replacing the placeholder.
local function create_notebook(private_path, marker, script)
  local buffer = vim.fn.bufadd(private_path)
  local virtual = script ~= nil and script ~= vim.NIL
  if virtual then
    -- The name only exists for the LSP (root markers, `.venv`): nothing is ever written there, no swap or undo file either
    vim.bo[buffer].swapfile = false
    vim.bo[buffer].undofile = false
  end
  vim.api.nvim_win_set_buf(0, buffer)
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
  nb = {
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
  }
  install_notebook(buffer)
  if placeholder_id ~= nil then
    local placeholder = tracked[placeholder_id]
    placeholder_id = nil
    if placeholder ~= nil and vim.api.nvim_buf_is_valid(placeholder) then vim.api.nvim_buf_delete(placeholder, { force = true }) end
  end
end

-- Builds the script of `cells` from scratch (a changed list of cells). The cursor stays on its line of its cell when that
-- cell is still there, and so does the active cell. A cell the caller still sends as it was told it (`known`) while its
-- text moved on in Neovim keeps the newer text, `cell_edit` brings it to the caller.
local function rebuild(cells)
  local old = nb.snapshot
  local texts = {}
  for index, id in ipairs(nb.order) do
    texts[id] = table.concat(old.lines, "\n", old.rows[index] + 2, old.rows[index + 1] or #old.lines)
  end
  local previous = nb.index[nb.active]
  local window = notebook_window()
  local cursor
  if window ~= nil and #nb.order > 0 then
    local position = vim.api.nvim_win_get_cursor(window)
    local index = locate(position[1] - 1)
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
  quietly(function()
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
  nb.lookalikes = marker_count(lines) - #cells
  if not settle() then
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
    show_cell(active, cursor ~= nil and cursor.id == active and cursor or "first")
  end
end

-- Makes the notebook buffer hold `cells`, a list of `{ id, text }` in notebook order, creating it first when `private_path`
-- is given. The same ids in the same order only replace the text of cells that differ (minimal hunks, the cursor stays).
-- No `cell_edit` comes back for what the caller sent.
function _G.gigachad_set_cells(private_path, marker, cells, script)
  if private_path ~= nil and private_path ~= vim.NIL then create_notebook(private_path, marker, script) end
  if nb == nil then error("No notebook buffer", 0) end
  sync()
  local same = #cells == #nb.order
  for index, cell in ipairs(cells) do
    same = same and cell.id == nb.order[index]
  end
  if same then
    for index, cell in ipairs(cells) do
      local start, stop = cell_range(index)
      local current = table.concat(vim.api.nvim_buf_get_lines(nb.buffer, start, stop, false), "\n")
      -- Text the caller got from this buffer earlier is no news: what was typed since stays
      if cell.text ~= current and cell.text ~= nb.known[cell.id] then
        quietly(function() apply_lines(nb.buffer, cell_lines(cell.text), false, start, stop) end)
      end
      nb.known[cell.id] = cell.text
    end
    nb.lookalikes = marker_count(vim.api.nvim_buf_get_lines(nb.buffer, 0, -1, false)) - #cells
    if settle() then
      if nb.active == nil and #cells > 0 then show_cell(cells[1].id, "first") end
      pin()
      return true
    end
  end
  rebuild(cells)
  return true
end

-- Activates the cell `id`, `cursor` is "first", "last" or "keep". False for a cell that is not there. No `cell_focus` comes back.
function _G.gigachad_focus_cell(id, cursor)
  if nb == nil or nb.index[id] == nil then return false end
  sync()
  if nb.active ~= id then flush_edits() end
  show_cell(id, cursor)
  return true
end

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
    if nb ~= nil and event.buf == nb.buffer then forget_notebook() end
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
