"""Msgpack-RPC with a running Neovim, plus the Lua that bridges its `:w`, yanks and selections back to the caller.

- `SETUP_LUA` tracks the first document's buffer, `_G.gigachad_open` any further one (a private copy each). Documents
  are told apart by an id, the document's path. Every tracked buffer stays loaded (`bufhidden=hide`) when another is shown
- Each tracked buffer's `:w` becomes a `save` request carrying the id instead of writing the disk, and echoes
  `Saved, merged with the agent's edits` when the answer says edits made elsewhere changed the text
- A `BufEnter` autocmd sends a `switched` notification with the id when the window shows another tracked buffer, except for
  the switch `_G.gigachad_open` and `_G.gigachad_show` made. A `BufDelete` autocmd forgets the buffer
- A `TextYankPost` autocmd sends a `yank` notification with the yanked text
- Normal-mode `<Esc>` sends a `leave` notification (the browser gives focus back to the app), after running the user's own
  normal-mode `<Esc>` mapping (`:noh`)
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
- With `notebook` the first document is a notebook's percent text (`# %%` before each cell): its buffer is named by its id, a
  virtual path beside the notebook (`<notebook>.ipynb.py`), so the user's LSP finds the notebook's project and `.venv`. Nothing is
  ever read from or written to that name: `BufReadCmd` loads the text the caller last agreed on (a `base` request, also for `:e!`),
  `BufWriteCmd` saves through the caller, `FileWriteCmd`/`FileAppendCmd` (`:1,2w`) and `FileReadCmd` of that name (`:r`) are
  refused, no swap or undo file. `:sav` or `:file` gets the virtual name back and the buffer left under it is wiped (`:e #`). The
  buffer keeps an empty `buftype`, else `vim.fs.root` would not use its name. `<C-CR>` saves (`:w`) and on success sends a `run_cell`
  notification with the id and the cursor line, `<S-CR>` then moves the cursor to the next cell's first line, appending a cell
  after the last one
- `more` is off and `messagesopt` makes messages time out after the user's config ran, a hit-enter prompt would stop every request
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import msgpack

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_READ_CHUNK = 65536

# Receives (channel, first document's buffer name, its id, optional baseline path, ask_keymap, own_clipboard, notebook). Mirrors
# Neovim's write rules: empty, otherwise a final newline unless both 'eol' and 'fixeol' are off.
SETUP_LUA = r"""
local channel, first_path, first_id, base_path, ask_keymap, own_clipboard, notebook = ...
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

-- Virtual names of the notebook texts' buffers, the name each one keeps, and the autocmd refusing `:r` of it. `:r` fires no
-- autocmd of the buffer it reads into, so reads are refused by name, from any buffer.
local virtual_names = {}
local read_refusals = {}

local function set_virtual_name(buffer, name)
  virtual_names[buffer] = name
  if read_refusals[buffer] ~= nil then vim.api.nvim_del_autocmd(read_refusals[buffer]) end
  read_refusals[buffer] = nil
  if name == nil then return end
  read_refusals[buffer] = vim.api.nvim_create_autocmd("FileReadCmd", {
    pattern = vim.fn.escape(name, [[\*?[]{},]]),
    callback = function() vim.api.nvim_echo({ { "The notebook text has no file to read" } }, true, {}) end,
  })
end

local function untrack(id)
  local buffer = tracked[id]
  tracked[id] = nil
  if buffer ~= nil then
    ids[buffer] = nil
    set_virtual_name(buffer, nil)
  end
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

-- Rewrites only the changed line ranges, bottom-up, so cursor, view and marks stay on their text.
local function apply_text(buffer, text)
  local new_lines = stored_lines(buffer, text)
  local old_lines = vim.api.nvim_buf_get_lines(buffer, 0, -1, false)
  local hunks = diff_lines(table.concat(old_lines, "\n") .. "\n", table.concat(new_lines, "\n") .. "\n", { result_type = "indices" })
  vim.api.nvim_buf_clear_namespace(buffer, remote_edit_namespace, 0, -1)
  for index = #hunks, 1, -1 do
    local start_old, count_old, start_new, count_new = unpack(hunks[index])
    local first = count_old > 0 and start_old - 1 or start_old
    local replacement = vim.list_slice(new_lines, start_new, start_new + count_new - 1)
    vim.api.nvim_buf_set_lines(buffer, first, first + count_old, false, replacement)
    highlight_remote_edit(buffer, first, count_new)
  end
end

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

-- Wipes the buffers other than `keep` named `name`.
local function wipe_named(name, keep)
  for _, other in ipairs(vim.api.nvim_list_bufs()) do
    if other ~= keep and vim.api.nvim_buf_get_name(other) == name then vim.api.nvim_buf_delete(other, { force = true }) end
  end
end

-- Names `buffer` `new_path`, keeping its modified flag. A buffer left under that name by an earlier rename would refuse it (E95),
-- the unloaded one Neovim leaves under the old name goes too, so `:e #` reads no file there.
local function rename_buffer(buffer, new_path)
  local modified = vim.bo[buffer].modified
  local old_path = vim.api.nvim_buf_get_name(buffer)
  if old_path == new_path then return end
  wipe_named(new_path, buffer)
  vim.api.nvim_buf_set_name(buffer, new_path)
  wipe_named(old_path, buffer)
  vim.bo[buffer].modified = modified
end

-- Renames the buffer for a file that was renamed, `:w` and the saved message follow the new name.
function _G.gigachad_rename(id, new_id, new_path)
  local buffer = buffer_of(id)
  if virtual_names[buffer] ~= nil then set_virtual_name(buffer, new_path) end
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

-- With `ask_keymap`, visual-mode `<leader>a` in `buffer` tells the caller the selected lines (`selection`).
local function install_ask(buffer)
  if ask_keymap ~= true then return end
  vim.keymap.set("x", "<leader>a", function()
    local first, last = vim.fn.line("v"), vim.fn.line(".")
    if first > last then first, last = last, first end
    vim.cmd("normal! \27")
    local lines = vim.api.nvim_buf_get_lines(buffer, first - 1, last, false)
    local selection = { path = ids[buffer], start_line = first, end_line = last, text = table.concat(lines, "\n") }
    vim.rpcnotify(channel, "selection", selection)
  end, { buffer = buffer })
end

-- Set by a `:w` the caller stored, so `<C-CR>` runs a cell only after a save that went through
local saved = false

-- `# %%`, alone or followed by whitespace and a title or kind
local function is_marker(line) return line == "# %%" or line:match("^# %%%%%s") ~= nil end

-- Saves the notebook text (`:w`), then tells the caller to run the cell at the cursor. With `advance` the cursor moves on to the
-- first line of the next cell, a new cell is appended after the last one.
local function run_cell(buffer, advance)
  saved = false
  local ok, failure = pcall(vim.cmd.write)
  if not saved then
    if not ok then vim.api.nvim_echo({ { tostring(failure), "ErrorMsg" } }, true, {}) end
    return
  end
  local window = vim.api.nvim_get_current_win()
  local row = vim.api.nvim_win_get_cursor(window)[1]
  vim.rpcnotify(channel, "run_cell", ids[buffer], row)
  if not advance then return end
  for offset, line in ipairs(vim.api.nvim_buf_get_lines(buffer, row, -1, false)) do
    if is_marker(line) then
      local target = row + offset + 1
      if target > vim.api.nvim_buf_line_count(buffer) then vim.api.nvim_buf_set_lines(buffer, -1, -1, false, { "" }) end
      vim.api.nvim_win_set_cursor(window, { target, 0 })
      return
    end
  end
  vim.api.nvim_buf_set_lines(buffer, -1, -1, false, { "", "# %%", "" })
  vim.api.nvim_win_set_cursor(window, { vim.api.nvim_buf_line_count(buffer), 0 })
end

-- A notebook's text under the virtual name `name`: never read from (`:r` of it is refused) or written to it, no swap or undo
-- file. A rename the caller did not ask for (`:sav`, `:file`) is undone once the command finished. The empty `buftype` stays,
-- else `vim.fs.root` would ignore the name and the LSP would find no project.
local function virtualize(buffer, name)
  set_virtual_name(buffer, name)
  vim.bo[buffer].swapfile = false
  vim.bo[buffer].undofile = false
  local loaded = false
  vim.api.nvim_create_autocmd("BufReadCmd", {
    buffer = buffer,
    callback = function()
      local ok, text = pcall(vim.rpcrequest, channel, "base", ids[buffer])
      if not ok then error("Reload failed: " .. tostring(text), 0) end
      local levels = vim.bo[buffer].undolevels -- the global one stays in charge unless the buffer had its own
      if not loaded then vim.bo[buffer].undolevels = -1 end -- the first text is no undo step
      vim.api.nvim_buf_set_lines(buffer, 0, -1, false, stored_lines(buffer, text))
      vim.bo[buffer].undolevels = levels
      vim.bo[buffer].modified = false
      loaded = true
      vim.api.nvim_buf_call(buffer, function() pcall(vim.cmd, "filetype detect") end)
    end,
  })
  vim.api.nvim_create_autocmd({ "FileWriteCmd", "FileAppendCmd" }, {
    buffer = buffer,
    callback = function() vim.api.nvim_echo({ { "The notebook is only written whole, with :w" } }, true, {}) end,
  })
  vim.api.nvim_create_autocmd("BufFilePost", {
    buffer = buffer,
    callback = function()
      vim.schedule(function()
        if virtual_names[buffer] ~= nil and vim.api.nvim_buf_is_valid(buffer) then rename_buffer(buffer, virtual_names[buffer]) end
      end)
    end,
  })
  for key, advance in pairs({ ["<S-CR>"] = true, ["<C-CR>"] = false }) do
    vim.keymap.set({ "n", "i" }, key, function() run_cell(buffer, advance) end, { buffer = buffer })
  end
end

-- `save` answers { text = stored text, merged = whether edits made elsewhere changed the buffer's text }. A `virtual` buffer is a
-- notebook's text named `name`, the others are private copies.
local function track(name, id, virtual)
  if tracked[id] ~= nil then untrack(id) end
  local buffer = vim.fn.bufadd(name)
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
      saved = true
      local name = vim.fn.fnamemodify(vim.api.nvim_buf_get_name(buffer), ":t")
      local message = stored.merged and "Saved, merged with the agent's edits" or "Saved " .. name
      vim.api.nvim_echo({ { message } }, false, {})
    end,
  })
  install_ask(buffer)
  if virtual then virtualize(buffer, name) end
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

if notebook == true then
  local empty = vim.api.nvim_get_current_buf() -- the empty placeholder file Neovim started on
  track(first_path, first_id, true)
  display(first_id)
  if vim.api.nvim_buf_is_valid(empty) and empty ~= tracked[first_id] then vim.api.nvim_buf_delete(empty, { force = true }) end
else
  track(first_path, first_id)
  display(first_id)
end

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
local function run_user_escape()
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
