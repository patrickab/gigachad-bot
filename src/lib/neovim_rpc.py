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
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import msgpack

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_READ_CHUNK = 65536

# Receives (channel, first document's private path, its id, optional baseline path, ask_keymap, own_clipboard). Mirrors
# Neovim's write rules: empty, otherwise a final newline unless both 'eol' and 'fixeol' are off.
SETUP_LUA = r"""
local channel, first_path, first_id, base_path, ask_keymap, own_clipboard = ...
if base_path == vim.NIL then base_path = nil end

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

-- Rewrites only the changed line ranges, bottom-up, so cursor, view and marks stay on their text.
local function apply_text(buffer, text)
  local old_lines = vim.api.nvim_buf_get_lines(buffer, 0, -1, false)
  local new_lines = stored_lines(buffer, text)
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
