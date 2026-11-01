"""Msgpack-RPC with a running Neovim, plus the Lua that bridges its `:w`, yanks and selections back to the caller.

- `SETUP_LUA` routes the document buffer's `:w` to a `save` request instead of the disk, and echoes
  `Saved, merged with the agent's edits` when the answer says edits made elsewhere changed the text
- A `TextYankPost` autocmd sends a `yank` notification with the yanked text
- With `ask_keymap`, visual-mode `<leader>a` sends a `selection` notification with the selected lines
- `_G.gigachad_buffer_state` reports the buffer text, modified flag and change tick
- `_G.gigachad_set_text` rewrites only the changed hunks, for edits made to the file elsewhere, keeping the modified flag
  (or setting it when the text differs from the stored file)
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import msgpack

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_READ_CHUNK = 65536

# Receives (channel, working path, optional baseline path, ask_keymap). Mirrors Neovim's write rules:
# empty, otherwise a final newline unless both 'eol' and 'fixeol' are off.
SETUP_LUA = r"""
local channel, path, base_path, ask_keymap = ...
if base_path == vim.NIL then base_path = nil end
local buffer = vim.fn.bufadd(path)
vim.fn.bufload(buffer)
vim.api.nvim_win_set_buf(0, buffer)

local function buffer_text()
  local lines = vim.api.nvim_buf_get_lines(buffer, 0, -1, false)
  if #lines == 1 and lines[1] == "" then return "" end
  local text = table.concat(lines, "\n")
  if vim.bo[buffer].eol or vim.bo[buffer].fixeol then text = text .. "\n" end
  return text
end

local function stored_lines(text)
  local lines = vim.split(text, "\n", { plain = true })
  if #lines > 1 and lines[#lines] == "" then table.remove(lines) else vim.bo[buffer].eol = false end
  return lines
end

local diff_lines = (vim.text and vim.text.diff) or vim.diff
local remote_edit_namespace = vim.api.nvim_create_namespace("gigachad_remote_edit")
local remote_edit_generation = 0
local REMOTE_EDIT_HIGHLIGHT_MS = 1500

local function highlight_remote_edit(first, count)
  remote_edit_generation = remote_edit_generation + 1
  local generation = remote_edit_generation
  for line = first, first + count - 1 do
    vim.api.nvim_buf_set_extmark(buffer, remote_edit_namespace, line, 0, { line_hl_group = "DiffText" })
  end
  vim.defer_fn(function()
    if generation == remote_edit_generation and vim.api.nvim_buf_is_valid(buffer) then
      vim.api.nvim_buf_clear_namespace(buffer, remote_edit_namespace, 0, -1)
    end
  end, REMOTE_EDIT_HIGHLIGHT_MS)
end

-- Rewrites only the changed line ranges, bottom-up, so cursor, view and marks stay on their text.
local function apply_text(text)
  local old_lines = vim.api.nvim_buf_get_lines(buffer, 0, -1, false)
  local new_lines = stored_lines(text)
  local hunks = diff_lines(table.concat(old_lines, "\n") .. "\n", table.concat(new_lines, "\n") .. "\n", { result_type = "indices" })
  vim.api.nvim_buf_clear_namespace(buffer, remote_edit_namespace, 0, -1)
  for index = #hunks, 1, -1 do
    local start_old, count_old, start_new, count_new = unpack(hunks[index])
    local first = count_old > 0 and start_old - 1 or start_old
    local replacement = vim.list_slice(new_lines, start_new, start_new + count_new - 1)
    vim.api.nvim_buf_set_lines(buffer, first, first + count_old, false, replacement)
    highlight_remote_edit(first, count_new)
  end
end

-- The text `:w` would store, whether it has unsaved edits, and the change tick `gigachad_set_text` checks.
function _G.gigachad_buffer_state()
  return { text = buffer_text(), modified = vim.bo[buffer].modified, tick = vim.api.nvim_buf_get_changedtick(buffer) }
end

-- Shows `text` as one undo step, false without a change when the buffer moved past `tick`.
-- An unmodified buffer stays unmodified (the text is on disk) unless `dirty`, a modified one keeps its unsaved state.
function _G.gigachad_set_text(text, tick, dirty)
  if tick ~= vim.api.nvim_buf_get_changedtick(buffer) then return false end
  local modified = vim.bo[buffer].modified or dirty == true
  if text ~= buffer_text() then apply_text(text) end
  vim.bo[buffer].modified = modified
  return true
end

-- `save` answers { text = stored text, merged = whether edits made elsewhere changed the buffer's text }.
vim.api.nvim_create_autocmd("BufWriteCmd", {
  buffer = buffer,
  callback = function()
    local text = buffer_text()
    local ok, stored = pcall(vim.rpcrequest, channel, "save", text)
    if not ok then error("Save failed: " .. tostring(stored), 0) end
    if stored.text ~= buffer_text() then apply_text(stored.text) end
    vim.bo[buffer].modified = false
    local message = stored.merged and "Saved, merged with the agent's edits" or "Saved " .. vim.fn.fnamemodify(path, ":t")
    vim.api.nvim_echo({ { message } }, false, {})
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

if ask_keymap == true then
  vim.keymap.set("x", "<leader>a", function()
    local first, last = vim.fn.line("v"), vim.fn.line(".")
    if first > last then first, last = last, first end
    vim.cmd("normal! \27")
    local lines = vim.api.nvim_buf_get_lines(buffer, first - 1, last, false)
    vim.rpcnotify(channel, "selection", { start_line = first, end_line = last, text = table.concat(lines, "\n") })
  end, { buffer = buffer })
end

if base_path then
  local work_window = vim.api.nvim_get_current_win()
  vim.cmd("leftabove vertical split " .. vim.fn.fnameescape(base_path))
  local base_buffer = vim.api.nvim_get_current_buf()
  vim.bo[base_buffer].readonly = true
  vim.bo[base_buffer].modifiable = false
  vim.api.nvim_create_autocmd("BufWriteCmd", {
    buffer = base_buffer,
    callback = function() error("Baseline is read-only", 0) end,
  })
  vim.cmd("diffthis")
  vim.api.nvim_set_current_win(work_window)
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
