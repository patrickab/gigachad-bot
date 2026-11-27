"""Msgpack-RPC with a running Neovim, plus the Lua that bridges its `:w`, yanks and selections back to the caller.

- `SETUP_LUA` tracks the first document's buffer, `_G.gigachad_open` any further one (a private copy each). Documents
  are told apart by an id, the document's path. Every tracked buffer stays loaded (`bufhidden=hide`) when another is shown
- Each tracked buffer's `:w` becomes a `save` request carrying the id instead of writing the disk, and echoes
  `Saved, merged with the agent's edits` when the answer says edits made elsewhere changed the text
- A `BufEnter` autocmd sends a `switched` notification with the id when the window shows another tracked buffer, except for
  the switch `_G.gigachad_open` and `_G.gigachad_show` made. A `BufDelete` autocmd forgets the buffer
- A `TextYankPost` autocmd sends a `yank` notification with the yanked or deleted (`x`, `d`, `c`) text
- Normal-mode `<Esc>` sends a `leave` notification (the browser gives focus back to the app), after running the user's own
  normal-mode `<Esc>` mapping (`:noh`)
- Each of the `keymaps` sends its `event` notification from every tracked buffer whose id ends in its `suffix` (any without one),
  a rename re-evaluates that: a visual-mode keymap sends the id and the selected lines, any other the id and the cursor line
- `_G.gigachad_buffer_state` reports a buffer's text, modified flag and change tick
- `_G.gigachad_set_text` rewrites only the changed hunks, for edits made to the file elsewhere, keeping the modified flag
  (or setting it when the text differs from the stored file)
- `_G.gigachad_rename` gives the buffer the file's new name and id, keeping its text, cursor and modified flag
- `_G.gigachad_insert_lines` inserts lines below the cursor of the shown document
- `_G.gigachad_cursor_state` reports the shown document's lines, cursor row and change tick, `_G.gigachad_insert_at` inserts
  lines at a row and moves the cursor, `_G.gigachad_set_line` replaces a line. Both refuse when the buffer moved past that
  tick, so the caller can work out an edit from the lines without racing the user's typing
- `_G.gigachad_show_error` shows a message as an error, `_G.gigachad_goto_line` moves the shown document's cursor to a line and column
- With `own_clipboard`, `+` and `*` are the caller's clipboard (the browser's, not the machine's Neovim runs on): a paste
  is a `paste` request with the path of the current buffer, answered `{ text }` or `{ error }`. An error is shown and the
  last copy is pasted instead (`yy` then `p` works without the browser's clipboard), a copy stays in Neovim's registers.
  It replaces the provider the user's config chose (`vim.g.clipboard`, even one already in use), the `clipboard` option
  decides which registers reach it. Only reads in normal and visual mode ask, others (plugins reading `+` while you type,
  insert-mode `<C-r>+`) get the last copy, each ask could show the browser's paste prompt
- With `notebook` the first document is a notebook's percent text (`# %%` before each cell): its buffer is named by its id, a
  virtual path beside the notebook (`<notebook>.ipynb.py`), so the user's LSP finds the notebook's project and `.venv`. Nothing is
  ever read from or written to that name: `BufReadCmd` loads the text the caller last agreed on (a `base` request, also for `:e!`),
  `BufWriteCmd` saves through the caller, `FileWriteCmd`/`FileAppendCmd` (`:1,2w`) and `FileReadCmd` of that name (`:r`) are
  refused, no swap or undo file. `:sav` or `:file` gets the virtual name back and the buffer left under it is wiped (`:e #`). The
  buffer keeps an empty `buftype`, else `vim.fs.root` would not use its name. `<C-CR>` saves (`:w`) and on success sends a `run_cell`
  notification with the id and the cursor line, `<S-CR>` then moves the cursor to the next cell's first line, appending a cell
  after the last one. Diagnostics on its IPython lines (magics, shell escapes, cell magic cells) are dropped, those are not Python
- `more` is off and `messagesopt` makes messages time out after the user's config ran, a hit-enter prompt would stop every request
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

import msgpack

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_READ_CHUNK = 65536

# Receives one options table { channel, first_path (the first document's buffer name), first_id, base_path (optional baseline),
# keymaps ({ mode, lhs, event, suffix? } each), own_clipboard, notebook }. Mirrors Neovim's write rules: empty, otherwise a final
# newline unless both 'eol' and 'fixeol' are off.
SETUP_LUA = Path(__file__).with_name("neovim_setup.lua").read_text(encoding="utf-8")


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
