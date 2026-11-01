"""Live reports of changed files: a replaceable set of files and folders watched with `watchfiles`.

A file is watched through its parent folder (non-recursive), so a save that writes a temp file and renames it over the
target is seen, a watch on the file itself would stay on the replaced inode. Folders are watched recursively without LaTeX
build output, `.git` and `node_modules`. Created, modified, moved-in and deleted files are all reported, once per
debounce window, as the absolute path the caller registered.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from watchfiles import Change, watch

from lib.latex import is_build_output

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

log = logging.getLogger(__name__)

DEBOUNCE_MS = 100
_SKIPPED_FOLDERS = frozenset({".git", "node_modules"})
_EDITOR_SCRATCH_SUFFIXES = (".swp", ".swo", "~")
# Neovim writes this file into a folder to test that it can write there
_EDITOR_SCRATCH_NAMES = frozenset({"4913"})


def _is_editor_scratch(path: Path) -> bool:
    """Whether *path* is a file an editor creates while saving, not a file of the user."""
    return path.name in _EDITOR_SCRATCH_NAMES or path.name.endswith(_EDITOR_SCRATCH_SUFFIXES)


def _reported_in_folder(change: Change, path: str) -> bool:
    """Whether a change below a watched folder is one of its files, not a skipped folder, build output, editor scratch or a folder itself."""
    target = Path(path)
    if not _SKIPPED_FOLDERS.isdisjoint(target.parts) or is_build_output(target) or _is_editor_scratch(target):
        return False
    return change is Change.deleted or not target.is_dir()


class FileWatcher:
    """Reports changed paths to *on_change* for one watch set that `replace` swaps as a whole.

    Every watch blocks a thread of its own, started here and never taken from the event loop's executor or anyio's shared
    pool, which sync routes depend on. A watch ends within a polling step of its set being replaced or closed.
    """

    def __init__(self, on_change: Callable[[str], Awaitable[None]]) -> None:
        self._on_change = on_change
        self._loop = asyncio.get_running_loop()
        self._reports: asyncio.Queue[tuple[threading.Event, list[str]]] = asyncio.Queue()
        self._stop = threading.Event()
        self._ended: list[asyncio.Future[None]] = []
        self._delivery = asyncio.create_task(self._deliver())

    def replace(self, files: Iterable[Path], folders: Iterable[Path]) -> None:
        """Watch exactly the absolute *files* and *folders* from now on, what the previous set held reports nothing more.

        The folders of the files and the folders themselves must exist, one that vanished later is logged and not watched.
        """
        self._stop.set()
        stop = self._stop = threading.Event()
        watched_files = frozenset(str(file) for file in files)
        file_folders = {str(Path(file).parent) for file in watched_files}
        watched_folders = {str(folder) for folder in folders}
        if file_folders:
            self._start(stop, file_folders, lambda _change, path: path in watched_files, recursive=False)
        if watched_folders:
            self._start(stop, watched_folders, _reported_in_folder, recursive=True)

    async def close(self) -> None:
        """Stop watching and wait until every watch has ended."""
        self._stop.set()
        self._delivery.cancel()
        await asyncio.gather(self._delivery, *self._ended, return_exceptions=True)

    def _start(self, stop: threading.Event, folders: set[str], accepts: Callable[[Change, str], bool], *, recursive: bool) -> None:
        ended = self._loop.create_future()
        self._ended = [future for future in self._ended if not future.done()] + [ended]
        threading.Thread(target=self._watch, args=(stop, folders, accepts, recursive, ended), name="file-watch", daemon=True).start()

    def _watch(
        self, stop: threading.Event, folders: set[str], accepts: Callable[[Change, str], bool], recursive: bool, ended: asyncio.Future[None]
    ) -> None:
        """Body of a watch thread: queue the changed paths for the event loop until *stop* is set."""
        try:
            for batch in watch(*sorted(folders), watch_filter=accepts, debounce=DEBOUNCE_MS, stop_event=stop, recursive=recursive):
                self._post(self._reports.put_nowait, (stop, sorted({path for _change, path in batch})))
        except OSError:
            log.warning("Watching %s failed", sorted(folders), exc_info=True)
        finally:
            self._post(ended.set_result, None)

    def _post(self, call: Callable[..., object], *args: object) -> None:
        # The loop is closed when the process shuts down under a running watch, nothing is left to tell
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(call, *args)

    async def _deliver(self) -> None:
        while True:
            stop, paths = await self._reports.get()
            for path in paths:
                if not stop.is_set():
                    await self._on_change(path)
