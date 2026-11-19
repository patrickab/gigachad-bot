"""The Pi `document` sessions linked to each workbench file or LaTeX project folder, one user's in `workbench/chats.json` (G4).

Consumes the user's `DataStore`, read and written through `JsonDocs` against the revision read, and `FileRoots` to find
the root file of a LaTeX project. Provides `Link`, `SessionLinks` and `owe_new_paths` for relocations.

Writers in one process take turns (`_WRITING`), so a first write, which no revision guards, never drops another's link.
A write another process landed in between is read again and the change reapplied (`_ATTEMPTS`).

Stored shape: absolute path -> `{"active", "sessions", "moved"}` (`moved` only when not empty), or a bare session id as
stored before sessions were kept.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import threading
from typing import TYPE_CHECKING

from lib import latex
from lib.data_store import DataStore, Revision, StorageConflictError, StorageNotFoundError
from lib.json_docs import JsonDocs
from lib.storage_namespace import WORKBENCH_CHATS
from lib.workbench import kind_of

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from lib.file_roots import FileRoots

_WRITING = threading.Lock()
_ATTEMPTS = 10


@dataclass(frozen=True)
class Link:
    """The active session (None: a new one), every session the file had (first linked first) and those owed its new path."""

    active: str | None = None
    sessions: tuple[str, ...] = ()
    moved: tuple[str, ...] = ()

    @classmethod
    def parse(cls, value: object) -> Link:
        """The link stored as *value*, a bare session id included, an empty link for anything else."""
        if isinstance(value, str):
            return cls(active=value, sessions=(value,))
        if not isinstance(value, dict):
            return cls()
        sessions = tuple(session for session in value.get("sessions", []) if isinstance(session, str))
        moved = tuple(session for session in value.get("moved", []) if session in sessions)
        active = value.get("active")
        return cls(active=active if isinstance(active, str) else None, sessions=sessions, moved=moved)

    def stored(self) -> dict[str, object]:
        """The stored value of this link, without an empty `moved`."""
        stored: dict[str, object] = {"active": self.active, "sessions": list(self.sessions)}
        return {**stored, "moved": list(self.moved)} if self.moved else stored


class SessionLinks:
    """One user's session links (`WORKBENCH_CHATS`).

    Every change rereads the links and retries when another process wrote them meanwhile, raising
    `StorageConflictError` only when they keep changing.
    """

    def __init__(self, docs: DataStore) -> None:
        self._json = JsonDocs(docs)

    def of(self, path: str, roots: FileRoots) -> Link:
        """The link of the absolute file or LaTeX project folder *path*.

        - A folder without a link adopts the links of the LaTeX files inside it (any depth), merged in the order they
          were stored, once: the root file's active session is the folder's, else the first file's
        """
        def adopting(links: dict) -> tuple[dict | None, Link]:
            if path not in links:
                adopted = _adopted(links, path, roots)
                if adopted is not None:
                    return adopted
            return None, Link.parse(links.get(path))
        return self._change(adopting)

    def activate(self, path: str, session: str | None) -> Link:
        """Make *session* the active one of the absolute file *path*, listed once after the earlier ones. None starts a new session."""
        def activated(current: Link) -> Link:
            sessions = current.sessions
            if session is not None and session not in sessions:
                sessions = (*sessions, session)
            return replace(current, active=session, sessions=sessions)
        return self._update(path, activated)

    def inform(self, path: str, session: str) -> Link:
        """Record that *session* knows where the file *path* is now."""
        return self._update(path, lambda current: replace(current, moved=tuple(owed for owed in current.moved if owed != session)))

    def discard(self, path: str, session: str) -> Link:
        """Unlink *session* from the file *path*, it is no longer the active one either."""
        def discarded(current: Link) -> Link:
            return Link(
                active=None if current.active == session else current.active,
                sessions=tuple(kept for kept in current.sessions if kept != session),
                moved=tuple(owed for owed in current.moved if owed != session),
            )
        return self._update(path, discarded)

    def forget_inside(self, folders: list[Path]) -> None:
        """Remove the links of the *folders* and of the files inside them."""
        def outside(links: dict) -> tuple[dict | None, None]:
            kept = {key: link for key, link in links.items() if not any(Path(key).is_relative_to(folder) for folder in folders)}
            return (kept if kept != links else None), None
        self._change(outside)

    def _read(self) -> tuple[dict, Revision | None]:
        """The links and their revision, empty and unrevised when there are none."""
        try:
            links, revision = self._json.read_with_revision(WORKBENCH_CHATS)
        except (StorageNotFoundError, ValueError):
            return {}, None
        return (links if isinstance(links, dict) else {}), revision

    def _update(self, path: str, change: Callable[[Link], Link]) -> Link:
        """Apply *change* to the link of *path* and store it when it changed or was not stored in the current shape."""
        def updated(links: dict) -> tuple[dict | None, Link]:
            current = Link.parse(links.get(path))
            link = change(current)
            if link != current or not isinstance(links.get(path), dict):
                return {**links, path: link.stored()}, link
            return None, link
        return self._change(updated)

    def _change[Result](self, change: Callable[[dict], tuple[dict | None, Result]]) -> Result:
        """Store the links *change* returns for the current ones (None: unchanged) and return its result.

        - *change* reruns on the links read again when another process wrote them in between
        - Raises StorageConflictError when they keep changing
        """
        for _ in range(_ATTEMPTS):
            with _WRITING:
                links, revision = self._read()
                changed, result = change(links)
                if changed is None:
                    return result
                try:
                    self._json.write(WORKBENCH_CHATS, changed, expected=revision)
                except StorageConflictError:
                    continue
                return result
        raise StorageConflictError(f"{WORKBENCH_CHATS} kept changing while it was updated")


def _root_key(files: Iterable[str], roots: FileRoots) -> str | None:
    """The one of *files* that is the root document of the project they belong to, None when none is."""
    keys = tuple(files)
    for key in keys:
        try:
            root = latex.project_of(Path(key), ceiling=roots.ceiling_of(Path(key))).root
        except OSError:
            continue
        if root is not None and str(root) in keys:
            return str(root)
    return None


def _adopted(links: dict, folder: str, roots: FileRoots) -> tuple[dict, Link] | None:
    """*links* with the links of the LaTeX files inside *folder* merged into one under *folder*, and that link. None without any."""
    prefix = f"{folder.rstrip('/')}/"
    files = {key: Link.parse(value) for key, value in links.items() if key.startswith(prefix) and kind_of(key) == "latex"}
    if not files:
        return None
    leading = files.get(_root_key(files, roots) or "") or next(iter(files.values()))
    merged = Link(
        active=leading.active,
        sessions=tuple(dict.fromkeys(session for link in files.values() for session in link.sessions)),
        moved=tuple(dict.fromkeys(session for link in files.values() for session in link.moved)),
    )
    return {**{key: value for key, value in links.items() if key not in files}, folder: merged.stored()}, merged


def owe_new_paths(before: object, after: object) -> object:
    """The stored links *after* a relocation, every file whose path changed owing each of its sessions the new path (`moved`).

    A session told its file's path earlier would edit the old one, so its next message names the file again
    """
    if not isinstance(before, dict) or not isinstance(after, dict):
        return after
    return {path: link if path in before else _owing_every_session(link) for path, link in after.items()}


def _owing_every_session(link: object) -> object:
    """A file's stored link with all its sessions in `moved`, a bare session id as stored before sessions were kept too."""
    if isinstance(link, str):
        return {"active": link, "sessions": [link], "moved": [link]}
    if not isinstance(link, dict):
        return link
    return {**link, "moved": [session for session in link.get("sessions", []) if isinstance(session, str)]}
