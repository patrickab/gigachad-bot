"""Validated architecture writes: read the workspace, apply through the C4 service, commit, retry on a lost race.

The commit writes every changed file in one transaction, each checked against the
revision it was read at. When another edit got there first, ``apply`` rereads the
workspace and re-validates the operations before retrying. New files, deletions, and
files read but not changed are not revision-checked: two devices racing on those can
still overwrite each other. That is accepted, as one user edits one workspace.
"""

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import re
from typing import Any

from lib import c4_service
from lib.architecture_workspace import ArchitectureError, ArchitectureNotFound, ArchitectureStore, Snapshot
from lib.c4_service import ApplyResult, C4Error
from lib.data_store import StorageConflictError
from lib.project_store import ProjectStore

_COMMIT_ATTEMPTS = 3


@dataclass(frozen=True)
class Change:
    """Operations over the workspace sources, or over ``edit(sources)`` (hand edits, deletions).

    New top-level elements go into *file*. With *strict*, any parse error refuses the
    change: hand edits must leave a model the diagram can show. Otherwise operations
    only refuse errors they introduce.
    """

    ops: list[dict[str, Any]]
    file: str | None = None
    edit: Callable[[dict[str, str]], dict[str, str]] | None = None
    strict: bool = False


def require_project(projects: ProjectStore, slug: str) -> None:
    """Raise ``ArchitectureNotFound`` unless *slug* is one of the user's projects."""
    try:
        projects.require(slug)
    except FileNotFoundError as exc:
        raise ArchitectureNotFound(str(exc)) from exc


def _refusal(error: C4Error, deleted: Iterable[str]) -> str:
    """The first parse error as a reason. A dangling reference, the usual cause after a
    delete or a hand edit removing an element, is phrased as which line still uses what."""
    where = f"{error.file}:{error.line + 1}"
    # LikeC4 reports a dangling reference as "Could not resolve reference to ... named 'x'".
    named = re.search(r"Could not resolve reference to .* named '([^']+)'", error.message)
    if deleted := sorted(deleted):
        use = f"still refers to '{named[1]}'" if named else f"still needs it ({error.message})"
        return f"Cannot delete {', '.join(deleted)}: {where} {use}. Remove that reference first."
    if named:
        return f"Cannot save: {where} refers to '{named[1]}', which does not exist. Fix or remove that reference first."
    return f"{where}: {error.message}"


async def apply(store: ArchitectureStore, change: Change, *, check_project: Callable[[], None] | None = None) -> ApplyResult:
    """Apply *change* and commit the outcome.

    With *check_project*, an empty workspace is accepted once that check passes.
    """
    if change.file is not None:
        store.source_key(change.file)

    def read() -> Snapshot:
        snapshot = store.read(allow_empty=check_project is not None)
        if check_project is not None and not snapshot.sources:
            check_project()
        return snapshot

    for attempt in range(_COMMIT_ATTEMPTS):
        # Store reads, the project check and the commit block on Postgres or the mount, so they run off the event loop.
        snapshot = await asyncio.to_thread(read)
        sources = change.edit(snapshot.sources) if change.edit else snapshot.sources
        deleted = set(snapshot.sources) - set(sources)
        result = await c4_service.apply_operations(sources, snapshot.snapshots, change.ops, change.file)
        if change.strict and (errors := result.model.errors):
            raise ArchitectureError(_refusal(errors[0], deleted))
        try:
            await asyncio.to_thread(store.commit, snapshot, result.sources, result.snapshots, deleted=deleted)
            return result
        except StorageConflictError:
            if attempt == _COMMIT_ATTEMPTS - 1:
                raise
