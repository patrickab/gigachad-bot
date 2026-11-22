"""One-time move of the factory's Postgres rows into the factory core, which owns repos, settings and tasks now.

- Consumes the tables `factory_repo_settings` and `factory_tasks` (`db/schema/0017_factory_tasks.sql`, `migrated_at`
  from `0020_factory_rows_migrated.sql`) and the core's actions through `lib.pi_host.factory_call`:
  `settings.get`/`settings.set` and `tasks.import`
- Provides `migrate_factory_rows`, run at backend start over the rows not marked yet: each row is marked
  (`migrated_at`) as soon as it moved or never can, so a later run never imports a task again that the core removed
  in between, and a failed row stays unmarked for the next start
- A repo's settings start from the old defaults (merge `review`, 3 rounds: `{}` meant that before the core) with its
  stored keys laid over them, written only when the core stores none for the repo; one it already stores is logged
  and kept (another backend on the same Pi agent dir may have moved them first)
- A repo the core cannot read (gone from disk, no default branch) keeps its settings unmoved, logged
- A task becomes the core's record (its 32-hex id as the dashed UUID the core expects, `sessionDir` the folder of its
  session file); the preview of the old id is stopped, a preview is keyed by the task id
- A task without worktree, branch or session path (an older version worked in the main checkout) is skipped with one
  log line: the core removes a task's worktree, which would be the main checkout
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from psycopg_pool import ConnectionPool

from lib import factory_preview, pi_host

log = logging.getLogger(__name__)

SETTINGS_TABLE = "factory_repo_settings"
TASKS_TABLE = "factory_tasks"
# The core's stored settings for a repo before any key of its own: what the old factory resolved `{}` to.
OLD_DEFAULT_SETTINGS: dict[str, Any] = {"merge": "review", "maxRounds": 3}
# Old snake_case keys to the core's stored settings keys; a key not named here is dropped.
_SETTINGS_KEYS = {
    "merge": "merge", "max_rounds": "maxRounds", "review": "review", "preview": "preview", "models": "models", "branch": "branch",
}
# The core's answer for a repo that is gone or has no default branch: its settings are skipped, they block nothing.
UNMOVABLE_REPO_CODES = ("not_found", "refused")
# Settings saved while a task had a source and a target branch name the one it merged into.
_LEGACY_BRANCH_KEYS = ("target_branch", "source_branch")


@dataclass(frozen=True)
class TaskRow:
    id: str
    repo: str
    session_id: str
    session_path: str | None
    worktree: str | None
    branch: str | None
    title: str
    created_at: datetime


def core_settings(stored: dict[str, Any]) -> dict[str, Any]:
    """A `factory_repo_settings.settings` value as the core's stored settings: the old defaults, then its keys in
    camelCase over them, unset values dropped."""
    settings = OLD_DEFAULT_SETTINGS | {core: stored[old] for old, core in _SETTINGS_KEYS.items() if stored.get(old) is not None}
    if "branch" not in settings and (legacy := next((stored[key] for key in _LEGACY_BRANCH_KEYS if stored.get(key)), None)):
        settings["branch"] = legacy
    if "models" in settings:
        settings["models"] = {role: model for role, model in settings["models"].items() if role in ("coder", "reviewer") and model}
    return settings


def core_task_id(row_id: str) -> str:
    """The core's id of the task *row_id*: the dashed form of a 32-hex UUID, any other id as it is."""
    try:
        return str(UUID(hex=row_id))
    except ValueError:
        return row_id


def task_record(row: TaskRow) -> dict[str, str] | None:
    """The core's record of *row*, None when it has no worktree, branch or session file of its own."""
    if not row.worktree or not row.branch or not row.session_path:
        return None
    return {
        "id": core_task_id(row.id),
        "repo": row.repo,
        "worktree": row.worktree,
        "branch": row.branch,
        "title": row.title,
        "sessionId": row.session_id,
        "sessionDir": str(Path(row.session_path).parent),
        "createdAt": row.created_at.isoformat(),
    }


def _rows(pool: ConnectionPool) -> tuple[list[tuple[str, dict[str, Any]]], list[TaskRow]]:
    """The rows still to move."""
    with pool.connection() as connection:
        pending = "migrated_at IS NULL"
        settings = connection.execute(f"SELECT repo, settings FROM {SETTINGS_TABLE} WHERE {pending} ORDER BY repo").fetchall()
        columns = "id, repo, session_id, session_path, worktree, branch, title, created_at"
        tasks = connection.execute(f"SELECT {columns} FROM {TASKS_TABLE} WHERE {pending} ORDER BY created_at, id").fetchall()
    return [(repo, stored or {}) for repo, stored in settings], [TaskRow(*row) for row in tasks]


def _mark(pool: ConnectionPool, table: str, key: str, value: str) -> None:
    with pool.connection() as connection:
        connection.execute(f"UPDATE {table} SET migrated_at = now() WHERE {key} = %s", (value,))


async def _move_settings(repo: str, stored: dict[str, Any]) -> None:
    current = await pi_host.factory_call("settings.get", {"repo": repo})
    if current.get("stored"):
        log.warning("The factory core already stores settings for %s, the old ones stay unmoved", repo)
        return
    await pi_host.factory_call("settings.set", {"repo": repo, "settings": core_settings(stored)})


async def migrate_factory_rows(pool: ConnectionPool) -> bool:
    """Move the unmarked rows of the factory tables into the core, marking each one. True when no row is left to move,
    False when a row failed: it stays for the next start, each failure logged."""
    settings, tasks = await asyncio.to_thread(_rows, pool)
    if not settings and not tasks:
        return True
    failed = moved = 0
    for repo, stored in settings:
        try:
            await _move_settings(repo, stored)
        except pi_host.FactoryCallError as exc:
            log.warning("The factory settings of %s did not move into the factory core: %s", repo, exc)
            if exc.code not in UNMOVABLE_REPO_CODES:
                failed += 1
                continue
        except Exception as exc:  # noqa: BLE001 - every failure is logged and keeps the row for the next start
            failed += 1
            log.warning("The factory settings of %s did not move into the factory core: %s", repo, exc)
            continue
        await asyncio.to_thread(_mark, pool, SETTINGS_TABLE, "repo", repo)
    for row in tasks:
        record = task_record(row)
        if record is None:
            log.warning("The factory task %s (%s) has no worktree of its own, not moved into the factory core", row.id, row.title)
            await asyncio.to_thread(_mark, pool, TASKS_TABLE, "id", row.id)
            continue
        try:
            await pi_host.factory_call("tasks.import", {"record": record})
        except Exception as exc:  # noqa: BLE001 - see above
            failed += 1
            log.warning("The factory task %s (%s) did not move into the factory core: %s", row.id, row.title, exc)
            continue
        await asyncio.to_thread(_mark, pool, TASKS_TABLE, "id", row.id)
        moved += 1
        try:
            await asyncio.to_thread(factory_preview.stop_preview, row.id)
        except factory_preview.PreviewError as exc:
            log.warning("The preview of the old factory task %s did not stop: %s", row.id, exc)
    if failed:
        log.warning("%d factory rows did not move into the factory core, they stay for the next start", failed)
        return False
    log.info("Moved the factory rows into the factory core: %d repos, %d of %d tasks", len(settings), moved, len(tasks))
    return True
