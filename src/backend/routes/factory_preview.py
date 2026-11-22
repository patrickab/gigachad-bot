"""`/api/factory/tasks/{id}/preview*`: start, read and stop the preview of a factory task, for logins with host access.

- Consumes `lib.factory_preview`, which runs the `preview` processes of the worktree's `factory.yaml` and keeps their
  state on disk, and the factory core's `tasks.get` (`lib.pi_host.factory_call`) for the task's worktree and repo
- `id` is a task id or a unique prefix of it; the preview is keyed by the full id
- Provides the preview as `Preview`: `GET` answers it or `null` when none was started, `start` answers `starting`
  at once (or the preview that already runs), `stop` answers `stopped`. The browser polls `GET` until `ready`
- Failures map to HTTP: the core's refusals as in `backend.routes.factory` (unknown task 404), no `preview` section 409,
  a start or stop that fails (no free port, the database, a command that does not start, an invalid `factory.yaml`)
  502 with its message

Host code execution: `host_access_only`.
"""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Path

from backend.http_plumbing import http_errors
from backend.identity import host_access_only
from backend.routes.factory import factory_result
from lib import pi_host
from lib.factory_preview import Preview, PreviewError, PreviewNotConfigured, PreviewTask, get_preview, start_preview, stop_preview

router = APIRouter(prefix="/api/factory", tags=["factory"], dependencies=[host_access_only("the factory")])

# The core's task ids are UUIDs; a unique prefix names one too.
TaskId = Annotated[str, Path(pattern=r"^[0-9a-f-]{1,36}$")]

_ERRORS = (
    (PreviewNotConfigured, 409, None),
    (PreviewError, 502, None),
)


async def _task(task_id: str) -> PreviewTask:
    task = await factory_result(pi_host.factory_call("tasks.get", {"id": task_id}))
    return PreviewTask(id=task["id"], repo=task["repo"], worktree=task["worktree"])


@router.get("/tasks/{task_id}/preview")
async def get_task_preview(task_id: TaskId) -> Preview | None:
    task = await _task(task_id)
    with http_errors(*_ERRORS):
        return await asyncio.to_thread(get_preview, task.id)


@router.post("/tasks/{task_id}/preview/start")
async def post_preview_start(task_id: TaskId) -> Preview:
    """Start the task's preview in its worktree. One that already starts or serves is answered as it is."""
    task = await _task(task_id)
    with http_errors(*_ERRORS):
        return await asyncio.to_thread(start_preview, task)


@router.post("/tasks/{task_id}/preview/stop")
async def post_preview_stop(task_id: TaskId) -> Preview:
    """Stop the task's preview with its database and data. Fine when none runs."""
    task = await _task(task_id)
    with http_errors(*_ERRORS):
        await asyncio.to_thread(stop_preview, task.id)
        return Preview(state="stopped")
