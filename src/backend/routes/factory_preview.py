"""`/api/factory/tasks/{id}/preview*`: start, read and stop the preview of a factory task, for logins with host access.

- Consumes `lib.factory_preview`, which runs the `preview` processes of the workspace's `factory.yaml` and keeps
  their state on disk, and `lib.factory`'s task rows
- Provides the preview as `Preview`: `GET` answers it or `null` when none was started, `start` answers `starting`
  at once (or the preview that already runs), `stop` answers `stopped`. The browser polls `GET` until `ready`
- Failures map to HTTP: unknown task 404, no `preview` section 409, a start or stop that fails (no free port, the
  database, a command that does not start, an invalid `factory.yaml`) 502 with its message

Host code execution: `host_access_only`.
"""

import asyncio

from fastapi import APIRouter

from backend.http_plumbing import http_errors
from backend.identity import host_access_only
from backend.routes.factory import Store, TaskId
from lib import factory
from lib.factory_preview import Preview, PreviewError, PreviewNotConfigured, get_preview, start_preview, stop_preview
from lib.factory_workspace import WorkspaceError

router = APIRouter(prefix="/api/factory", tags=["factory"], dependencies=[host_access_only("the factory")])

_ERRORS = (
    (factory.UnknownTask, 404, None),
    (PreviewNotConfigured, 409, None),
    (PreviewError, 502, None),
    (WorkspaceError, 502, None),
)


@router.get("/tasks/{task_id}/preview")
async def get_task_preview(task_id: TaskId, store: Store) -> Preview | None:
    with http_errors(*_ERRORS):
        await asyncio.to_thread(store.task, task_id)
        return await asyncio.to_thread(get_preview, task_id)


@router.post("/tasks/{task_id}/preview/start")
async def post_preview_start(task_id: TaskId, store: Store) -> Preview:
    """Start the task's preview in its workspace. One that already starts or serves is answered as it is."""
    with http_errors(*_ERRORS):
        task = await asyncio.to_thread(store.task, task_id)
        return await asyncio.to_thread(start_preview, task)


@router.post("/tasks/{task_id}/preview/stop")
async def post_preview_stop(task_id: TaskId, store: Store) -> Preview:
    """Stop the task's preview with its database and data. Fine when none runs."""
    with http_errors(*_ERRORS):
        await asyncio.to_thread(store.task, task_id)
        await asyncio.to_thread(stop_preview, task_id)
        return Preview(state="stopped")
