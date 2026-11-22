"""`/api/factory/*`: the agent factory for logins with host access (`backend.identity.HOST_CAPABILITY`).

- Consumes `lib.factory`, which drives each task's Pi session through `lib.pi_host` and keeps the factory's tables
- `repo` is the absolute path of a registered primary checkout (`RepoInfo.path`), `RepoInfo.name` is for display
- Provides repos, per-repo settings and local branches, and tasks: start, list, read, diff, approve, retry, cancel, message, delete
- A task's timeline is not served here: the browser reads `GET /api/pi/sessions/{id}/entries?customType=factory-`
- Failures map to HTTP: unknown repo or task 404, a path that is no primary checkout, an unknown branch or a source
  branch that does not merge into the target 400, a busy checkout or a task at work 409, a workspace, git, start or
  preview failure 502 with its message, pi-host unreachable 503, its refusals passed on

Host code execution: `host_access_only`.
"""

from collections.abc import Awaitable
from typing import Annotated

from fastapi import APIRouter, Depends, Path

from backend.http_plumbing import http_errors
from backend.identity import host_access_only
from backend.routes.upstream import call_upstream
from lib import factory, factory_preview
from lib.factory_workspace import WorkspaceBusy, WorkspaceError
from lib.git import GitError
from lib.postgres_pool import get_postgres_pool

router = APIRouter(prefix="/api/factory", tags=["factory"], dependencies=[host_access_only("the factory")])

# Task ids are lowercase alphanumerics.
TaskId = Annotated[str, Path(pattern=r"^[a-z0-9]{8,32}$")]


def get_factory_store() -> factory.FactoryStore:
    return factory.FactoryStore(get_postgres_pool())


Store = Annotated[factory.FactoryStore, Depends(get_factory_store)]


async def _call[Result](request: Awaitable[Result]) -> Result:
    """Await a `lib.factory` call, mapping its failures and pi-host's (`call_upstream`) to HTTP."""
    with http_errors(
        (factory.UnknownRepo, 404, None),
        (factory.UnknownTask, 404, None),
        (factory.NotACheckout, 400, None),
        (factory.UnknownBranch, 400, None),
        (factory.BranchConflict, 400, None),
        (factory.TaskBusy, 409, None),
        (WorkspaceBusy, 409, None),
        (factory.StartFailed, 502, None),
        (WorkspaceError, 502, None),
        (GitError, 502, None),
        (factory_preview.PreviewError, 502, None),
    ):
        return await call_upstream(request)


@router.get("/repos")
async def get_repos(store: Store) -> list[factory.RepoInfo]:
    return await _call(factory.list_repos(store))


@router.post("/repos")
async def post_repo(body: factory.NewRepo, store: Store) -> factory.RepoInfo:
    """Register the primary git checkout at `path`, repeating a known path answers the same repo."""
    return await _call(factory.register_repo(store, body.path))


@router.get("/settings")
async def get_settings(repo: str, store: Store) -> factory.RepoSettings:
    return await _call(factory.repo_settings(store, repo))


@router.get("/branches")
async def get_branches(repo: str, store: Store) -> list[str]:
    return await _call(factory.repo_branches(store, repo))


@router.put("/settings")
async def put_settings(repo: str, body: factory.RepoSettings, store: Store) -> factory.RepoSettings:
    return await _call(factory.save_repo_settings(store, repo, body))


@router.get("/tasks")
async def get_tasks(repo: str, store: Store) -> list[factory.Task]:
    """The tasks of *repo*, newest first, each with its `factory` state and `live` status."""
    return await _call(factory.list_tasks(store, repo))


@router.post("/tasks")
async def post_task(body: factory.NewTask, store: Store) -> factory.Task:
    """Start a task: its workspace, a coding session in it and `/factory start`. A start the session does not take
    answers 502 and leaves nothing behind."""
    return await _call(factory.create_task(store, body))


@router.get("/tasks/{task_id}")
async def get_task(task_id: TaskId, store: Store) -> factory.Task:
    return await _call(factory.read_task(store, task_id))


@router.get("/tasks/{task_id}/diff")
async def get_diff(task_id: TaskId, store: Store) -> factory.TaskDiff:
    """The task's changes as a unified diff against its diff base, untracked files included. A task in the checkout
    starts from the working tree as it was: what the user had not committed then is not part of its diff."""
    return await _call(factory.task_diff(store, task_id))


@router.post("/tasks/{task_id}/messages")
async def post_message(task_id: TaskId, body: factory.NewMessage, store: Store) -> factory.Task:
    """Hand a message to the task's coder. 409 for a task in the checkout while another task works there."""
    return await _call(factory.message_task(store, task_id, body.text))


@router.post("/tasks/{task_id}/session")
async def post_session(task_id: TaskId, store: Store) -> factory.Task:
    """Open the task's Pi session when it is not live, without a prompt, so the client can attach to it."""
    return await _call(factory.open_task_session(store, task_id))


@router.post("/tasks/{task_id}/{command}")
async def post_command(task_id: TaskId, command: factory.TaskCommand, store: Store) -> factory.Task:
    """Approve, retry or cancel the task. A retry answers 409 for a task in the checkout while another task works there."""
    return await _call(factory.command_task(store, task_id, command))


@router.delete("/tasks/{task_id}", status_code=204)
async def delete_task(task_id: TaskId, store: Store) -> None:
    """Remove the task with its preview, session, worktree and branch. 409 while it is running or merging (interrupted
    is removed), 502 when its preview does not stop."""
    await _call(factory.delete_task(store, task_id))
