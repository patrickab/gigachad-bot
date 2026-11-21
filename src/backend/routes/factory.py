import asyncio
from collections.abc import Awaitable
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query

from backend.http_plumbing import http_errors
from backend.identity import host_access_only
from backend.routes.upstream import call_upstream
from lib import factory, factory_sessions, pi_host

router = APIRouter(prefix="/api/factory", tags=["factory"], dependencies=[host_access_only("the factory")])

# Task ids are lowercase alphanumerics, the pattern keeps anything else out of the engine's URL space.
TaskId = Annotated[str, Path(pattern=r"^[a-z0-9]{8,32}$")]
# Repo names follow the engine's identifier rule.
RepoName = Annotated[str, Query(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]


@router.get("/repos")
async def get_repos() -> list[factory.RepoInfo]:
    return await call_upstream(factory.repos())


@router.post("/repos")
async def post_repo(body: factory.NewRepo) -> factory.RepoInfo:
    return await call_upstream(factory.register_repo(body.path))


@router.get("/defaults")
async def get_defaults() -> factory.FactoryDefaults:
    return await call_upstream(factory.defaults())


@router.put("/defaults")
async def put_defaults(body: factory.FactoryDefaults) -> factory.FactoryDefaults:
    return await call_upstream(factory.save_defaults(body))


@router.get("/workflows")
async def get_workflows(repo: RepoName) -> list[factory.WorkflowInfo]:
    return await call_upstream(factory.workflows(repo))


@router.get("/tasks")
async def get_tasks(repo: RepoName, state: factory.TaskState | None = None) -> list[factory.Task]:
    return await call_upstream(factory.tasks(repo, state))


@router.post("/tasks")
async def post_task(body: factory.NewTask) -> factory.Task:
    return await call_upstream(factory.create_task(body))


@router.get("/tasks/{task_id}")
async def get_task(task_id: TaskId) -> factory.TaskDetail:
    return await call_upstream(factory.task(task_id))


@router.get("/events")
async def get_events(repo: RepoName, after: Annotated[int, Query(ge=0)] = 0) -> list[factory.Event]:
    return await call_upstream(factory.events(repo, after))


@router.get("/events/latest")
async def get_latest_event() -> factory.EventCursor:
    return await call_upstream(factory.latest_event())


@router.post("/tasks/{task_id}/messages")
async def post_message(task_id: TaskId, body: factory.NewMessage) -> factory.Message:
    return await call_upstream(factory.send_message(task_id, body))


@router.post("/tasks/{task_id}/preview/restart")
async def post_preview_restart(task_id: TaskId) -> factory.Preview:
    return await call_upstream(factory.restart_preview(task_id))


class TaskSessionAnswer(pi_host.ContractModel):
    phase: str
    session: pi_host.SessionInfo


async def _pi[Result](request: Awaitable[Result]) -> Result:
    """Await a `pi_host` call: unreachable 503, unknown shape 502, refusal with its status."""
    try:
        return await request
    except pi_host.PiHostUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except pi_host.PiHostBadResponse as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except pi_host.PiHostRejected as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


@router.post("/tasks/{task_id}/session")
async def post_session(task_id: TaskId) -> TaskSessionAnswer:
    """Open the task agent's newest saved Pi session in pi-host, as a coding session in the task's worktree.

    - The session is copied into the worktree's Pi sessions (`lib.factory_sessions`), a copy pi-host has live is left as is
    - 404 before the task has a worktree or a saved session
    """
    task, saved = await asyncio.gather(call_upstream(factory.task(task_id)), call_upstream(factory.latest_session(task_id)))
    if task.worktree is None:
        raise HTTPException(status_code=404, detail="The task has no worktree")
    live = {session.session_id for session in await _pi(pi_host.list_sessions(task.worktree, archived=False)) if session.live}
    if saved.session_id in live:
        session = await _pi(pi_host.open_session(task.worktree, pi_host.ResumeById(session_id=saved.session_id)))
    else:
        with http_errors((ValueError, 502, None)):
            path = await asyncio.to_thread(factory_sessions.import_session, saved.content, task.worktree)
        session = await _pi(pi_host.open_session(task.worktree, pi_host.ResumeFromFile(session_path=str(path))))
    return TaskSessionAnswer(phase=saved.phase, session=session)


@router.post("/tasks/{task_id}/{command}")
async def post_command(task_id: TaskId, command: factory.TaskCommand) -> factory.Task:
    return await call_upstream(factory.command(task_id, command))
