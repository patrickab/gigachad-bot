from typing import Annotated

from fastapi import APIRouter, Path, Query

from backend.identity import host_access_only
from backend.routes.upstream import call_upstream
from lib import factory

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


@router.post("/tasks/{task_id}/{command}")
async def post_command(task_id: TaskId, command: factory.TaskCommand) -> factory.Task:
    return await call_upstream(factory.command(task_id, command))
