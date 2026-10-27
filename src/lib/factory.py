"""Server-to-server client for the agent factory engine (`agent-sandbox factory serve`).

- The browser never calls the engine, every factory read and command goes through here
- The engine owns tasks, worktrees, agents and previews, gigachad only forwards and displays them
- Responses are validated against the contract models below, a changed engine shape fails loudly with the field name
- Listings and new tasks name their repo, task ids are engine-wide and the other calls forward them unchecked
- `GIGACHAD_FACTORY_URL` is the engine (default 127.0.0.1:8090)
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lib.upstream import UpstreamClient

DEFAULT_URL = "http://127.0.0.1:8090"
# The engine's default readiness wait (180) plus its scenario run (120) plus headroom for starting the preview's processes.
PREVIEW_RESTART_TIMEOUT_SECONDS = 330.0

_client = UpstreamClient(env="GIGACHAD_FACTORY_URL", default=DEFAULT_URL, label="factory engine")

TaskState = Literal["backlog", "todo", "in_progress", "human_review", "rework", "blocked", "merging", "done", "cancelled"]
Phase = Literal["prepare", "scout", "plan", "code", "hooks", "gates", "review", "preview", "merge"]
TaskCommand = Literal["start", "approve", "cancel", "retry"]


class ContractModel(BaseModel):
    """Engine JSON is snake_case like gigachad's API: validate and serialize by field name, ignore fields added later."""

    model_config = ConfigDict(extra="ignore", frozen=True)


class ImageInput(ContractModel):
    filename: str
    data_base64: str


class Overrides(ContractModel):
    merge: Literal["auto", "review"] | None = None
    max_rounds: int | None = None
    preview: bool | None = None
    sandbox: bool | None = None
    models: dict[str, str] | None = None


class NewTask(ContractModel):
    """A task request as the browser sends it, the engine's `TaskCreate`."""

    repo: str
    workflow: str
    title: str
    request: str
    images: list[ImageInput] = []
    overrides: Overrides = Overrides()
    start: bool = True


class NewMessage(ContractModel):
    text: str
    images: list[ImageInput] = []


class RepoInfo(ContractModel):
    """A repo registered with the engine, `path` is its main checkout, matched against omp workspaces."""

    name: str
    path: str


class NewRepo(ContractModel):
    path: str


class FactorySettings(ContractModel):
    """Engine-wide defaults every repo and workflow may override, `models` maps an agent name to a model selector."""

    merge: Literal["auto", "review"] = "review"
    max_rounds: int = Field(default=3, ge=1, le=20)
    sandbox: bool = True
    preview: bool = True
    models: dict[str, str] = {}
    agent_timeout_s: int = Field(default=3600, ge=1, le=86400)
    command_timeout_s: int = Field(default=1800, ge=1, le=86400)


class FactoryDefaults(ContractModel):
    """The user's defaults: settings plus the full markdown (front matter and prompt) of each agent, keyed by name."""

    settings: FactorySettings
    agents: dict[str, str]


class WorkflowInfo(ContractModel):
    id: str
    name: str
    phases: list[str]
    merge: str
    preview: bool


class GateResult(ContractModel):
    argv: list[str]
    exit_code: int
    output_tail: str


class PhaseRun(ContractModel):
    id: int
    phase: Phase
    round: int
    status: Literal["running", "ok", "failed"]
    started_at: datetime
    ended_at: datetime | None
    summary: str
    envelope: dict[str, Any] | None
    gate_results: list[GateResult] = []


class Message(ContractModel):
    id: int
    role: Literal["user", "agent"]
    text: str
    phase_run_id: int | None
    created_at: datetime


class Preview(ContractModel):
    status: Literal["starting", "ready", "failed", "stopped"]
    url: str
    open_url: str
    error: str | None


class Task(ContractModel):
    id: str
    repo: str
    workflow: str
    title: str
    request: str
    state: TaskState
    phase: Phase | None
    round: int
    config: dict[str, Any]
    worktree: str | None
    branch: str | None
    base_sha: str | None
    error: str | None
    created_at: datetime
    updated_at: datetime


class TaskDetail(Task):
    phases: list[PhaseRun]
    messages: list[Message]
    preview: Preview | None


class Event(ContractModel):
    seq: int
    task_id: str
    kind: str
    data: dict[str, Any]
    created_at: datetime


class EventCursor(ContractModel):
    seq: int


async def repos() -> list[RepoInfo]:
    """Repos the engine serves."""
    return await _client.request("GET", "/v1/repos", list[RepoInfo])


async def register_repo(path: str) -> RepoInfo:
    """Register the git checkout at *path* with the engine, repeating a known path returns the same repo."""
    return await _client.request("POST", "/v1/repos", RepoInfo, json=NewRepo(path=path).model_dump(mode="json"))


async def defaults() -> FactoryDefaults:
    """The engine-wide defaults: the user's saved ones, else the built-in ones."""
    return await _client.request("GET", "/v1/defaults", FactoryDefaults)


async def save_defaults(new_defaults: FactoryDefaults) -> FactoryDefaults:
    """Replace the user's defaults, the engine validates every agent and refuses with 422 otherwise."""
    return await _client.request("PUT", "/v1/defaults", FactoryDefaults, json=new_defaults.model_dump(mode="json"))


async def workflows(repo: str) -> list[WorkflowInfo]:
    """Workflows of a repo: the built-in ones plus its `.factory/workflows`."""
    return await _client.request("GET", f"/v1/repos/{repo}/workflows", list[WorkflowInfo])


async def tasks(repo: str, state: TaskState | None = None) -> list[Task]:
    """Tasks of a repo, optionally in one state."""
    params: dict[str, str | int] = {"repo": repo}
    if state is not None:
        params["state"] = state
    return await _client.request("GET", "/v1/tasks", list[Task], params=params)


async def create_task(new_task: NewTask) -> Task:
    """Create a task, it starts at once unless `new_task.start` is false."""
    return await _client.request("POST", "/v1/tasks", Task, json=new_task.model_dump(mode="json"))


async def task(task_id: str) -> TaskDetail:
    """One task with its phase runs, messages and preview."""
    return await _client.request("GET", f"/v1/tasks/{task_id}", TaskDetail)


async def events(repo: str, after: int = 0) -> list[Event]:
    """Task events of a repo with `seq` above *after*, oldest first."""
    return await _client.request("GET", "/v1/events", list[Event], params={"after": after, "repo": repo})


async def latest_event() -> EventCursor:
    """Newest event `seq` of the whole engine (0 when empty), where a fresh poller starts."""
    return await _client.request("GET", "/v1/events/latest", EventCursor)


async def command(task_id: str, name: TaskCommand) -> Task:
    """Run a state command on a task: start a backlog task, approve a reviewed one, cancel, retry a blocked one."""
    return await _client.request("POST", f"/v1/tasks/{task_id}/{name}", Task)


async def send_message(task_id: str, message: NewMessage) -> Message:
    """Hand the user's message to the task's agent."""
    return await _client.request("POST", f"/v1/tasks/{task_id}/messages", Message, json=message.model_dump(mode="json"))


async def restart_preview(task_id: str) -> Preview:
    """Restart the task's preview and wait until it is ready or failed."""
    return await _client.request("POST", f"/v1/tasks/{task_id}/preview/restart", Preview, timeout=PREVIEW_RESTART_TIMEOUT_SECONDS)
