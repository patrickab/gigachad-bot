"""Server-to-server client for omp-web, the omp tabs' chat and settings surface.

- The browser frames omp-web for chat and settings, every other omp read goes through here
- Requests carry no `Origin`, so omp-web's cross-origin API guard lets them through
- Password-protected omp-web: set `GIGACHAD_OMP_WEB_AUTH` to the value whose SHA-256 is its `OMP_WEB_TRUSTED_HEADER_SHA256`
- Responses are validated against the contract models below, a changed omp-web shape fails loudly with the field name
"""

import asyncio
import os
from typing import Literal, TypeVar

import httpx
from pydantic import AliasGenerator, BaseModel, ConfigDict, ValidationError
from pydantic.alias_generators import to_camel

DEFAULT_URL = "http://127.0.0.1:30177"
TRUSTED_HEADER = "x-omp-web-auth"
TIMEOUT_SECONDS = 10.0

# Tests swap in an `httpx.MockTransport`, None is the real network.
_transport: httpx.AsyncBaseTransport | None = None

Model = TypeVar("Model", bound=BaseModel)


class OmpWebUnavailable(Exception):
    """omp-web does not answer."""


class OmpWebContractError(Exception):
    """omp-web answered with a shape this adapter does not know."""


class OmpWebRequestFailed(Exception):
    """omp-web refused the request, `status` and `message` are its own."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class ContractModel(BaseModel):
    """omp-web JSON is camelCase, gigachad's API is snake_case: validate by alias, serialize by field name."""

    model_config = ConfigDict(alias_generator=AliasGenerator(validation_alias=to_camel), extra="ignore", frozen=True)


class OmpSession(ContractModel):
    id: str
    path: str
    cwd: str
    name: str | None = None
    created: str
    modified: str
    message_count: int
    first_message: str = ""
    parent_session_id: str | None = None
    project_root: str | None = None
    worktree_branch: str | None = None


class OmpWorkspace(ContractModel):
    path: str
    alias: str | None = None
    added_at: str | None = None


class OmpArchivedSession(ContractModel):
    key: str
    id: str
    cwd: str
    name: str | None = None
    created: str
    archived_at: str
    message_count: int
    first_message: str = ""
    status: Literal["complete", "interrupted", "aborted", "error", "pending", "unknown"] | None = None


class OmpOverview(BaseModel):
    sessions: list[OmpSession]
    running_session_ids: list[str]
    workspaces: list[OmpWorkspace]


class OmpSystemPrompt(BaseModel):
    running: bool
    system_prompt: str | None


class _SessionList(ContractModel):
    sessions: list[OmpSession]
    running_session_ids: list[str] = []


class _WorkspaceList(ContractModel):
    projects: list[OmpWorkspace]


class _ArchiveList(ContractModel):
    archives: list[OmpArchivedSession]


class _AgentState(ContractModel):
    system_prompt: str | list[str] | None = None


class _AgentStatus(ContractModel):
    running: bool
    state: _AgentState | None = None


class _Added(ContractModel):
    project: OmpWorkspace


class _Archived(ContractModel):
    ok: bool


class _Restored(ContractModel):
    session_id: str


def omp_web_url() -> str:
    """Base URL of omp-web, also the frame source in the browser."""
    return os.environ.get("GIGACHAD_OMP_WEB_URL", DEFAULT_URL).rstrip("/")


def _headers() -> dict[str, str]:
    auth = os.environ.get("GIGACHAD_OMP_WEB_AUTH")
    return {TRUSTED_HEADER: auth} if auth else {}


def _contract_violation(path: str, error: ValidationError) -> OmpWebContractError:
    first = error.errors()[0]
    field = ".".join(str(part) for part in first["loc"]) or "<root>"
    return OmpWebContractError(f"omp-web {path}: field `{field}` {first['msg'].lower()}")


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200] or f"HTTP {response.status_code}"
    return str(body.get("error") or body) if isinstance(body, dict) else str(body)


async def _request(method: str, path: str, model: type[Model], json: object = None) -> Model:
    try:
        async with httpx.AsyncClient(
            base_url=omp_web_url(), timeout=TIMEOUT_SECONDS, headers=_headers(), transport=_transport
        ) as client:
            response = await client.request(method, path, json=json)
    except httpx.HTTPError as exc:
        raise OmpWebUnavailable(f"omp-web at {omp_web_url()} does not answer: {exc}") from exc
    if response.is_error:
        raise OmpWebRequestFailed(response.status_code, _error_message(response))
    try:
        return model.model_validate_json(response.content)
    except ValidationError as exc:
        raise _contract_violation(path, exc) from exc


async def overview() -> OmpOverview:
    """Every session, the running ones and every workspace (registered plus session-discovered)."""
    session_list, workspace_list = await asyncio.gather(
        _request("GET", "/api/sessions", _SessionList),
        _request("GET", "/api/projects", _WorkspaceList),
    )
    return OmpOverview(
        sessions=session_list.sessions,
        running_session_ids=session_list.running_session_ids,
        workspaces=workspace_list.projects,
    )


async def add_workspace(path: str) -> OmpWorkspace:
    """Register a directory as workspace, worktrees resolve to their main repository."""
    return (await _request("POST", "/api/projects", _Added, json={"cwd": path})).project


async def archived_sessions() -> list[OmpArchivedSession]:
    return (await _request("GET", "/api/sessions/archive", _ArchiveList)).archives


async def archive_session(session_id: str) -> None:
    """Stop the session's omp process and move its file into omp's archive. Sessions with forks answer 409."""
    await _request("POST", f"/api/sessions/{session_id}/archive", _Archived)


async def restore_session(key: str) -> str:
    """Restore an archived session, returns its session id."""
    return (await _request("POST", "/api/sessions/archive", _Restored, json={"key": key})).session_id


async def system_prompt(session_id: str) -> OmpSystemPrompt:
    """System prompt of a running session. Never starts omp, a stopped session has none."""
    status = await _request("GET", f"/api/agent/{session_id}", _AgentStatus)
    prompt = status.state.system_prompt if status.state else None
    joined = "\n\n".join(prompt) if isinstance(prompt, list) else prompt
    return OmpSystemPrompt(running=status.running, system_prompt=joined)
