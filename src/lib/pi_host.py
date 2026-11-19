"""Server-to-server client for pi-host, the Bun service that supervises Pi sessions and serves the completion gateway.

- The only module that knows pi-host (D6): its URL, its shared bearer token, its HTTP shapes and its `/ws` endpoint
- Consumes pi-host's HTTP API (`.docs/pi-migration-contracts.md`, "pi-host HTTP") and the token file pi-host creates on start
- Provides the session, file and private-state calls of `/api/pi/*`, a WebSocket opener for the `/ws` proxy and the gateway URL
- Answers are validated against the contract models below, a changed shape fails loudly with the field name
- Failures are `PiHostUnavailable` (503), `PiHostBadResponse` (502) or `PiHostRejected` (pi-host's own status and body)
- Also lists the repositories under `~/git` a session may start in, and resolves a repository name to its folder
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

from lib.upstream import (
    TIMEOUT_SECONDS,
    UpstreamContractError,
    UpstreamRequestFailed,
    UpstreamUnavailable,
    error_message,
    upstream_url,
    validate_answer,
)

DEFAULT_URL = "http://127.0.0.1:8300"
LABEL = "pi-host"
REPOSITORIES_ROOT = Path.home() / "git"

# Tests swap in an `httpx.MockTransport`, None is the real network.
_transport: httpx.AsyncBaseTransport | None = None


class PiHostUnavailable(UpstreamUnavailable):
    """pi-host does not answer, or its token file is missing (it never started)."""


class PiHostBadResponse(UpstreamContractError):
    """pi-host answered with a shape this adapter does not know, the message names the field."""


class PiHostRejected(UpstreamRequestFailed):
    """pi-host refused the request: `status` and `body` (its JSON, else its text) are its own."""

    def __init__(self, status: int, message: str, body: object) -> None:
        super().__init__(status, message)
        self.body = body


class ContractModel(BaseModel):
    """pi-host JSON is camelCase and passes through to the browser as camelCase (the Pi feature speaks Pi's shapes)."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore", frozen=True)


class LiveStatus(ContractModel):
    running: bool
    waiting: bool
    attached: int


class SessionInfo(ContractModel):
    session_id: str
    session_path: str
    cwd: str
    machine: Literal["local"]
    created_at: str
    modified_at: str
    first_message: str
    message_count: int
    parent_session: str | None = None
    live: LiveStatus | None
    pinned: bool
    archived: bool


class SessionList(ContractModel):
    sessions: list[SessionInfo]


class SessionAnswer(ContractModel):
    session: SessionInfo


class ModelChoice(ContractModel):
    provider: str
    id: str


class ResumeFromFile(ContractModel):
    session_path: str


class ForkPoint(ContractModel):
    session_path: str
    entry_id: str


class ResumeFork(ContractModel):
    """A new session with the branch of *session_path* from the root to *entry_id* (inclusive); the leaf clones."""

    fork_of: ForkPoint


Resume = Literal["continue"] | ResumeFromFile | ResumeFork | None


class SessionState(ContractModel):
    """Private per-session state (M19): `revision` is the one to send back on the next write."""

    revision: int
    data: dict[str, object]


class StateRevision(ContractModel):
    revision: int


class FileMatch(ContractModel):
    path: str
    score: float


class FileList(ContractModel):
    files: list[FileMatch]


class SearchHit(ContractModel):
    """A session matching a transcript search (M24): the text around the hit and the entry holding it."""

    session: SessionInfo
    snippet: str
    entry_id: str


class SearchResult(ContractModel):
    hits: list[SearchHit]


class Health(ContractModel):
    ok: bool
    pi_version: str


@dataclass(frozen=True)
class Repository:
    name: str
    path: str


def base_url() -> str:
    """Base URL the backend reaches pi-host at (`GIGACHAD_PI_HOST_URL`)."""
    return upstream_url("GIGACHAD_PI_HOST_URL", DEFAULT_URL)


def gateway_url() -> str:
    """Base URL of pi-host's OpenAI-compatible completion gateway."""
    return f"{base_url()}/v1"


def _token_file() -> Path:
    if override := os.environ.get("PI_HOST_TOKEN_FILE"):
        return Path(override)
    state_home = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state_home) / "gigachad-bot" / "pi-host.token"


def token() -> str:
    """The shared bearer token pi-host wrote on first start, `PiHostUnavailable` when there is none."""
    path = _token_file()
    try:
        value = path.read_text().strip()
    except OSError as exc:
        raise PiHostUnavailable(f"{LABEL} token {path} is unreadable, is pi-host running? ({exc})") from exc
    if not value:
        raise PiHostUnavailable(f"{LABEL} token {path} is empty")
    return value


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {token()}"}


def _body(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return response.text


async def _send(method: str, path: str, *, json: object = None, params: dict[str, str | int] | None = None) -> httpx.Response:
    """One authenticated request, `PiHostRejected` for an error status."""
    headers = _auth_headers()
    try:
        async with httpx.AsyncClient(base_url=base_url(), timeout=TIMEOUT_SECONDS, headers=headers, transport=_transport) as client:
            response = await client.request(method, path, json=json, params=params)
    except httpx.HTTPError as exc:
        raise PiHostUnavailable(f"{LABEL} at {base_url()} does not answer: {exc}") from exc
    if response.is_error:
        raise PiHostRejected(response.status_code, error_message(response), _body(response))
    return response


async def _request[Result](
    method: str, path: str, result: type[Result], *, json: object = None, params: dict[str, str | int] | None = None
) -> Result:
    response = await _send(method, path, json=json, params=params)
    try:
        return validate_answer(result, response.content, label=LABEL, path=path)
    except UpstreamContractError as exc:
        raise PiHostBadResponse(str(exc)) from exc


async def health() -> Health:
    return await _request("GET", "/health", Health)


async def list_sessions(cwd: str | None, *, archived: bool) -> list[SessionInfo]:
    """Sessions newest first, of *cwd* when given, from the archive when *archived*."""
    params: dict[str, str | int] = {"archived": int(archived)}
    if cwd is not None:
        params["cwd"] = cwd
    return (await _request("GET", "/sessions", SessionList, params=params)).sessions


async def open_session(
    cwd: str, resume: Resume, *, model: ModelChoice | None = None, thinking: str | None = None
) -> SessionInfo:
    """Spawn a coding session in *cwd*, or return the live one; a fork writes its file first. A session file held
    elsewhere answers 409 `locked`, an unknown fork source or entry 404."""
    body: dict[str, object] = {
        "cwd": cwd,
        "profile": "coding",
        "resume": resume.model_dump(by_alias=True) if isinstance(resume, ContractModel) else resume,
    }
    if model is not None:
        body["model"] = model.model_dump(by_alias=True)
    if thinking is not None:
        body["thinking"] = thinking
    return (await _request("POST", "/sessions", SessionAnswer, json=body)).session


async def close_session(session_id: str) -> None:
    """Stop the session's Pi child, its file stays."""
    await _send("POST", f"/sessions/{session_id}/close")


async def archive_session(session_id: str) -> SessionInfo:
    """Move the session file into the archive, 409 while it is live."""
    return (await _request("POST", f"/sessions/{session_id}/archive", SessionAnswer)).session


async def restore_session(session_id: str) -> SessionInfo:
    return (await _request("POST", f"/sessions/{session_id}/restore", SessionAnswer)).session


async def pin_session(session_id: str, pinned: bool) -> SessionInfo:
    """Pin (never idle-evicted) or unpin the session."""
    return (await _request("PUT", f"/sessions/{session_id}/pin", SessionAnswer, json={"pinned": pinned})).session


async def session_state(session_id: str) -> SessionState:
    return await _request("GET", f"/sessions/{session_id}/state", SessionState)


async def write_session_state(session_id: str, state: SessionState) -> int:
    """Store *state*, returns the new revision. A stale revision answers 409 with the current state as body."""
    body = state.model_dump(by_alias=True)
    return (await _request("PUT", f"/sessions/{session_id}/state", StateRevision, json=body)).revision


async def find_files(cwd: str, query: str, limit: int | None = None) -> list[FileMatch]:
    """Files of *cwd* fuzzy-ranked against *query*, relative paths (M7). A blank *query* answers the cached list
    unranked, whole unless *limit* is given (K6: the UI ranks it)."""
    params: dict[str, str | int] = {"cwd": cwd, "q": query}
    if limit is not None:
        params["limit"] = limit
    return (await _request("GET", "/files", FileList, params=params)).files


async def search_sessions(query: str, limit: int = 20) -> list[SearchHit]:
    """Live sessions whose user and assistant text matches *query*, best first, newest first on ties (M24)."""
    params: dict[str, str | int] = {"q": query, "limit": limit}
    return (await _request("GET", "/search", SearchResult, params=params)).hits


async def open_websocket() -> ClientConnection:
    """Connect to pi-host's `/ws` with the bearer token and no `Origin`, `PiHostUnavailable` when it refuses or is down."""
    url = "ws" + base_url().removeprefix("http") + "/ws"
    try:
        return await connect(
            url, additional_headers=_auth_headers(), max_size=None, compression=None, ping_interval=None, proxy=None
        )
    except (OSError, WebSocketException, TimeoutError) as exc:
        raise PiHostUnavailable(f"{LABEL} refused the WebSocket: {exc}") from exc


def _is_repository(directory: Path) -> bool:
    return directory.is_dir() and not directory.name.startswith(".") and (directory / ".git").exists()


def list_repositories() -> list[Repository]:
    """Git checkouts directly under `~/git`, by name."""
    if not REPOSITORIES_ROOT.is_dir():
        return []
    children = sorted(REPOSITORIES_ROOT.iterdir(), key=lambda child: child.name.lower())
    return [Repository(name=child.name, path=str(child)) for child in children if _is_repository(child)]


def resolve_repository(name: str) -> Path | None:
    """The checkout called *name* directly under `~/git`, None for anything else (paths, symlinks out, non-repos)."""
    candidate = REPOSITORIES_ROOT / name
    if "/" in name or name in ("", ".", "..") or candidate.is_symlink() or not _is_repository(candidate):
        return None
    return candidate
