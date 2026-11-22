"""Server-to-server client for pi-host, the Bun service that supervises Pi sessions and serves the completion gateway.

- The only module that knows pi-host (D6): its URL, its shared bearer token, its HTTP shapes and its `/ws` endpoint
- Consumes pi-host's HTTP API (`.docs/pi-migration-contracts.md`, "pi-host HTTP") and the token file pi-host creates on start
- Provides the session, file and private-state calls of `/api/pi/*` and the workbench chats (`document` profile with the
  backend's system prompt, P3), a WebSocket opener for the `/ws` proxy and the gateway URL
- Provides what the factory drives a task's session with: a prompt without an attached client, its `factory-` entries,
  its removal, and `SessionInfo.factory`
- Answers are validated against the contract models below, a changed shape fails loudly with the field name
- Failures are `PiHostUnavailable` (503), `PiHostBadResponse` (502) or `PiHostRejected` (pi-host's own status and body)
- Also scans the disk for the repositories a session may start in (`scan_repositories`, stored by `repository_index`)
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time
from typing import Any, Literal

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
# pi-host gives each RPC to a Pi child 30 s (`REQUEST_TIMEOUT_MS`), a start runs several (`get_state`, `set_model`,
# `set_thinking_level`) and a prompt waits for the extension command it runs. Calls that wait on a child get more than
# that, so the backend never hangs up on a request pi-host is still allowed to finish. Reads keep `TIMEOUT_SECONDS`.
CHILD_TIMEOUT_SECONDS = 75.0
# A child closes within pi-host's `CLOSE_GRACE_MS` (10 s).
CLOSE_TIMEOUT_SECONDS = 15.0
# A pi-host answer slower than this is logged, so a stall names its call.
SLOW_SECONDS = 5.0

logger = logging.getLogger(__name__)
LABEL = "pi-host"
REPOSITORIES_ROOT = Path.home() / "git"
HOME = Path.home()
DEV_ROOT = HOME / "dev"
SCAN_DEPTH = 4
SCAN_SKIP = frozenset({"node_modules", ".cache", ".local", ".cargo", ".rustup", ".npm", ".venv", "venv", ".git", "__pycache__"})

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


# `coding`: a repository session with Pi's own tools and prompt. `document`: a workbench chat, the same Pi with the
# backend's prompt appended, restricted when the backend passes a tool allowlist.
# `sandboxed`: `coding` in an Agents-in-a-box container that reaches models only through the gateway.
Profile = Literal["coding", "document", "sandboxed"]


class LiveStatus(ContractModel):
    running: bool
    waiting: bool
    attached: int


class FactoryState(ContractModel):
    """A factory task's state as its session file last recorded it (a `factory-state` entry), `at` its timestamp."""

    state: Literal["running", "waiting_approve", "blocked", "merging", "done", "cancelled"]
    round: int
    reason: str
    at: str


class SessionInfo(ContractModel):
    session_id: str
    session_path: str
    cwd: str
    machine: Literal["local"]
    created_at: str
    modified_at: str
    first_message: str
    message_count: int
    # Context size after the last answer, 0 before one.
    tokens: int
    parent_session: str | None = None
    live: LiveStatus | None
    pinned: bool
    archived: bool
    # Sessions written before profiles existed are coding sessions.
    profile: Profile = "coding"
    # Set when the session is a factory task.
    factory: FactoryState | None = None


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


class ResumeById(ContractModel):
    """The session *session_id*, 404 `not_found` when pi-host knows none."""

    session_id: str


Resume = Literal["continue"] | ResumeFromFile | ResumeFork | ResumeById | None


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


class SessionEntries(ContractModel):
    """The persisted entries of a session file, pi's own shapes (`session-format.md`) passed through untouched."""

    entries: list[dict[str, Any]]
    leaf_id: str | None


class SearchResult(ContractModel):
    hits: list[SearchHit]


class PromptAnswer(ContractModel):
    """How Pi took a prompt: `handled` for an extension command, else Pi's own disposition of a message."""

    disposition: str


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


def gateway_token() -> str:
    """The gateway-only token (`/v1/*`) pi-host wrote on first start, for sandboxed callers. `PiHostUnavailable` when absent."""
    state_home = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    path = Path(os.environ.get("PI_GATEWAY_TOKEN_FILE") or Path(state_home) / "gigachad-bot" / "pi-gateway.token")
    try:
        value = path.read_text().strip()
    except OSError as exc:
        raise PiHostUnavailable(f"{LABEL} gateway token {path} is unreadable, is pi-host running? ({exc})") from exc
    if not value:
        raise PiHostUnavailable(f"{LABEL} gateway token {path} is empty")
    return value


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {token()}"}


def _body(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return response.text


async def _send(
    method: str,
    path: str,
    *,
    json: object = None,
    params: dict[str, str | int] | None = None,
    timeout: float = TIMEOUT_SECONDS,
) -> httpx.Response:
    """One authenticated request, `PiHostRejected` for an error status. A failure and a slow answer are logged with
    the call, its duration and the exception type."""
    headers = _auth_headers()
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(base_url=base_url(), timeout=timeout, headers=headers, transport=_transport) as client:
            response = await client.request(method, path, json=json, params=params)
    except httpx.HTTPError as exc:
        elapsed = time.monotonic() - started
        kind = type(exc).__name__
        logger.warning("%s %s %s failed after %.1f s (limit %.0f s): %s %s", LABEL, method, path, elapsed, timeout, kind, exc)
        raise PiHostUnavailable(
            f"{LABEL} at {base_url()} does not answer {method} {path} after {elapsed:.0f} s ({kind}: {exc})"
        ) from exc
    elapsed = time.monotonic() - started
    if elapsed > SLOW_SECONDS:
        logger.warning("%s %s %s took %.1f s", LABEL, method, path, elapsed)
    if response.is_error:
        raise PiHostRejected(response.status_code, error_message(response), _body(response))
    return response


async def _request[Result](
    method: str,
    path: str,
    result: type[Result],
    *,
    json: object = None,
    params: dict[str, str | int] | None = None,
    timeout: float = TIMEOUT_SECONDS,
) -> Result:
    response = await _send(method, path, json=json, params=params, timeout=timeout)
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
    cwd: str,
    resume: Resume,
    *,
    model: ModelChoice | None = None,
    thinking: str | None = None,
    profile: Profile = "coding",
    system_prompt: str | None = None,
    tools: list[str] | None = None,
) -> SessionInfo:
    """Spawn a *profile* session in *cwd*, or return the live one; a fork writes its file first. A session file held
    elsewhere answers 409 `locked`, an unknown fork source, entry or session id 404, a live session of the other
    profile 409 `profile_mismatch`.

    - `document` needs *system_prompt* (sent on every create and resume, Pi keeps it per process), *tools* restricts it
      to those tools and nothing discovered. `coding` takes neither
    """
    body: dict[str, object] = {
        "cwd": cwd,
        "profile": profile,
        "resume": resume.model_dump(by_alias=True) if isinstance(resume, ContractModel) else resume,
    }
    if system_prompt is not None:
        body["systemPrompt"] = system_prompt
    if tools is not None:
        body["tools"] = tools
    if model is not None:
        body["model"] = model.model_dump(by_alias=True)
    if thinking is not None:
        body["thinking"] = thinking
    return (await _request("POST", "/sessions", SessionAnswer, json=body, timeout=CHILD_TIMEOUT_SECONDS)).session


async def close_session(session_id: str) -> None:
    """Stop the session's Pi child, its file stays."""
    await _send("POST", f"/sessions/{session_id}/close", timeout=CLOSE_TIMEOUT_SECONDS)


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


async def session_entries(
    session_id: str, turns: int | None = None, *, custom_type: str | None = None, since: str | None = None
) -> SessionEntries:
    """Every persisted entry of the session in append order, read from its file; with *turns* only the active branch's
    last *turns* turns.

    - *custom_type* keeps only the `custom` entries whose `customType` starts with it, *since* those after that entry
      id (400 `invalid_since` for an unknown one). `leaf_id` is then the file's last entry. Neither goes with *turns*
    """
    given = {"turns": turns, "customType": custom_type, "since": since}
    params: dict[str, str | int] = {name: value for name, value in given.items() if value is not None}
    return await _request("GET", f"/sessions/{session_id}/entries", SessionEntries, params=params or None)


async def prompt_session(
    session_id: str, message: str, streaming_behavior: Literal["steer", "followUp"] | None = None
) -> str:
    """Send *message* to the live session as a Pi prompt, returns Pi's disposition (`handled` for an extension command).

    - A session that is not live answers 409 `not_live`, a prompt Pi refuses 502 `prompt_failed`
    - Pi refuses a prompt during a run unless *streaming_behavior* says how to queue it
    """
    body: dict[str, object] = {"message": message}
    if streaming_behavior is not None:
        body["streamingBehavior"] = streaming_behavior
    answer = await _request("POST", f"/sessions/{session_id}/prompt", PromptAnswer, json=body, timeout=CHILD_TIMEOUT_SECONDS)
    return answer.disposition


async def delete_session(session_id: str) -> None:
    """Remove the session file and its sidecar, stopping a live child first. 404 for an unknown session."""
    await _send("DELETE", f"/sessions/{session_id}")


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


def _is_folder(directory: Path) -> bool:
    return directory.is_dir() and not directory.name.startswith(".") and not directory.is_symlink()


def _git_repositories(directory: Path, depth: int) -> list[Path]:
    """Folders under *directory* holding a `.git`, not descending into a repository or `SCAN_SKIP`."""
    try:
        children = sorted(directory.iterdir(), key=lambda child: child.name.lower())
    except OSError:
        return []
    found: list[Path] = []
    for child in children:
        if child.name in SCAN_SKIP or child.is_symlink() or not child.is_dir():
            continue
        if (child / ".git").exists():
            found.append(child)
        elif depth > 1:
            found.extend(_git_repositories(child, depth - 1))
    return found


def repository_name(folder: Path) -> str:
    """The name `scan_repositories` gives *folder*: plain `<folder>` directly under `~/git`, else `<path from ~>`."""
    if folder.parent in (REPOSITORIES_ROOT, REPOSITORIES_ROOT.resolve()):
        return folder.name
    return str(folder.relative_to(HOME))


def scan_repositories() -> list[Repository]:
    """Walk the disk for where a session may start, by name: every folder directly under `~/git`, every folder
    directly under `~/dev` and every Git checkout found within `SCAN_DEPTH` levels of `~` (names: `repository_name`).

    - Takes seconds on a full home folder: requests read the stored list (`repository_index`), never this
    """
    paths: dict[str, Path] = {}
    for parent in (REPOSITORIES_ROOT, DEV_ROOT):
        if parent.is_dir():
            for child in sorted(parent.iterdir(), key=lambda child: child.name.lower()):
                if _is_folder(child):
                    paths.setdefault(repository_name(child), child)
    for checkout in _git_repositories(HOME, SCAN_DEPTH):
        if checkout.parent != REPOSITORIES_ROOT:  # those are listed above, unless hidden
            paths.setdefault(repository_name(checkout), checkout)
    return [Repository(name=name, path=str(path)) for name, path in paths.items()]
