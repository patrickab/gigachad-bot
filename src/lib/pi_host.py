"""Server-to-server client for pi-host, the Bun service that supervises Pi sessions and serves the completion gateway.

- The only module that knows pi-host (D6): its URL, its shared bearer token, its HTTP shapes and its `/ws` endpoint
- Consumes pi-host's HTTP API (`.docs/pi-migration-contracts.md`, "pi-host HTTP") and the token file pi-host creates on start
- Provides the session, file and private-state calls of `/api/pi/*` and the workbench chats (`document` profile with the
  backend's system prompt, P3), a WebSocket opener for the `/ws` proxy and the gateway URL
- Provides the factory core's actions (`factory_call`, `factory_actions`): pi-host runs them in the agent-factory
  package's `core.ts`, a refusal is `FactoryCallError` with the core's code, a missing package `FactoryPluginMissing`
- Provides `media`: the content-addressed tool-result images pi-host stores (`pi-host/src/media.ts`)
- `SessionInfo.factory`: a session that is a factory task carries its state, so the session picker can hide it
- Answers are validated against the contract models below, a changed shape fails loudly with the field name
- Failures are `PiHostUnavailable` (503), `PiHostBadResponse` (502) or `PiHostRejected` (pi-host's own status and body)
- Also scans the disk for the repositories a session may start in (`scan_repositories`, stored by `repository_index`)
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time
from typing import Any, Literal, cast

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


# The factory is a package of its own (CLI `factory`, Pi extension), not part of this repo.
FACTORY_PLUGIN_SOURCE = "https://github.com/patrickab/agent-factory"
FACTORY_PLUGIN_HINT = f"Install the factory (its `factory` CLI on pi-host's PATH) from {FACTORY_PLUGIN_SOURCE}"


class FactoryPluginMissing(RuntimeError):
    """Pi has no factory plugin: the message names where it comes from."""


FactoryErrorCode = Literal["invalid", "not_found", "busy", "refused", "failed"]
FACTORY_ERROR_CODES: tuple[FactoryErrorCode, ...] = ("invalid", "not_found", "busy", "refused", "failed")
# `tasks.start` creates a worktree and runs its `after_create` hooks (10 min each in the plugin) before it answers.
FACTORY_CALL_TIMEOUT_SECONDS = 900.0


class FactoryCallError(RuntimeError):
    """The factory core refused an action: `code` is the core's (`FACTORY_ERROR_CODES`), the message one sentence,
    `detail` the core's structured hint when it gives one (`{"force": true}`: `force` would allow the refused removal)."""

    def __init__(self, code: FactoryErrorCode, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail


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


class SessionUsage(ContractModel):
    """What a session's models used: its assistant answers plus the reviews a factory task recorded. `input` counts cache
    reads and writes too, `cost` is in dollars."""

    input: int
    output: int
    cost: float


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
    # Set once a model answered.
    usage: SessionUsage | None = None
    # Until when the prompt cache is likely warm (ISO), set when the last answer read or wrote one.
    cached_until: str | None = None


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


class SessionMatch(ContractModel):
    """One hit in a session: the *ordinal*th match (text order) in the user or assistant text of entry *entry_id*,
    with the text around it."""

    entry_id: str
    ordinal: int
    quality: float
    before: str
    match: str
    after: str


class SessionMatches(ContractModel):
    """A session's best hits, ranked, and how many it has in all."""

    matches: list[SessionMatch]
    total: int


class SearchHit(ContractModel):
    """A session matching a transcript search (M24): the text around the hit and the entry holding it, and its best
    few hits."""

    session: SessionInfo
    snippet: str
    entry_id: str
    matches: list[SessionMatch] = []


class SessionEntries(ContractModel):
    """The persisted entries of a session file, pi's own shapes (`session-format.md`) passed through untouched."""

    entries: list[dict[str, Any]]
    leaf_id: str | None


class SearchResult(ContractModel):
    hits: list[SearchHit]


class TurnDeleted(ContractModel):
    """What a turn delete removed: session entries, and the sha256 of the tool images whose files went. `not_removed`: the
    pastes and media hashes a cleanup after the rewrite could not remove, still on disk."""

    removed_entries: int
    removed_media: list[str]
    not_removed: list[str] = []


class PromptAnswer(ContractModel):
    """How Pi took a prompt: `handled` for an extension command, else Pi's own disposition of a message."""

    disposition: str


class PiSettings(ContractModel):
    """pi-host's own settings: `auto_resume` reopens runs a restart interrupted (off by default)."""

    auto_resume: bool


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


async def media(media_hash: str) -> tuple[bytes, str]:
    """The tool image stored under *media_hash* (sha256 of its bytes) and its content type. `PiHostRejected` 404 if unknown."""
    response = await _send("GET", f"/media/{media_hash}")
    return response.content, response.headers.get("content-type", "application/octet-stream")


async def health() -> Health:
    return await _request("GET", "/health", Health)


async def get_settings() -> PiSettings:
    return await _request("GET", "/settings", PiSettings)


async def set_settings(*, auto_resume: bool) -> PiSettings:
    return await _request("PUT", "/settings", PiSettings, json={"autoResume": auto_resume})


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


def _factory_failure(exc: PiHostRejected) -> Exception:
    """pi-host's refusal of a factory call as the core's error, a missing plugin as `FactoryPluginMissing`; any other
    refusal stays as it is."""
    body = exc.body if isinstance(exc.body, dict) else {}
    code, message = body.get("error"), str(body.get("message") or exc)
    if code == "factory_plugin_missing":
        return FactoryPluginMissing(str(body.get("message") or FACTORY_PLUGIN_HINT))
    if code in FACTORY_ERROR_CODES:
        detail = body.get("detail")
        return FactoryCallError(cast("FactoryErrorCode", code), message, detail if isinstance(detail, dict) else None)
    return exc


async def _factory_answer(method: str, path: str, key: str, *, json: object = None, timeout: float = TIMEOUT_SECONDS) -> Any:  # noqa: ANN401 - core JSON
    """The value under *key* of pi-host's answer to a factory call."""
    try:
        response = await _send(method, path, json=json, timeout=timeout)
    except PiHostRejected as exc:
        raise _factory_failure(exc) from exc
    try:
        answer = response.json()
    except ValueError as exc:
        raise PiHostBadResponse(f"{LABEL} {path} answered no JSON") from exc
    if not isinstance(answer, dict) or key not in answer:
        raise PiHostBadResponse(f"{LABEL} {path} answered without {key!r}")
    return answer[key]


async def factory_call(action: str, params: dict[str, Any]) -> Any:  # noqa: ANN401 - the core's JSON, as each action shapes it
    """Run the factory core's *action* with *params* and return its result, plain JSON as the core gives it.

    `FactoryCallError` when the core refuses, `FactoryPluginMissing` when Pi has no factory plugin.
    """
    body = {"action": action, "params": params}
    return await _factory_answer("POST", "/factory/call", "result", json=body, timeout=FACTORY_CALL_TIMEOUT_SECONDS)


async def factory_actions() -> dict[str, Any]:
    """The factory core's `ACTIONS`: each action's summary, JSON schema of its params and whether it changes anything."""
    actions = await _factory_answer("GET", "/factory/actions", "actions")
    if not isinstance(actions, dict):
        raise PiHostBadResponse(f"{LABEL} /factory/actions answered actions that are no object")
    return actions


async def session_entries(
    session_id: str,
    turns: int | None = None,
    *,
    custom_type: str | None = None,
    since: str | None = None,
    assistant: bool = False,
) -> SessionEntries:
    """Every persisted entry of the session in append order, read from its file; with *turns* only the active branch's
    last *turns* turns.

    - *custom_type* keeps only the `custom` entries whose `customType` starts with it, *since* those after that entry
      id (400 `invalid_since` for an unknown one). `leaf_id` is then the file's last entry. Neither goes with *turns*
    - *assistant* adds the assistant answers among them, cut to `text`, `model`, `stopReason` and `usage`
    """
    given = {"turns": turns, "customType": custom_type, "since": since, "assistant": "1" if assistant else None}
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


async def delete_turn(session_id: str, entry_id: str, through: Literal["turn", "end"]) -> TurnDeleted:
    """Remove the turn that *entry_id* (a user prompt) starts from the session file: `turn` that turn, `end` it and all later ones.

    pi-host stops the child, rewrites the file and starts it again, so the timeout is the child's. 409 `running`, `compacted`,
    `locked` or `changed`, 404 unknown session or entry, 400 not a prompt: `PiHostRejected`.
    """
    path = f"/sessions/{session_id}/turns/{entry_id}/delete"
    return await _request("POST", path, TurnDeleted, json={"through": through}, timeout=CHILD_TIMEOUT_SECONDS)


async def search_sessions(query: str, limit: int = 20) -> list[SearchHit]:
    """Live sessions whose user and assistant text matches *query*, best first, newest first on ties (M24)."""
    params: dict[str, str | int] = {"q": query, "limit": limit}
    return (await _request("GET", "/search", SearchResult, params=params)).hits


async def session_matches(session_id: str, query: str, limit: int = 50) -> SessionMatches:
    """The best *limit* hits of one session's transcript for *query*, ranked, each addressed by entry id and ordinal."""
    params: dict[str, str | int] = {"q": query, "limit": limit}
    return await _request("GET", f"/sessions/{session_id}/matches", SessionMatches, params=params)


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
    """The name `scan_repositories` gives *folder*: plain `<folder>` directly under `~/git` or `~/dev`, else
    `<path from ~>`, the absolute path outside `~`."""
    if folder.parent in (REPOSITORIES_ROOT, REPOSITORIES_ROOT.resolve(), DEV_ROOT, DEV_ROOT.resolve()):
        return folder.name
    return str(folder.relative_to(HOME)) if folder.is_relative_to(HOME) else str(folder)


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
        if checkout.parent not in (REPOSITORIES_ROOT, DEV_ROOT):  # those are listed above, unless hidden
            paths.setdefault(repository_name(checkout), checkout)
    return [Repository(name=name, path=str(path)) for name, path in paths.items()]
