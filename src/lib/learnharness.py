"""Server-to-server client for LearnHarness, the learn mode's teaching surface.

- Unmodified upstream LearnHarness (`run-learnharness.sh`) on loopback, the browser frames it, every other read goes through here
- `GIGACHAD_LEARNHARNESS_URL` is LearnHarness for the backend (default 127.0.0.1:30181),
  `GIGACHAD_LEARNHARNESS_FRAME_URL` the origin the browser frames it from (the Tailnet gate in production)
- Learn sessions (section "Sessions"): list, create, first turn, rename, delete, Markdown export. Responses are validated
  against the contract models, a changed LearnHarness shape fails loudly with the field name
- Consumes `lib.upstream` for the HTTP client and error types, provides the URL helpers, the contract models and the session calls
- The model configuration generator lives in `lib.learnharness_config`
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

import httpx
from pydantic import AliasGenerator, BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from lib.upstream import (
    TIMEOUT_SECONDS,
    UpstreamClient,
    UpstreamContractError,
    UpstreamRequestFailed,
    UpstreamUnavailable,
    error_message,
    upstream_url,
)

DEFAULT_URL = "http://127.0.0.1:30181"

client = UpstreamClient(env="GIGACHAD_LEARNHARNESS_URL", default=DEFAULT_URL, label="LearnHarness")


def learnharness_url() -> str:
    """Base URL the backend reaches LearnHarness at."""
    return upstream_url("GIGACHAD_LEARNHARNESS_URL", DEFAULT_URL)


def learnharness_frame_url() -> str:
    """Base URL the browser frames LearnHarness from, `GIGACHAD_LEARNHARNESS_FRAME_URL` when it sits behind the Tailnet gate."""
    return upstream_url("GIGACHAD_LEARNHARNESS_FRAME_URL", learnharness_url())


# ── Sessions ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# LearnHarness's chat API (`app/api/chats`, `app/api/chat`, v0.4.0) is internal and churns, so every URL and shape lives here.

# A turn is acknowledged once the turn gates passed, which can wait for a running context compaction.
TURN_START_TIMEOUT = httpx.Timeout(10.0, read=60.0)
# Deleting a session waits up to 10 s for its extraction to wind down.
DELETE_TIMEOUT_SECONDS = 30.0
# LearnHarness's error text for an unknown session (`deleteChatSession`); any other 404 is a moved or removed route.
NO_SUCH_SESSION = "No such session"


class ContractModel(BaseModel):
    """LearnHarness JSON is camelCase, gigachad's API is snake_case: validate by alias, serialize by field name."""

    model_config = ConfigDict(alias_generator=AliasGenerator(validation_alias=to_camel), extra="ignore", frozen=True)


class LearnSession(ContractModel):
    """A learn session: LearnHarness's `ChatListItem`, reduced to what the shell shows (its sidebar row's state included)."""

    id: str
    title: str
    # LearnHarness's session kind, "teach" or "practice"; a freshly created session reports none.
    kind: str | None = None
    # A teach turn is running.
    busy: bool = False
    # A context compaction runs.
    compacting: bool = False
    # A question waits for the learner's answer.
    pending_question: bool = False
    # Practice: a posed problem waits for the learner.
    pending_problem: bool = False
    # When the session last changed (ISO time), None for a freshly created one.
    updated_at: str | None = None


# LearnHarness's two deletes: the session goes, its Knowledge stays in the graph, or its graph traces go too.
DeleteMode = Literal["preserve-graph", "delete-nodes"]


@dataclass(frozen=True)
class SessionExport:
    """A session as LearnHarness exports it: one Markdown document and the file name it suggests."""

    filename: str
    markdown: str


class _SessionList(ContractModel):
    chats: list[LearnSession]


class _Deleted(ContractModel):
    ok: bool


class _Renamed(ContractModel):
    title: str


class _Accepted(ContractModel):
    """LearnHarness's acknowledgement of a turn: `POST /api/chat` answers `{ok: true}` and closes, the turn runs on."""

    ok: bool


async def list_sessions() -> list[LearnSession]:
    """Every learn session, most recently updated first."""
    return (await client.request("GET", "/api/chats", _SessionList)).chats


async def create_session(title: str) -> LearnSession:
    """A new, empty learn session. LearnHarness stamps it with its current strong model tier."""
    return await client.request("POST", "/api/chats", LearnSession, json={"title": title})


async def submit_text_turn(session_id: str, text: str) -> None:
    """Start a turn on *text* in session *session_id*.

    Returns once LearnHarness acknowledged the turn with JSON `{ok: true}`. The turn runs on without a reader.
    A refusal (unknown session, no model configured, a turn already running) raises `UpstreamRequestFailed` with its message,
    any other 2xx body `UpstreamContractError`.
    """
    body = {"chatId": session_id, "input": {"type": "text", "text": text}}
    await client.request("POST", "/api/chat", _Accepted, json=body, timeout=TURN_START_TIMEOUT)


async def rename_session(session_id: str, title: str) -> str:
    """Rename session *session_id* to *title*, returning the title LearnHarness stored (trimmed).

    LearnHarness keeps the session's place in its list. It refuses a blank title (400) and an unknown session (404,
    `UpstreamRequestFailed`).
    """
    return (await client.request("PUT", f"/api/chats/{session_id}", _Renamed, json={"title": title})).title


async def delete_session(session_id: str, mode: DeleteMode = "preserve-graph") -> None:
    """Delete a learn session and its jobs. An unknown session counts as deleted.

    *mode* `preserve-graph` keeps its Knowledge in the graph, `delete-nodes` removes its graph traces too (topics only
    this session created go, shared ones stay). LearnHarness refuses (409) while a turn or background job of the session
    runs. A 404 without LearnHarness's "No such session" (a moved route, another app on the URL) raises
    `UpstreamContractError`: nothing counts as deleted that was not.
    """
    path = f"/api/chats/{session_id}"
    try:
        await client.request("DELETE", path, _Deleted, params={"mode": mode}, timeout=DELETE_TIMEOUT_SECONDS)
    except UpstreamRequestFailed as exc:
        if exc.status != 404:
            raise
        if str(exc) != NO_SUCH_SESSION:
            raise UpstreamContractError(f'LearnHarness DELETE {path}: 404 without "{NO_SUCH_SESSION}" ({exc})') from exc


# `filename="<slug>.md"` of LearnHarness's export `Content-Disposition`.
_EXPORT_FILENAME = re.compile(r'filename="([^"/\\]+)"')


async def export_markdown(session_id: str) -> SessionExport:
    """The whole session as one Markdown document (`GET /api/chats/<id>/markdown`, the sidebar's "Export as Markdown").

    An unknown session raises `UpstreamRequestFailed` (404), a body that is no Markdown attachment `UpstreamContractError`.
    """
    path = f"/api/chats/{session_id}/markdown"
    try:
        async with httpx.AsyncClient(base_url=client.url, timeout=TIMEOUT_SECONDS, transport=client.transport) as http:
            response = await http.get(path)
    except httpx.HTTPError as exc:
        raise UpstreamUnavailable(f"LearnHarness at {client.url} does not answer: {exc}") from exc
    if response.is_error:
        raise UpstreamRequestFailed(response.status_code, error_message(response))
    filename = _EXPORT_FILENAME.search(response.headers.get("content-disposition", ""))
    if not response.headers.get("content-type", "").startswith("text/markdown") or filename is None:
        raise UpstreamContractError(f"LearnHarness {path}: no Markdown attachment")
    return SessionExport(filename=filename.group(1), markdown=response.text)
