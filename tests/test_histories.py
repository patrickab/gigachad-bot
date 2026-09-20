from fastapi import HTTPException
import pytest

from backend.routes.histories import SaveRequest, save_chat_history
from backend.routes.projects import SaveTabRequest, save_project_tab


class ChatStoreStub:
    def __init__(self, events: list[tuple[str, str | None]]) -> None:
        self.events = events

    def save(self, filename, payload, *, expected_revision=None):
        self.events.append(("save", filename))
        return {"status": "ok", "filename": filename, "revision": "rev"}


class SandboxServiceStub:
    def __init__(self, events: list[tuple[str, str | None]]) -> None:
        self.events = events

    async def checkpoint(self, *, chat_id: str) -> None:
        self.events.append(("checkpoint", chat_id))


@pytest.mark.asyncio
async def test_history_save_checkpoints_staged_workspace_after_chat_save():
    events: list[tuple[str, str | None]] = []

    result = await save_chat_history(
        "history.json",
        SaveRequest(chat_id="chat-1", messages=[]),
        ChatStoreStub(events),
        SandboxServiceStub(events),
    )

    assert result["status"] == "ok"
    assert events == [("save", "history.json"), ("checkpoint", "chat-1")]


@pytest.mark.asyncio
async def test_history_save_without_chat_id_does_not_checkpoint():
    events: list[tuple[str, str | None]] = []

    await save_chat_history("history.json", SaveRequest(messages=[]), ChatStoreStub(events), SandboxServiceStub(events))

    assert events == [("save", "history.json")]


class ProjectStoreStub:
    def __init__(self, events: list[tuple[str, str | None]], *, error: Exception | None = None) -> None:
        self.events = events
        self.error = error

    def save_tab(self, slug, filename, data):
        self.events.append(("save_tab", f"{slug}/{filename}"))
        if self.error:
            raise self.error
        return {"status": "ok"}


@pytest.mark.asyncio
async def test_project_tab_save_checkpoints_staged_workspace_after_chat_save():
    events: list[tuple[str, str | None]] = []

    result = await save_project_tab(
        "proj",
        "tab.json",
        SaveTabRequest(filename="tab.json", chat_id="chat-1", messages=[]),
        ProjectStoreStub(events),
        SandboxServiceStub(events),
    )

    assert result["status"] == "ok"
    assert events == [("save_tab", "proj/tab.json"), ("checkpoint", "chat-1")]


@pytest.mark.asyncio
async def test_project_tab_save_failure_does_not_checkpoint():
    events: list[tuple[str, str | None]] = []

    with pytest.raises(HTTPException) as excinfo:
        await save_project_tab(
            "proj",
            "tab.json",
            SaveTabRequest(filename="tab.json", chat_id="chat-1", messages=[]),
            ProjectStoreStub(events, error=ValueError("stale revision")),
            SandboxServiceStub(events),
        )

    assert excinfo.value.status_code == 409
    assert events == [("save_tab", "proj/tab.json")]
