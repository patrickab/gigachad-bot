import pytest

from backend.routes.histories import SaveRequest, save_chat_history


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
