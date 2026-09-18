"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def no_enqueue(monkeypatch: pytest.MonkeyPatch):
    """Replace extract_queue.enqueue with a recorder so tests never touch the
    real asyncio queue or run MinerU. Records the PDF name each call queued."""
    calls: list[str] = []
    monkeypatch.setattr(
        "lib.attachment_materialize.extract_queue.enqueue",
        lambda name, content, assets: calls.append(name),
    )
    return calls
