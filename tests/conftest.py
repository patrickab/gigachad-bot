"""Shared pytest fixtures.

The attach/materialize core reads from ``config.DIRECTORY_OUTPUT_MINERU`` which
defaults to ``~/Nextcloud/linux/Documents/Mineru`` — that directory does not
exist on CI or dev machines without the cloud mount. We point it at a
per-test tmp_path by monkeypatching the *module attribute* the helpers import.

Why monkeypatch the module attribute and not ``config.DIRECTORY_OUTPUT_MINERU``
itself: ``attachment_materialize`` does ``from config import
DIRECTORY_OUTPUT_MINERU`` which binds a local reference, so patching the
``config`` module won't affect it. We patch the reference inside the modules
that actually read it.
"""

from pathlib import Path

import pytest


@pytest.fixture
def mineru_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the MinerU cache directory at a tmp dir for the duration of a test."""
    cache_dir = tmp_path / "mineru"
    cache_dir.mkdir()
    # The cache path helper imports DIRECTORY_OUTPUT_MINERU at module load time,
    # so we patch the already-bound reference inside the helper module.
    from lib import attachment_materialize

    monkeypatch.setattr(attachment_materialize, "DIRECTORY_OUTPUT_MINERU", cache_dir, raising=True)
    # extract_queue imports the helper lazily inside enqueue/_worker, so it reads
    # the patched attribute via the module reference at call time — no patch needed.
    return cache_dir


@pytest.fixture
def no_enqueue(monkeypatch: pytest.MonkeyPatch):
    """Replace extract_queue.enqueue with a recorder so tests never touch the real asyncio queue."""
    calls: list[Path] = []
    monkeypatch.setattr("lib.attachment_materialize.extract_queue.enqueue", lambda p: calls.append(p))
    return calls
