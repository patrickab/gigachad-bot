import json
from pathlib import Path

import pytest

from lib import factory_sessions

HEADER = {"type": "session", "version": 3, "id": "abc-1", "timestamp": "2026-01-02T03:04:05.678Z", "cwd": "/runtime/active"}


def _session(*entries: str) -> str:
    return "\n".join([json.dumps(HEADER), *entries]) + "\n"


@pytest.fixture
def agent_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("PI_CODING_AGENT_SESSION_DIR", raising=False)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))
    return tmp_path


def test_import_rehomes_the_session_into_the_worktrees_pi_folder(agent_dir: Path) -> None:
    path = factory_sessions.import_session(_session('{"type":"message"}'), "/home/u/git/worktrees/repo/t1")
    assert path == agent_dir / "sessions" / "--home-u-git-worktrees-repo-t1--" / "2026-01-02T03-04-05-678Z_abc-1.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["cwd"] == "/home/u/git/worktrees/repo/t1"
    assert lines[1] == '{"type":"message"}'


def test_import_fast_forwards_a_grown_session_and_keeps_a_continued_copy(agent_dir: Path) -> None:
    cwd = "/w"
    path = factory_sessions.import_session(_session("1"), cwd)
    factory_sessions.import_session(_session("1", "2"), cwd)
    assert path.read_text(encoding="utf-8").splitlines()[1:] == ["1", "2"]
    path.write_text(path.read_text(encoding="utf-8") + "mine\n", encoding="utf-8")
    factory_sessions.import_session(_session("1", "2", "3", "4"), cwd)
    assert path.read_text(encoding="utf-8").splitlines()[1:] == ["1", "2", "mine"]


@pytest.mark.parametrize("content", ["", '{"type":"message"}\n', json.dumps({**HEADER, "id": "../x"})])
def test_import_refuses_a_file_without_a_valid_header(agent_dir: Path, content: str) -> None:
    with pytest.raises(ValueError):
        factory_sessions.import_session(content, "/w")
