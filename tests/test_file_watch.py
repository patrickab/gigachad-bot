import asyncio
import json
import os
import threading
from pathlib import Path

import pytest
from anyio import to_thread

from backend import websocket_origin
from backend.routes import file_watch as file_watch_route
from backend.routes.deps import get_file_roots
from lib.file_roots import FileRoots
from tests.conftest import mounted_roots, route_client

# Longer than the debounce plus the watcher's polling step, so a silence this long means nothing more is coming.
QUIET = 0.3


@pytest.fixture(autouse=True)
def trusted_origins(monkeypatch):
    monkeypatch.setattr(websocket_origin, "CORS_ORIGINS", ["*"])


@pytest.fixture
def vault(tmp_path) -> Path:
    path = tmp_path / "vault"
    path.mkdir()
    return path


class Socket:
    """A websocket client of the route's app that runs on the test's loop, so the test sees the tasks the route leaves behind."""

    def __init__(self, roots: FileRoots, *, origin: str | None = "http://localhost:2999") -> None:
        app = route_client(file_watch_route.router, overrides={get_file_roots: roots}).app
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "scheme": "ws",
            "path": "/api/files/watch",
            "raw_path": b"/api/files/watch",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"origin", origin.encode())] if origin else [],
            "client": ("127.0.0.1", 50000),
            "server": ("testserver", 80),
            "subprotocols": [],
        }
        self._to_app: asyncio.Queue[dict] = asyncio.Queue()
        self._from_app: asyncio.Queue[dict] = asyncio.Queue()
        self._to_app.put_nowait({"type": "websocket.connect"})
        self._app = asyncio.create_task(app(scope, self._to_app.get, self._from_app.put))

    async def __aenter__(self) -> "Socket":
        return self

    async def __aexit__(self, *exception) -> None:
        self._to_app.put_nowait({"type": "websocket.disconnect", "code": 1000})
        await asyncio.wait_for(self._app, 5)

    def send(self, text: str) -> None:
        self._to_app.put_nowait({"type": "websocket.receive", "text": text})

    async def next_message(self) -> dict:
        return await asyncio.wait_for(self._from_app.get(), 5)

    async def collect(self) -> list[dict]:
        """The JSON frames the route sends until it has been silent for `QUIET`."""
        frames = []
        while True:
            try:
                message = await asyncio.wait_for(self._from_app.get(), QUIET)
            except TimeoutError:
                return frames
            if message["type"] == "websocket.send":
                frames.append(json.loads(message["text"]))

    async def watch(self, files: list[Path | str] = (), folders: list[Path | str] = ()) -> list[dict]:
        """Replace the watch set, answering what the route sent back."""
        paths = {"files": [str(file) for file in files], "folders": [str(folder) for folder in folders]}
        self.send(json.dumps({"type": "watch", **paths}))
        return await self.collect()

    async def running(self, probe: Path) -> list[dict]:
        """Rewrite the watched *probe* until the route reports a change, which shows its watch is running and quiet."""
        for attempt in range(20):
            probe.write_text(f"probe {attempt}")
            if frames := await self.collect():
                return frames
        pytest.fail(f"{probe} was never reported")


def watch_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == "file-watch"]


def changed(frames: list[dict]) -> set[str]:
    return {frame["path"] for frame in frames if frame["type"] == "changed"}


async def test_modified_file_reports_its_path(vault):
    note = vault / "note.md"
    note.write_text("one")
    async with Socket(mounted_roots(vault)) as socket:
        await socket.watch(files=[note])
        await socket.running(note)

        note.write_text("two")

        assert changed(await socket.collect()) == {str(note)}


async def test_atomic_save_reports_the_replaced_file(vault):
    note = vault / "note.md"
    note.write_text("one")
    async with Socket(mounted_roots(vault)) as socket:
        await socket.watch(files=[note])
        await socket.running(note)

        temporary = vault / ".note.md.tmp"
        temporary.write_text("two")
        os.replace(temporary, note)

        assert changed(await socket.collect()) == {str(note)}
    assert note.read_text() == "two"


async def test_deleted_file_reports_its_path(vault):
    note = vault / "note.md"
    note.write_text("one")
    async with Socket(mounted_roots(vault)) as socket:
        await socket.watch(files=[note])
        await socket.running(note)

        note.unlink()

        assert changed(await socket.collect()) == {str(note)}


async def test_file_created_later_is_reported(vault):
    later = vault / "later.md"
    async with Socket(mounted_roots(vault)) as socket:
        await socket.watch(files=[later])

        assert changed(await socket.running(later)) == {str(later)}


async def test_unwatched_sibling_is_not_reported(vault):
    note = vault / "note.md"
    sibling = vault / "sibling.md"
    note.write_text("one")
    async with Socket(mounted_roots(vault)) as socket:
        await socket.watch(files=[note])
        await socket.running(note)

        sibling.write_text("other")
        note.write_text("two")

        assert changed(await socket.collect()) == {str(note)}


async def test_folder_reports_nested_files_but_not_build_output_internals_or_editor_scratch(vault):
    project = vault / "paper"
    (project / "chapters").mkdir(parents=True)
    for internal in (".latex-build", ".git", "node_modules/pkg"):
        (project / internal).mkdir(parents=True)
    (project / "main.tex").write_text(r"\documentclass{article}")
    intro = project / "chapters" / "intro.tex"
    async with Socket(mounted_roots(vault)) as socket:
        await socket.watch(folders=[project])
        await socket.running(project / "main.tex")

        intro.write_text("intro")
        (project / "main.aux").write_text("aux")
        (project / "main.view.pdf").write_text("pdf")
        (project / ".latex-build" / "main.pdf").write_text("pdf")
        (project / ".git" / "HEAD").write_text("ref")
        (project / "node_modules" / "pkg" / "index.js").write_text("js")
        (project / "chapters" / "appendix").mkdir()
        for scratch in ("4913", ".intro.tex.swp", ".intro.tex.swo", "intro.tex~"):
            (project / "chapters" / scratch).write_text("scratch")

        assert changed(await socket.collect()) == {str(intro)}


async def test_new_watch_set_replaces_the_old_one(vault):
    old = vault / "old.md"
    new = vault / "new.md"
    async with Socket(mounted_roots(vault)) as socket:
        await socket.watch(files=[old])
        await socket.running(old)

        await socket.watch(files=[new])
        await socket.running(new)

        old.write_text("changed")
        new.write_text("changed")

        assert changed(await socket.collect()) == {str(new)}


async def test_empty_watch_set_stops_everything(vault):
    note = vault / "note.md"
    async with Socket(mounted_roots(vault)) as socket:
        await socket.watch(files=[note])
        await socket.running(note)

        await socket.watch()
        note.write_text("changed")

        assert await socket.collect() == []


async def test_paths_outside_the_roots_are_refused_and_the_rest_is_watched(vault, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    secret = elsewhere / "secret.md"
    note = vault / "note.md"
    async with Socket(mounted_roots(vault)) as socket:
        answers = await socket.watch(
            files=[secret, note, "relative.md", vault / "missing" / "note.md"], folders=[elsewhere, vault / "missing", note]
        )

        refused = {answer["path"]: answer["message"] for answer in answers if answer["type"] == "refused"}
        assert set(refused) == {
            str(secret),
            "relative.md",
            str(vault / "missing" / "note.md"),
            str(elsewhere),
            str(vault / "missing"),
            str(note),
        }
        assert "outside file roots" in refused[str(secret)]
        assert "outside file roots" in refused[str(elsewhere)]
        assert all(message for message in refused.values())

        await socket.running(note)
        secret.write_text("changed")
        note.write_text("changed")

        assert changed(await socket.collect()) == {str(note)}


async def test_paths_are_reported_as_the_client_named_them(vault, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(vault)
    project = vault / "paper"
    project.mkdir()
    note = vault / "note.md"
    (project / "main.tex").write_text("main")
    async with Socket(mounted_roots(vault)) as socket:
        await socket.watch(files=[link / "note.md"], folders=[link / "paper"])
        await socket.running(note)
        await socket.running(project / "main.tex")

        note.write_text("changed")
        (project / "main.tex").write_text("changed")

        assert changed(await socket.collect()) == {str(link / "note.md"), str(link / "paper" / "main.tex")}


async def test_malformed_frames_do_not_end_the_socket(vault):
    note = vault / "note.md"
    async with Socket(mounted_roots(vault)) as socket:
        socket.send("not json")
        socket.send(json.dumps({"type": "unknown"}))
        socket.send(json.dumps({"type": "watch", "files": "note.md"}))
        await socket.watch(files=[note])

        assert changed(await socket.running(note)) == {str(note)}


@pytest.mark.parametrize("origin", [None, "https://evil.example"])
async def test_untrusted_origin_is_refused(vault, origin):
    async with Socket(mounted_roots(vault), origin=origin) as socket:
        refusal = await socket.next_message()

    assert (refusal["type"], refusal["code"]) == ("websocket.close", 1008)


async def test_closing_the_socket_stops_the_watcher_and_its_threads(vault):
    note = vault / "note.md"
    tasks_before = asyncio.all_tasks()
    async with Socket(mounted_roots(vault)) as socket:
        await socket.watch(files=[note], folders=[vault])
        await socket.running(note)
        assert watch_threads()

    assert asyncio.all_tasks() == tasks_before
    assert not watch_threads()


async def test_watching_leaves_the_shared_thread_pool_free(vault):
    limiter = to_thread.current_default_thread_limiter()
    note = vault / "note.md"
    async with Socket(mounted_roots(vault)) as socket:
        await socket.watch(files=[note], folders=[vault])
        await socket.running(note)

        assert limiter.borrowed_tokens == 0
