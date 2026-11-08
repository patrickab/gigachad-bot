"""Shared pytest fixtures."""

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from psycopg_pool import ConnectionPool
import pytest

from backend.identity import RequestIdentity, get_request_identity
from backend.routes import latex as latex_route
from backend.routes.deps import get_file_roots, get_user_scope
from lib import latex_templates
from lib.asset_store import AssetStore
from lib.changes import UserScope
from lib.db_schema import upgrade
from lib.file_roots import FileRoots
from lib.postgres_data_store import PostgresDataStore
from lib.process_owner import OWNER_VARIABLE, reap_orphans
from tests.fakes import RecordingScope

TRUNCATE_ALL = "TRUNCATE changes, assets, vault_roots, devices, documents, users, model_reasoning_levels CASCADE"
SRC = Path(__file__).resolve().parents[1] / "src"
requires_proc = pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="reads processes from /proc (Linux)")


def create_user(pool: ConnectionPool, login: str | None = None) -> UUID:
    """Insert a user (a unique throwaway login by default) and return its id."""
    login = login or f"test-{uuid4()}@example.test"
    with pool.connection() as connection, connection.transaction():
        return connection.execute(
            "INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", (login,)
        ).fetchone()[0]


def create_device(pool: ConnectionPool, user_id: UUID) -> UUID:
    device_id = uuid4()
    with pool.connection() as connection, connection.transaction():
        connection.execute("INSERT INTO devices (id, user_id) VALUES (%s, %s)", (device_id, user_id))
    return device_id


def create_user_and_device(pool: ConnectionPool, login: str = "alice@example.test") -> tuple[UUID, UUID]:
    user_id = create_user(pool, login)
    return user_id, create_device(pool, user_id)


def git(directory, *args: str) -> str:
    """Run git in *directory* with a fixed committer identity; returns stripped stdout."""
    return subprocess.run(
        ["git", "-C", str(directory), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def make_asset_store(pool: ConnectionPool, login: str = "alice@example.test") -> tuple[AssetStore, UUID]:
    """Register *login* with one device; return that user's asset store and the device id."""
    user_id, device_id = create_user_and_device(pool, login)
    return AssetStore(UserScope(pool, user_id, device_id)), device_id


def mounted_roots(*directories: Path | str, host_access: bool = False) -> FileRoots:
    """File roots whose saved mounts are *directories*; saving more mounts is discarded."""
    repository = SimpleNamespace(load=lambda: [str(directory) for directory in directories], save=lambda roots: None)
    return FileRoots(repository=repository, host_access=host_access)


def request_identity(has_host_access: bool = True) -> RequestIdentity:
    """A fresh identity for me@example.test without a device."""
    return RequestIdentity(uuid4(), "me@example.test", None, has_host_access)


def _returning(value: Any) -> Callable[[], Any]:
    return lambda: value


def route_client(*routers: APIRouter, overrides: Mapping[Callable[..., Any], Any] | None = None) -> TestClient:
    """A client for *routers* mounted on a bare app, each dependency in *overrides* answering its fixed value."""
    app = FastAPI()
    for router in routers:
        app.include_router(router)
    for dependency, value in (overrides or {}).items():
        app.dependency_overrides[dependency] = _returning(value)
    return TestClient(app)


@pytest.fixture(scope="module")
def database_url() -> str:
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for Postgres-backed tests")
    upgrade(url)
    return url


@pytest.fixture(scope="module")
def postgres_pool(database_url: str):
    pool = ConnectionPool(database_url, min_size=1, max_size=4)
    yield pool
    pool.close()


@pytest.fixture
def clean_database(postgres_pool: ConnectionPool) -> None:
    """Empty every user-owned table. Modules that want it for every test apply it with
    ``pytestmark = pytest.mark.usefixtures("clean_database")``."""
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute(TRUNCATE_ALL)


@pytest.fixture
def data_store(postgres_pool: ConnectionPool, clean_database: None) -> PostgresDataStore:
    """A new user's document store in an emptied database."""
    return PostgresDataStore(UserScope(postgres_pool, create_user(postgres_pool)))


@pytest.fixture
def asset_store(postgres_pool: ConnectionPool, clean_database: None) -> AssetStore:
    """alice's asset store in an emptied database."""
    return make_asset_store(postgres_pool)[0]


@pytest.fixture
def documents_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the Nextcloud mirror root at a tmp dir; ``attachment_materialize`` reads it at call time."""
    import lib.attachment_materialize as attachment_materialize

    root = tmp_path / "Documents"
    monkeypatch.setattr(attachment_materialize, "DOCUMENTS", root, raising=True)
    return root


@pytest.fixture
def no_enqueue(monkeypatch: pytest.MonkeyPatch):
    """Replace extract_queue.enqueue with a recorder so tests never touch the
    real asyncio queue or run MinerU. Records the PDF name each call queued."""
    calls: list[str] = []
    monkeypatch.setattr(
        "lib.attachment_materialize.extract_queue.enqueue",
        lambda name, content, store: calls.append(name),
    )
    return calls


def is_running(pid: int) -> bool:
    """Whether *pid* runs, a zombie waiting for its parent does not."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat[stat.rindex(")") + 2] != "Z"


def still_running(pids: list[int], seconds: float) -> list[int]:
    """The *pids* that still run after waiting up to *seconds* for all of them to end."""
    deadline = time.monotonic() + seconds
    while (running := [pid for pid in pids if is_running(pid)]) and time.monotonic() < deadline:
        time.sleep(0.1)
    return running


@pytest.fixture
def backend_process():
    """`start(script, env=None)` runs *script* from `src` as an unmarked backend-like parent, killed and reaped at teardown."""
    started: list[subprocess.Popen] = []

    def start(script: str, env: Mapping[str, str] | None = None) -> tuple[subprocess.Popen, Any]:
        environment = {name: value for name, value in (env or os.environ).items() if name != OWNER_VARIABLE}
        process = subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", textwrap.dedent(script)], cwd=SRC, env=environment, stdout=subprocess.PIPE, text=True
        )
        started.append(process)
        line = process.stdout.readline()
        assert line, f"the backend exited with {process.wait()} before it printed"
        return process, json.loads(line)

    yield start
    for process in started:
        process.kill()
        process.wait()
    reap_orphans()


@pytest.fixture
def scope():
    return RecordingScope()


@pytest.fixture
def vault(tmp_path):
    """The only mount of `tex_client`."""
    path = tmp_path / "vault"
    path.mkdir()
    return path


@pytest.fixture
def tex_client(vault, scope):
    """A host-access client whose only mount is `vault`."""
    return route_client(
        latex_route.router,
        overrides={
            get_request_identity: request_identity(),
            get_file_roots: mounted_roots(vault),
            get_user_scope: scope,
        },
    )


@pytest.fixture
def document(vault):
    source = vault / "paper" / "main.tex"
    source.parent.mkdir(parents=True)
    source.write_text("\\documentclass{article}\n", encoding="utf-8")
    return source


PRIMITIVE_FRAME = "\\begin{frame}{Title}\n  Body\n\\end{frame}"
PRESENTATION_SOURCE = "\\documentclass{beamer}\n\\usetheme[progress]{demo}\n\\begin{document}\n\\end{document}\n"
PREVIEW_PNG = b"\x89PNG\r\n\x1a\npreview"
_DEMO_MANIFEST = 'label = "Demo deck"\nfiles = ["beamerthemedemo.sty"]\nimage_macro = "demoimage"\n' + "".join(
    f'\n[[groups]]\nkey = "{key}"\nlabel = "{key.capitalize()}"\n' for key in ("text", "numbers", "tables", "figures", "process")
)
_COMPACT_MANIFEST = """label = "Compact"
files = ["beamerthemecompact.sty"]
image_macro = "photo"

[[groups]]
key = "figures"
label = "Pictures"

[[groups]]
key = "text"
label = "Words"
"""


@pytest.fixture
def template_sets(tmp_path, monkeypatch) -> Path:
    """Template sets in `<tmp_path>/sets` standing in for `templates/latex`.

    - `demo`: primitives in every order, malformed ones and previews
    - `compact`: its own label, image macro and group order
    - `letter`: a document template without primitives, its `sender.lco` copied into new documents
    - not sets: `bare` without `template.toml` and `skeleton.tex`, `broken` whose `files` names a missing file
    """
    base = tmp_path / "sets"
    files = {
        "demo/template.toml": _DEMO_MANIFEST,
        "demo/skeleton.tex": "\\documentclass{beamer}\n\\usetheme{demo}\n",
        "demo/beamerthemedemo.sty": "\\RequirePackage{tikz}\n",
        "demo/STYLE.md": "Style\n",
        "demo/example.tex": "% Generated\n",
        "demo/primitives/text/01-bullets.tex": f"% Bullets: points with one message\n{PRIMITIVE_FRAME}\n",
        "demo/primitives/text/02-statement.tex": (
            "% Statement: a \\emph{thesis} \\hfill  or quote, short\n\\begin{frame}[plain]\\statement{A}{B}\\end{frame}\n"
        ),
        "demo/primitives/text/03-headless.tex": f"{PRIMITIVE_FRAME}\n",
        "demo/primitives/text/04-twice.tex": f"% Twice: two frames\n{PRIMITIVE_FRAME}\n{PRIMITIVE_FRAME}\n",
        "demo/primitives/text/unnumbered.tex": f"% Unnumbered: no number\n{PRIMITIVE_FRAME}\n",
        "demo/primitives/appendix/01-extra.tex": f"% Extra: an unknown group\n{PRIMITIVE_FRAME}\n",
        "demo/primitives/numbers/10-late.tex": f"% Late: tenth\n{PRIMITIVE_FRAME}\n",
        "demo/primitives/numbers/2-early.tex": f"% Key figures: second\n{PRIMITIVE_FRAME}\n",
        "demo/primitives/figures/01-figrow.tex": f"% Figure row: images side by side\n{PRIMITIVE_FRAME}\n",
        "compact/template.toml": _COMPACT_MANIFEST,
        "compact/skeleton.tex": "\\documentclass{beamer}\n\\usetheme{compact}\n",
        "compact/beamerthemecompact.sty": "\\RequirePackage{xcolor}\n",
        "compact/primitives/text/01-quote.tex": f"% Quote: a quote\n{PRIMITIVE_FRAME}\n",
        "compact/primitives/figures/01-photo.tex": f"% Photo: one photo\n{PRIMITIVE_FRAME}\n",
        "letter/template.toml": 'label = "Letter"\nfiles = ["sender.lco"]\n',
        "letter/skeleton.tex": "\\documentclass{scrlttr2}\n\\LoadLetterOption{sender}\n",
        "letter/sender.lco": "\\setkomavar{fromname}{Me}\n",
        "bare/primitives/text/01-quote.tex": f"% Quote: a quote\n{PRIMITIVE_FRAME}\n",
        "broken/template.toml": 'label = "Broken"\nfiles = ["missing.sty"]\n',
        "broken/skeleton.tex": "\\documentclass{beamer}\n",
        "broken/primitives/text/01-quote.tex": f"% Quote: a quote\n{PRIMITIVE_FRAME}\n",
    }
    for name, content in files.items():
        (base / name).parent.mkdir(parents=True, exist_ok=True)
        (base / name).write_text(content, encoding="utf-8")
    (base / "demo" / "example.pdf").write_bytes(b"%PDF-1")
    for name in ("previews/text/01-bullets.png", "previews/text/03-headless.png", "secret.png"):
        (base / "demo" / name).parent.mkdir(parents=True, exist_ok=True)
        (base / "demo" / name).write_bytes(PREVIEW_PNG)
    monkeypatch.setattr(latex_templates, "TEMPLATE_SETS", base)
    return base
