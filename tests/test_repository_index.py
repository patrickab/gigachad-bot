"""Pi's start folders come from the stored list, a sweep brings it in line with the disk, a miss sweeps once."""

from pathlib import Path

import pytest

from lib import pi_host, repository_index
from lib.pi_host import Repository
from lib.repository_index import PostgresRepositoryStore


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    (tmp_path / "git" / "app" / ".git").mkdir(parents=True)
    (tmp_path / "dev" / "tool").mkdir(parents=True)
    monkeypatch.setattr(pi_host, "HOME", tmp_path)
    monkeypatch.setattr(pi_host, "DEV_ROOT", tmp_path / "dev")
    monkeypatch.setattr(pi_host, "REPOSITORIES_ROOT", tmp_path / "git")
    return tmp_path


def _names(repositories: list[Repository]) -> list[str]:
    return [repository.name for repository in repositories]


def test_the_first_listing_scans_and_stores(home, repository_store):
    assert _names(repository_index.list_repositories()) == ["app", "tool"]
    assert _names(repository_store.load()) == ["app", "tool"]


def test_listing_reads_the_stored_list_without_scanning(home, repository_store, monkeypatch):
    repository_index.sweep()
    monkeypatch.setattr(pi_host, "scan_repositories", lambda: pytest.fail("scanned"))
    assert _names(repository_index.list_repositories()) == ["app", "tool"]


def test_a_sweep_adds_new_folders_and_drops_vanished_ones(home, repository_store):
    repository_index.sweep()
    (home / "git" / "app" / ".git").rmdir()
    (home / "git" / "app").rmdir()
    (home / "code" / "proj" / ".git").mkdir(parents=True)

    repository_index.sweep()

    assert _names(repository_store.load()) == ["tool", "code/proj"]


def test_resolving_a_new_folder_sweeps_once(home, repository_store):
    repository_index.sweep()
    (home / "git" / "fresh").mkdir()
    assert repository_index.resolve("fresh") == home / "git" / "fresh"
    assert repository_index.resolve("missing") is None


def test_resolving_a_vanished_folder_sweeps_it_away(home, repository_store):
    repository_index.sweep()
    (home / "dev" / "tool").rmdir()
    assert repository_index.resolve("tool") is None
    assert _names(repository_store.load()) == ["app"]


def test_a_clone_is_added_by_name(home, repository_store):
    repository_index.add(home / "git" / "cloned")
    assert repository_store.load() == [Repository(name="cloned", path=str(home / "git" / "cloned"))]


def test_a_failed_startup_sweep_keeps_the_stored_list(repository_store, monkeypatch):
    repository_store.repositories = [Repository(name="app", path="/x/app")]
    monkeypatch.setattr(pi_host, "scan_repositories", lambda: (_ for _ in ()).throw(OSError("disk")))
    repository_index.sweep_quietly()
    assert _names(repository_store.load()) == ["app"]


def test_the_postgres_store_keeps_scan_order_and_counts_changes(postgres_pool):
    store = PostgresRepositoryStore(postgres_pool)
    with postgres_pool.connection() as connection:
        connection.execute("TRUNCATE repositories")
    a, b, c = (Repository(name=name, path=f"/home/{name}") for name in ("a", "b", "c"))

    assert store.replace([b, a]) == (2, 0)
    assert store.replace([c, b]) == (1, 1)
    store.add(Repository(name="d", path="/home/d"))
    store.add(Repository(name="c", path="/elsewhere/c"))

    assert store.load() == [Repository(name="c", path="/elsewhere/c"), b, Repository(name="d", path="/home/d")]
