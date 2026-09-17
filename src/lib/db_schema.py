"""Apply checksum-protected PostgreSQL schema steps.

Each file in ``db/schema`` is one forward step, applied once and then frozen: the
database records its checksum, so editing an applied file is refused instead of
silently diverging from the live schema. A schema change is therefore always a new
numbered file, never an edit to an old one.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import psycopg

SCHEMA_DIRECTORY = Path(__file__).resolve().parents[2] / "db" / "schema"
LOCK_KEY = "gigachad-schema-steps"
APPLIED_TABLE = "applied_schema_steps"


@dataclass(frozen=True)
class SchemaStep:
    name: str
    checksum: str
    sql: str


def steps() -> list[SchemaStep]:
    """Load the ordered schema steps shipped with the application."""
    result: list[SchemaStep] = []
    for path in sorted(SCHEMA_DIRECTORY.glob("[0-9][0-9][0-9][0-9]_*.sql")):
        sql = path.read_text()
        result.append(SchemaStep(path.name, hashlib.sha256(sql.encode()).hexdigest(), sql))
    return result


def database_url(explicit_url: str | None = None) -> str:
    """Return an explicit URL or the required runtime configuration value."""
    if explicit_url:
        return explicit_url
    try:
        return os.environ["GIGACHAD_DATABASE_URL"]
    except KeyError as exc:
        raise RuntimeError("GIGACHAD_DATABASE_URL is required") from exc


def _prepare(connection: psycopg.Connection) -> None:
    connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (LOCK_KEY,))
    # Restoring a dump taken before the rename brings back ``schema_migrations``.
    # Renaming it keeps that history instead of replaying steps against populated tables.
    connection.execute(f"ALTER TABLE IF EXISTS schema_migrations RENAME TO {APPLIED_TABLE}")
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {APPLIED_TABLE} (
            name text PRIMARY KEY,
            checksum text NOT NULL,
            applied_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )


def _applied(connection: psycopg.Connection) -> dict[str, str]:
    return dict(connection.execute(f"SELECT name, checksum FROM {APPLIED_TABLE}").fetchall())


def status(url: str) -> list[tuple[SchemaStep, bool]]:
    """Return every shipped step and whether the database has applied it."""
    with psycopg.connect(url) as connection, connection.transaction():
        _prepare(connection)
        applied = _applied(connection)
        result: list[tuple[SchemaStep, bool]] = []
        for step in steps():
            checksum = applied.get(step.name)
            if checksum is not None and checksum != step.checksum:
                raise RuntimeError(f"Applied schema step checksum differs: {step.name}")
            result.append((step, checksum is not None))
        return result


def upgrade(url: str) -> list[str]:
    """Apply each unapplied step in a single transaction."""
    with psycopg.connect(url) as connection, connection.transaction():
        _prepare(connection)
        applied = _applied(connection)
        applied_names: list[str] = []
        for step in steps():
            checksum = applied.get(step.name)
            if checksum is not None:
                if checksum != step.checksum:
                    raise RuntimeError(f"Applied schema step checksum differs: {step.name}")
                continue
            connection.execute(step.sql)
            connection.execute(
                f"INSERT INTO {APPLIED_TABLE} (name, checksum) VALUES (%s, %s)",
                (step.name, step.checksum),
            )
            applied_names.append(step.name)

        return applied_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "upgrade"))
    parser.add_argument("--database-url")
    args = parser.parse_args()
    url = database_url(args.database_url)

    if args.command == "status":
        for step, is_applied in status(url):
            print(f"{'applied' if is_applied else 'pending'} {step.name}")
        return

    for name in upgrade(url):
        print(f"applied {name}")


if __name__ == "__main__":
    main()
