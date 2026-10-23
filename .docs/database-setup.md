# Database setup

- Use PostgreSQL 17 through the repository's `compose.yaml`.
- Keep development and production databases separate and bound to loopback.
- Follow [deployment](deployment.md) for production credentials, identity, and service setup.
- Use [architecture](architecture.md#persistence-and-synchronization) for storage and migration contracts.

## Environments

| Setting | Development | Production |
| --- | --- | --- |
| Compose project | `gigachad-dev` | `gigachad-prod` |
| Listener | `127.0.0.1:5432` | `127.0.0.2:5433` |
| Data volume | `gigachad-dev_gigachad-postgres` | `gigachad-prod_gigachad-postgres` |
| Password in `~/.secrets` | `POSTGRES_PASSWORD_DEV` | `POSTGRES_PASSWORD_PROD` |
| Database | `gigachad` | `gigachad` |
| Database role | Launching Linux user | Launching Linux user |

- Require Docker with Compose and access to its daemon.
- Preserve existing passwords. Changing an environment variable does not change credentials in an initialized volume.
- Store credentials in `~/.secrets`, not this runbook or the shared nonsecret configuration file.
- Managed backend launchers set `GIGACHAD_DATABASE_URL` and `PGPASSWORD` from the selected environment. Do not add a hard-coded database URL to the shared configuration file for this setup.

## 1. Prepare development

Run from the repository root:

```bash
./scripts/setup-postgres.sh
```

- The helper starts only PostgreSQL, never the application servers.
- Without a password, it asks before generating and saving one in `~/.secrets`.
- If a development volume already exists without saved credentials, restore its original password before retrying.
- The helper waits for container health and verifies password authentication.
- Let the user start `./run.sh` separately. For production, provision credentials and services through [deployment](deployment.md).

## 2. Select a database for maintenance

Set the checkout path and define the maintenance helper:

```bash
REPO="$HOME/git/gigachad-bot"
source "$HOME/.secrets"
export POSTGRES_USER="$(whoami)"

compose() {
    docker compose --project-directory "$REPO" --project-name "$PROJECT" "$@"
}
```

Select development:

```bash
export POSTGRES_HOST=127.0.0.1
export POSTGRES_PORT=5432
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD_DEV:?Restore development credentials first}"
PROJECT=gigachad-dev
```

Or select production:

```bash
export POSTGRES_HOST=127.0.0.2
export POSTGRES_PORT=5433
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD_PROD:?Configure production credentials first}"
PROJECT=gigachad-prod
```

- Confirm the selected project before any restore. Container names are Compose-generated, use the `postgres` service rather than a hard-coded container name.
- To start only the selected database:

```bash
compose up --detach --wait postgres
compose ps postgres
```

## 3. Verify connectivity and data

```bash
compose exec -T -e PGPASSWORD="$POSTGRES_PASSWORD" postgres \
    psql -h 127.0.0.1 -U "$POSTGRES_USER" -d gigachad \
    -v ON_ERROR_STOP=1 -c 'SELECT current_database(), current_user;'

compose exec -T postgres psql -U "$POSTGRES_USER" -d gigachad -c '\dt'

compose exec -T postgres psql -U "$POSTGRES_USER" -d gigachad \
    -v ON_ERROR_STOP=1 -c "
SELECT 'documents' AS table_name, count(*) FROM documents
UNION ALL SELECT 'assets', count(*) FROM assets
UNION ALL SELECT 'users', count(*) FROM users
UNION ALL SELECT 'devices', count(*) FROM devices
UNION ALL SELECT 'changes', count(*) FROM changes
UNION ALL SELECT 'vault_roots', count(*) FROM vault_roots;
"
```

- New containers have no application tables until backend startup applies `db/schema/` through `src/lib/db_schema.py`.
- Add new migration files rather than changing applied steps. Startup rejects changed migration checksums.
- Compare row counts with the intended source or backup, not historical machine-specific counts.

## 4. Back up or seed an empty database

- A dump/restore is a one-time copy, not replication. Source and destination diverge afterward.
- Back up the destination before replacing any existing deployment. Stop application writers during a cutover and keep them stopped until verification finishes.
- Restore only into an empty destination database. Do not use these commands to merge databases or overwrite newer user data.
- Use PostgreSQL 17 on both sides. Also back up the shared Documents/PDF mirror and any mounted source files that need preservation.

Create a private custom-format dump on the source:

```bash
umask 077
DUMP="$HOME/gigachad.dump"
compose exec -T postgres pg_dump -U "$POSTGRES_USER" -d gigachad -Fc > "$DUMP"
```

If moving between hosts, transfer it over an authenticated private connection:

```bash
DESTINATION=user@private-host
scp "$DUMP" "$DESTINATION:gigachad.dump"
```

On the destination, select its maintenance environment, then restore:

```bash
DUMP="$HOME/gigachad.dump"
compose exec -T postgres pg_restore -U "$POSTGRES_USER" -d gigachad \
    --no-owner --no-acl --single-transaction --exit-on-error < "$DUMP"
```

- The restore preserves database records, including users and mounted paths. Verify that identity and filesystem references match the destination.
- Repeat connectivity, table, and row-count checks before resuming application writers.
- Retain the original database and dump until the destination is verified.
