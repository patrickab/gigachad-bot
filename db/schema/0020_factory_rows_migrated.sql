-- The factory's rows move into the factory core one by one (`lib/factory_migration.py`): `migrated_at` marks a row
-- that moved or never can (logged), a null one is still to move. The tables stay, so a row that was not moved stays
-- readable. A database where an earlier build renamed the tables to `*_migrated` after moving every row gets them
-- back with every row marked.
DO $$
DECLARE
    renamed boolean := to_regclass('factory_tasks_migrated') IS NOT NULL;
BEGIN
    IF renamed THEN
        ALTER TABLE factory_tasks_migrated RENAME TO factory_tasks;
        ALTER TABLE factory_repo_settings_migrated RENAME TO factory_repo_settings;
    END IF;
    ALTER TABLE factory_repo_settings ADD COLUMN migrated_at timestamptz;
    ALTER TABLE factory_tasks ADD COLUMN migrated_at timestamptz;
    IF renamed THEN
        UPDATE factory_repo_settings SET migrated_at = now();
        UPDATE factory_tasks SET migrated_at = now();
    END IF;
END $$;
