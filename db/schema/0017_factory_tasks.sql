-- The agent factory on Pi sessions: the repos it serves with their settings, and its tasks.
-- `repo` is the absolute path of a primary git checkout. A task's state lives in its Pi session file only;
-- `worktree` and `branch` are null for a task that works in the checkout itself.
CREATE TABLE factory_repo_settings (
    repo text PRIMARY KEY,
    settings jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE factory_tasks (
    id text PRIMARY KEY,
    repo text NOT NULL REFERENCES factory_repo_settings (repo),
    session_id text NOT NULL UNIQUE,
    session_path text NOT NULL,
    worktree text,
    branch text,
    title text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX factory_tasks_repo_created_at_idx ON factory_tasks (repo, created_at);
