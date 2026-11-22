-- Where Pi sessions may start: the folders `pi_host.scan_repositories` finds under `~`, host-wide (no user).
-- A cache of the disk, rewritten by every sweep (`lib/repository_index.py`), so requests never walk `~`.
-- `position` keeps the scan order (`~/git`, `~/dev`, then checkouts deeper in `~`).
CREATE TABLE repositories (
    name text PRIMARY KEY,
    path text NOT NULL,
    position integer NOT NULL
);
