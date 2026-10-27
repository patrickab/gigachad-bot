-- Sync replay sends only the latest change per resource, so every older row of the same
-- (user, kind, key) is redundant. The app prunes them on each write (`UserScope.record`)
-- through this index; this step deletes the backlog once. The latest row per resource,
-- deletes included, always stays, so replay from any cursor returns the same events.
CREATE INDEX IF NOT EXISTS changes_resource_idx ON changes (user_id, resource_kind, resource_key, seq);

DELETE FROM changes AS superseded
USING (
    SELECT user_id, resource_kind, resource_key, max(seq) AS seq
    FROM changes
    GROUP BY user_id, resource_kind, resource_key
) AS latest
WHERE superseded.user_id = latest.user_id
    AND superseded.resource_kind = latest.resource_kind
    AND superseded.resource_key = latest.resource_key
    AND superseded.seq < latest.seq;
