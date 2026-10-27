-- Derived document data next to the content, so hot reads skip parsing or hashing it.
-- Every change is additive: code that predates this step neither reads nor writes these columns.

-- Chat tree metadata (chat_id, parent_id, branch_message_idx, children, qa_count) written by
-- ChatStore in the same statement as the content and stamped with the row version it describes.
-- A NULL or stale value (older rows, writers that do not set it) is derived from the content on
-- the next read and stored back, so no backfill runs here and no stored body can make this step fail.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS meta jsonb;

-- Large content is stored uncompressed out of line: a byte range read fetches only the TOAST chunks
-- it covers and binary writes skip a futile compression attempt. Applies to values written from now on.
ALTER TABLE documents ALTER COLUMN content SET STORAGE EXTERNAL;
ALTER TABLE assets ALTER COLUMN content SET STORAGE EXTERNAL;

-- SHA-256 of the content, maintained by PostgreSQL on every write. PDF annotation canvases are keyed
-- by it, so they no longer hash the whole file per request. Adding it rewrites documents once.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS sha256 bytea GENERATED ALWAYS AS (sha256(content)) STORED;
