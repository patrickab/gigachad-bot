-- Canvas snapshots remain in documents; this idempotent log orders fine-grained
-- collaboration. A canvas's revision is its highest logged batch, serialized by
-- the documents row lock. Accepted batches also write a changes row, whose
-- existing notification wakes canvas streams.
CREATE TABLE canvas_mutations (
    mutation_id uuid PRIMARY KEY,
    user_id uuid NOT NULL,
    canvas_key text NOT NULL,
    revision bigint NOT NULL CHECK (revision > 0),
    position integer NOT NULL CHECK (position >= 0),
    kind text NOT NULL CHECK (kind IN ('upsert', 'delete')),
    entity text NOT NULL CHECK (entity IN ('stroke', 'frame', 'attachment', 'text')),
    entity_id text NOT NULL CHECK (entity_id <> ''),
    value jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK ((kind = 'upsert' AND value IS NOT NULL) OR (kind = 'delete' AND value IS NULL)),
    FOREIGN KEY (user_id, canvas_key) REFERENCES documents (user_id, key)
        ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE INDEX canvas_mutations_replay_idx ON canvas_mutations (user_id, canvas_key, revision, position);
