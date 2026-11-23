-- Revocable links granting one stored resource of its owner to whoever holds the secret.
-- Only the secret's SHA-256 is stored. Each target column references its resource, so a link
-- follows renames and dies with its resource; a new resource kind adds its own column.
CREATE TABLE share_links (
    secret_hash bytea PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    document_key text,
    asset_path text,
    access text NOT NULL CHECK (access IN ('view', 'edit')),
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (num_nonnulls(document_key, asset_path) = 1),
    -- One link per resource. NULLs are distinct, so each constraint only binds its own kind.
    UNIQUE (user_id, document_key),
    UNIQUE (user_id, asset_path),
    FOREIGN KEY (user_id, document_key) REFERENCES documents (user_id, key) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (user_id, asset_path) REFERENCES assets (user_id, logical_path) ON UPDATE CASCADE ON DELETE CASCADE
);

-- The canvas-only predecessor, never served by any route.
INSERT INTO share_links (secret_hash, user_id, document_key, access, created_at)
SELECT secret_hash, user_id, canvas_key, 'edit', created_at FROM canvas_shares;
DROP TABLE canvas_shares;
