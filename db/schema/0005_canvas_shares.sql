-- One revocable edit link per canvas. Only the SHA-256 of the link secret is stored;
-- revoking deletes the row, and the row follows its canvas through renames and deletes.
CREATE TABLE canvas_shares (
    secret_hash bytea PRIMARY KEY,
    user_id uuid NOT NULL,
    canvas_key text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (user_id, canvas_key),
    FOREIGN KEY (user_id, canvas_key) REFERENCES documents (user_id, key)
        ON UPDATE CASCADE ON DELETE CASCADE
);
