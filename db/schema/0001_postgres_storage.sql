CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE users (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tailscale_login text NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE devices (
    id uuid PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    label text,
    last_seen_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE documents (
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key text NOT NULL,
    content bytea NOT NULL,
    is_dir boolean NOT NULL DEFAULT false,
    version bigint NOT NULL DEFAULT 1 CHECK (version > 0),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, key),
    CHECK (key <> ''),
    CHECK (key !~ '(^|/)\.\.?(/|$)')
);
CREATE INDEX documents_user_prefix_idx ON documents (user_id, key text_pattern_ops);

CREATE TABLE assets (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind text NOT NULL CHECK (kind IN ('upload', 'pdf', 'mineru_markdown', 'mineru_image', 'drawing')),
    logical_path text NOT NULL,
    content bytea NOT NULL,
    mime text NOT NULL,
    sha256 text NOT NULL,
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    version bigint NOT NULL DEFAULT 1 CHECK (version > 0),
    UNIQUE (user_id, logical_path)
);

CREATE TABLE vault_roots (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    path text NOT NULL,
    project_slug text,
    mountpoints jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (user_id, path)
);

CREATE TABLE changes (
    seq bigserial PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    resource_kind text NOT NULL CHECK (resource_kind IN ('document', 'asset', 'vault_root')),
    resource_key text NOT NULL,
    version bigint,
    operation text NOT NULL CHECK (operation IN ('write', 'delete', 'move')),
    device_id uuid REFERENCES devices(id) ON DELETE SET NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX changes_user_seq_idx ON changes (user_id, seq);
