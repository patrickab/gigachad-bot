-- Allow SandboxService to persist sandbox assets.
ALTER TABLE assets DROP CONSTRAINT assets_kind_check;
ALTER TABLE assets ADD CONSTRAINT assets_kind_check CHECK (kind IN ('upload', 'pdf', 'mineru_markdown', 'mineru_image', 'drawing', 'sandbox'));