ALTER TABLE changes DROP CONSTRAINT changes_resource_kind_check;
ALTER TABLE changes ADD CONSTRAINT changes_resource_kind_check
    CHECK (resource_kind IN ('document', 'asset', 'vault_root', 'project_rename'));
