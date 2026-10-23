DELETE FROM changes WHERE resource_kind = 'project_rename';
ALTER TABLE changes DROP CONSTRAINT changes_resource_kind_check;
ALTER TABLE changes ADD CONSTRAINT changes_resource_kind_check
    CHECK (resource_kind IN ('document', 'asset', 'vault_root'));
