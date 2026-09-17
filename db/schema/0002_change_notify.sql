-- Publish every committed change-log row so other devices learn about it without polling.
-- The payload is metadata only: PostgreSQL caps a NOTIFY payload at 8 kB, and document
-- or asset content is far larger than that, so subscribers re-fetch the resource by key.
CREATE FUNCTION notify_change() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM pg_notify(
        'gigachad_changes',
        json_build_object(
            'seq', NEW.seq,
            'user_id', NEW.user_id,
            'resource_kind', NEW.resource_kind,
            'resource_key', NEW.resource_key,
            'version', NEW.version,
            'device_id', NEW.device_id
        )::text
    );
    RETURN NULL;
END;
$$;

CREATE TRIGGER changes_notify AFTER INSERT ON changes
    FOR EACH ROW EXECUTE FUNCTION notify_change();
