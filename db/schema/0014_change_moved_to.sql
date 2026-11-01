-- A renamed resource's vacated change row names the key it moved to, so devices showing the old key follow it.
ALTER TABLE changes ADD COLUMN moved_to text;

CREATE OR REPLACE FUNCTION notify_change() RETURNS trigger
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
            'device_id', NEW.device_id,
            'moved_to', NEW.moved_to
        )::text
    );
    RETURN NULL;
END;
$$;
