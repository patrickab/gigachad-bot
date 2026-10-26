-- Memory "category" becomes "topic": rename storage keys and JSON fields (description -> summary, category -> topic).
DO $$
DECLARE
    doc record;
    rewritten jsonb;
BEGIN
    -- Topic files: rewrite description -> summary, then rename the key.
    FOR doc IN
        SELECT user_id, key
        FROM documents
        WHERE NOT is_dir
          AND (key = 'memory/global-categories.json' OR key LIKE 'memory/project/%/memory/categories.json')
    LOOP
        BEGIN
            SELECT jsonb_agg(
                       CASE
                           WHEN jsonb_typeof(item.value) = 'object' AND item.value ? 'description'
                           THEN (item.value - 'description') || jsonb_build_object('summary', item.value -> 'description')
                           ELSE item.value
                       END
                       ORDER BY item.ordinality
                   )
              INTO rewritten
              FROM documents d,
                   LATERAL jsonb_array_elements(convert_from(d.content, 'UTF8')::jsonb) WITH ORDINALITY AS item(value, ordinality)
             WHERE d.user_id = doc.user_id AND d.key = doc.key;

            UPDATE documents
               SET key = regexp_replace(key, 'categories\.json$', 'topics.json'),
                   content = convert_to(COALESCE(rewritten, '[]'::jsonb)::text, 'UTF8'),
                   version = version + 1,
                   updated_at = now()
             WHERE user_id = doc.user_id AND key = doc.key;
        EXCEPTION WHEN others THEN
            RAISE WARNING 'memory topics migration skipped % for user %: %', doc.key, doc.user_id, SQLERRM;
        END;
    END LOOP;

    -- Memory lists: rename category -> topic.
    FOR doc IN
        SELECT user_id, key
        FROM documents
        WHERE NOT is_dir
          AND (key = 'memory/global-profile.json' OR key LIKE 'memory/project/%/memory/memory.json')
    LOOP
        BEGIN
            SELECT jsonb_agg(
                       CASE
                           WHEN jsonb_typeof(item.value) = 'object' AND item.value ? 'category'
                           THEN (item.value - 'category') || jsonb_build_object('topic', item.value -> 'category')
                           ELSE item.value
                       END
                       ORDER BY item.ordinality
                   )
              INTO rewritten
              FROM documents d,
                   LATERAL jsonb_array_elements(convert_from(d.content, 'UTF8')::jsonb) WITH ORDINALITY AS item(value, ordinality)
             WHERE d.user_id = doc.user_id AND d.key = doc.key;

            UPDATE documents
               SET content = convert_to(COALESCE(rewritten, '[]'::jsonb)::text, 'UTF8'),
                   version = version + 1,
                   updated_at = now()
             WHERE user_id = doc.user_id AND key = doc.key;
        EXCEPTION WHEN others THEN
            RAISE WARNING 'memory topics migration skipped % for user %: %', doc.key, doc.user_id, SQLERRM;
        END;
    END LOOP;
END
$$;
