"""Transactional authority for collaborative ``.canvas`` mutation batches.

The materialized snapshot stays in ``documents``; ``canvas_mutations`` is the
ordered, idempotent log of accepted batches. The document row lock serializes
writers, and every accepted batch writes a ``changes`` row so the existing
change broker wakes canvas streams, which then replay the durable log.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
import json
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from lib.data_store import InvalidStorageKey, validate_key

Entity = Literal["stroke", "frame", "attachment", "text"]
MutationKind = Literal["upsert", "delete"]
_COLLECTIONS: dict[Entity, str] = {
    "frame": "frames",
    "stroke": "strokes",
    "attachment": "attachments",
    "text": "texts",
}


class CanvasError(RuntimeError):
    """Base error for the canvas mutation module."""


class CanvasNotFound(CanvasError):
    """The authenticated user does not own the requested canvas snapshot."""


class InvalidCanvasSnapshot(CanvasError):
    """A stored canvas cannot be safely materialized as a mutation target."""


class InvalidCanvasMutation(CanvasError):
    """A mutation would violate the canvas's ID-keyed entity invariant."""


@dataclass(frozen=True)
class CanvasMutation:
    mutation_id: UUID
    kind: MutationKind
    entity: Entity
    entity_id: str
    value: dict[str, Any] | None = None

    def event_data(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mutationId": str(self.mutation_id),
            "kind": self.kind,
            "entity": self.entity,
            "id": self.entity_id,
        }
        if self.value is not None:
            result["value"] = self.value
        return result


@dataclass(frozen=True)
class CanvasBatch:
    canvas_key: str
    revision: int
    mutations: tuple[CanvasMutation, ...]

    def event_data(self) -> dict[str, Any]:
        return {
            "canvasKey": self.canvas_key,
            "revision": self.revision,
            "mutations": [mutation.event_data() for mutation in self.mutations],
        }


def normalize_canvas_key(canvas_key: str) -> str:
    """Return a logical canvas key, rejecting paths outside document storage."""
    try:
        key = validate_key(canvas_key)
    except InvalidStorageKey as exc:
        raise CanvasNotFound("Canvas not found") from exc
    if not key.endswith(".canvas"):
        raise CanvasNotFound("Canvas not found")
    return key


def _entities(value: Any, name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise InvalidCanvasSnapshot(f"Canvas {name} must be an array of objects")
    return [dict(item) for item in value]


def _with_ids(entity: Entity, values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Deterministic IDs make every read of an un-migrated canvas agree with the
    # IDs its first accepted batch persists.
    used: set[str] = set()
    for index, item in enumerate(values):
        entity_id = item.get("id")
        if not isinstance(entity_id, str) or not entity_id or entity_id in used:
            payload = json.dumps({k: v for k, v in item.items() if k != "id"}, separators=(",", ":"), sort_keys=True)
            item["id"] = entity_id = str(uuid5(NAMESPACE_URL, f"gigachad-canvas-v1:{entity}:{index}:{payload}"))
        used.add(entity_id)
    return values


def canonical_snapshot(content: bytes) -> dict[str, Any]:
    """Parse stored canvas bytes into the ID-keyed v1 shape; fail closed on anything else."""
    if not content.strip():
        return {"version": 1, "frames": [], "strokes": [], "attachments": [], "texts": []}
    try:
        document = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCanvasSnapshot("Canvas snapshot is not valid JSON") from exc
    if not isinstance(document, dict) or document.get("version") != 1:
        raise InvalidCanvasSnapshot("Canvas snapshot has an unsupported shape")
    frames = _entities(document["frames"], "frames") if "frames" in document else [
        *({**page, "kind": "page", "width": 794} for page in _entities(document.get("pages"), "pages")),
        *({**image, "kind": "image"} for image in _entities(document.get("imageEmbeds"), "imageEmbeds")),
    ]
    attachments = _entities(document["attachments"], "attachments") if "attachments" in document else [
        {**pdf, "kind": "pdf"} for pdf in _entities(document.get("pdfEmbeds"), "pdfEmbeds")
    ]
    texts = [
        {**text, "width": 240 if text.get("width") is None else text["width"], "height": 80 if text.get("height") is None else text["height"]}
        for text in _entities(document.get("texts"), "texts")
    ]
    canonical: dict[str, Any] = {
        "version": 1,
        "frames": _with_ids("frame", frames),
        "strokes": _with_ids("stroke", _entities(document.get("strokes"), "strokes")),
        "attachments": _with_ids("attachment", attachments),
        "texts": _with_ids("text", texts),
    }
    if isinstance(document.get("viewport"), dict):
        canonical["viewport"] = document["viewport"]
    return canonical


def _canonical_mutation(mutation: CanvasMutation) -> CanvasMutation:
    if not mutation.entity_id:
        raise InvalidCanvasMutation("Canvas mutation id is required")
    if mutation.kind == "delete":
        if mutation.value is not None:
            raise InvalidCanvasMutation("Delete mutations cannot carry a value")
        return mutation
    if mutation.value is None:
        raise InvalidCanvasMutation("Upsert mutations require a value")
    if mutation.value.get("id", mutation.entity_id) != mutation.entity_id:
        raise InvalidCanvasMutation("Mutation value id must match mutation id")
    value = {**mutation.value, "id": mutation.entity_id}
    return CanvasMutation(mutation.mutation_id, mutation.kind, mutation.entity, mutation.entity_id, value)


def _apply_mutation(document: dict[str, Any], mutation: CanvasMutation) -> None:
    items = document[_COLLECTIONS[mutation.entity]]
    index = next((i for i, item in enumerate(items) if item["id"] == mutation.entity_id), None)
    if mutation.kind == "delete":
        if index is not None:
            del items[index]
    elif index is not None:
        items[index] = mutation.value
    else:
        items.append(mutation.value)


_REVISION_SQL = "SELECT COALESCE(MAX(revision), 0) FROM canvas_mutations WHERE user_id = %s AND canvas_key = %s"


class CanvasMutationStore:
    """Serialize one user's canvas mutations and materialize them atomically."""

    def __init__(self, pool: ConnectionPool, user_id: UUID, device_id: UUID | None = None) -> None:
        self._pool = pool
        self._user_id = user_id
        self._device_id = device_id

    def snapshot(self, canvas_key: str) -> tuple[dict[str, Any], int]:
        """Return the canonical materialized snapshot and the revision that produced it."""
        key = normalize_canvas_key(canvas_key)
        with self._pool.connection() as connection:
            # One statement reads content and revision from the same MVCC snapshot.
            row = connection.execute(
                f"SELECT content, ({_REVISION_SQL}) FROM documents WHERE user_id = %s AND key = %s AND NOT is_dir",
                (self._user_id, key, self._user_id, key),
            ).fetchone()
        if row is None:
            raise CanvasNotFound("Canvas not found")
        return canonical_snapshot(bytes(row[0])), row[1]

    def replay(self, canvas_key: str, since_revision: int) -> list[CanvasBatch]:
        """Return one ordered durable batch per revision after ``since_revision``."""
        key = normalize_canvas_key(canvas_key)
        with self._pool.connection() as connection:
            if connection.execute(
                "SELECT 1 FROM documents WHERE user_id = %s AND key = %s AND NOT is_dir", (self._user_id, key)
            ).fetchone() is None:
                raise CanvasNotFound("Canvas not found")
            rows = connection.execute(
                """
                SELECT revision, mutation_id, kind, entity, entity_id, value
                FROM canvas_mutations
                WHERE user_id = %s AND canvas_key = %s AND revision > %s
                ORDER BY revision, position
                """,
                (self._user_id, key, since_revision),
            ).fetchall()
        return [
            CanvasBatch(key, revision, tuple(CanvasMutation(*row[1:]) for row in group))
            for revision, group in groupby(rows, key=lambda row: row[0])
        ]

    def apply(self, canvas_key: str, mutations: list[CanvasMutation]) -> int:
        """Accept unseen mutations as one new revision and return the revision that holds them.

        Retried mutation IDs are acknowledged with the revision they were first
        accepted at, so a client can wait for the stream to reach that revision.
        """
        key = normalize_canvas_key(canvas_key)
        requested = [_canonical_mutation(mutation) for mutation in mutations]
        with self._pool.connection() as connection, connection.transaction():
            row = connection.execute(
                "SELECT content FROM documents WHERE user_id = %s AND key = %s AND NOT is_dir FOR UPDATE",
                (self._user_id, key),
            ).fetchone()
            if row is None:
                raise CanvasNotFound("Canvas not found")
            document = canonical_snapshot(bytes(row[0]))
            revision = connection.execute(_REVISION_SQL, (self._user_id, key)).fetchone()[0] + 1
            accepted = [
                mutation
                for position, mutation in enumerate(requested)
                if connection.execute(
                    """
                    INSERT INTO canvas_mutations
                        (mutation_id, user_id, canvas_key, revision, position, kind, entity, entity_id, value)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (mutation_id) DO NOTHING
                    RETURNING 1
                    """,
                    (
                        mutation.mutation_id, self._user_id, key, revision, position, mutation.kind,
                        mutation.entity, mutation.entity_id, Jsonb(mutation.value) if mutation.value is not None else None,
                    ),
                ).fetchone()
            ]
            if not accepted:
                return connection.execute(
                    f"SELECT COALESCE(MAX(revision), ({_REVISION_SQL})) FROM canvas_mutations "
                    "WHERE user_id = %s AND canvas_key = %s AND mutation_id = ANY(%s)",
                    (self._user_id, key, self._user_id, key, [mutation.mutation_id for mutation in requested]),
                ).fetchone()[0]
            for mutation in accepted:
                _apply_mutation(document, mutation)
            version = connection.execute(
                """
                UPDATE documents SET content = %s, version = version + 1, updated_at = now()
                WHERE user_id = %s AND key = %s
                RETURNING version
                """,
                (json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode(), self._user_id, key),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO changes (user_id, resource_kind, resource_key, version, operation, device_id)
                VALUES (%s, 'document', %s, %s, 'write', %s)
                """,
                (self._user_id, key, version, self._device_id),
            )
        return revision
