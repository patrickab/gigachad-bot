"""JSON documents over a ``DataStore``: one reader with a fallback, one writer in the stored byte format."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from lib.data_store import DataStore, Entry, Revision, StorageNotFoundError, overwrite_text
from lib.json_io import dump_json

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

JsonValue = dict[str, Any] | list[Any]

logger = logging.getLogger(__name__)


class JsonDocs:
    """Read and write JSON values by storage key."""

    def __init__(self, store: DataStore) -> None:
        self._store = store

    def read(self, key: str, default: object = None, *, log: str | None = None) -> Any:  # noqa: ANN401 - untyped until the caller checks it
        """Parsed JSON at *key*, *default* when it is missing or malformed. A malformed document is logged under *log* when given."""
        try:
            content, _ = self._store.read_bytes(key)
            return json.loads(content.decode("utf-8"))
        except StorageNotFoundError:
            return default
        except ValueError as exc:
            if log is not None:
                logger.error("Failed to read %s at %s: %s", log, key, exc)
            return default

    def read_many(self, keys: Iterable[str], default: object = None) -> dict[str, Any]:
        """Parsed JSON of every key in *keys*, read in one round trip, *default* for a missing or malformed one."""
        keys = list(keys)
        stored = self._store.read_many(keys)
        return {key: _parse(stored[key][0], default) if key in stored else default for key in keys}

    def read_with_revision(self, key: str) -> tuple[Any, Revision]:
        """Parsed JSON at *key* plus its revision. StorageNotFoundError when missing, ValueError when malformed."""
        content, revision = self._store.read_bytes(key)
        return json.loads(content.decode("utf-8")), revision

    def write(self, key: str, data: JsonValue, *, expected: Revision | None = None) -> Revision:
        """Store *data* in the shared JSON format, guarded by *expected* when given."""
        return self._store.write_bytes(key, _encode(data), expected=expected)

    def overwrite(self, key: str, data: JsonValue) -> Revision:
        """Store *data* over the current value of *key*, guarded by the revision read just before."""
        return overwrite_text(self._store, key, dump_json(data))

    def apply(self, writes: Iterable[tuple[str, JsonValue]], deletes: Sequence[str] = ()) -> list[Revision]:
        """Write and delete several documents in one unguarded transaction, so the last writer wins."""
        return self._store.apply([(key, _encode(data), None) for key, data in writes], list(deletes))

    def mkdir(self, key: str) -> None:
        self._store.mkdir(key)

    def is_file(self, key: str) -> bool:
        return self._store.is_file(key)

    def is_dir(self, key: str) -> bool:
        return self._store.is_dir(key)

    def exists(self, key: str) -> bool:
        return self._store.exists(key)

    def list(self, prefix: str = "", *, recursive: bool = False) -> list[Entry]:
        return self._store.list(prefix, recursive=recursive)

    def delete(self, key: str, *, recursive: bool = False) -> None:
        self._store.delete(key, recursive=recursive)


def _encode(data: JsonValue) -> bytes:
    """*data* in the stored byte format."""
    return dump_json(data).encode("utf-8")


def _parse(content: bytes, default: object) -> Any:  # noqa: ANN401 - untyped until the caller checks it
    """*content* parsed as JSON, *default* when it is malformed."""
    try:
        return json.loads(content.decode("utf-8"))
    except ValueError:
        return default
