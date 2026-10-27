"""Backend-neutral storage for application-owned data.

Keys identify logical application artifacts, addressed by database-native
namespaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime


class StorageError(RuntimeError):
    """The configured persistent storage could not complete an operation."""


class StorageConflictError(StorageError):
    """A file changed after the caller read the revision it is trying to save."""


class StorageNotFoundError(StorageError, FileNotFoundError):
    """A requested storage key does not exist."""


class InvalidStorageKey(ValueError):
    """A caller supplied an absolute or traversal-containing storage key."""


@dataclass(frozen=True)
class Revision:
    """Opaque content revision; a digest of the stored bytes."""

    token: str


@dataclass(frozen=True)
class Entry:
    """Listing item. `size` and `revision` are None for directories, `modified` (last write) is set by `stat` only."""
    key: str
    is_dir: bool
    size: int | None = None
    revision: Revision | None = None
    modified: datetime | None = None


class ByteStore(Protocol):
    """The key-addressed byte operations shared by the database store and a workspace directory."""
    def read_bytes(self, key: str) -> tuple[bytes, Revision]: ...
    def apply(self, writes: list[tuple[str, bytes, Revision | None]], deletes: list[str]) -> list[Revision]:
        """Write every ``(key, content, expected)`` then delete *deletes*, atomically: one stale revision aborts them all."""
        ...
    def list(self, prefix: str = "", *, recursive: bool = False) -> list[Entry]: ...
    def delete(self, key: str) -> None: ...


class DataStore(ByteStore, Protocol):
    """Key-addressed byte store. A stale `expected` revision raises StorageConflictError."""
    def read_many(self, keys: Iterable[str]) -> dict[str, tuple[bytes, Revision]]:
        """Content and revision of every file among *keys* at once. Missing keys and directories are left out."""
        ...
    def write_bytes(self, key: str, content: bytes, *, expected: Revision | None = None) -> Revision: ...
    def is_file(self, key: str) -> bool: ...
    def is_dir(self, key: str) -> bool: ...
    def exists(self, key: str) -> bool: ...
    def stat(self, key: str) -> Entry | None:
        """The file or directory at *key* (one implied by descendants is a directory) without content, or None when absent."""
        ...
    def read_range(self, key: str, start: int, length: int) -> bytes:
        """Up to *length* bytes of the file at *key* from offset *start*. Raises StorageNotFoundError."""
        ...
    def sha256(self, key: str) -> str:
        """Hex SHA-256 of the file at *key*, hashed where it is stored. Raises StorageNotFoundError."""
        ...
    def delete(self, key: str, *, recursive: bool = False) -> None: ...
    def move(self, source: str, destination: str) -> None: ...


def read_text(store: DataStore, key: str) -> tuple[str, Revision]:
    content, revision = store.read_bytes(key)
    return content.decode("utf-8"), revision


def write_text(store: DataStore, key: str, content: str, *, expected: Revision | None = None) -> Revision:
    return store.write_bytes(key, content.encode("utf-8"), expected=expected)


def read_text_or(store: DataStore, key: str, default: str | None = None) -> str | None:
    """Text stored at *key*, or *default* when the key does not exist."""
    try:
        return read_text(store, key)[0]
    except StorageNotFoundError:
        return default


def overwrite_text(store: DataStore, key: str, content: str) -> Revision:
    """Write *content* over the current value of *key*, guarded by the revision read just before."""
    try:
        revision = read_text(store, key)[1]
    except StorageNotFoundError:
        revision = None
    return write_text(store, key, content, expected=revision)


def validate_key(key: str, *, allow_empty: bool = False) -> str:
    """Normalize a logical key and reject paths outside the storage root."""
    if not isinstance(key, str):
        raise InvalidStorageKey("Storage key must be a string")
    # Backslashes count as separators so `..\\x` can't bypass the traversal check.
    normalized = key.replace("\\", "/").strip()
    if PurePosixPath(normalized).is_absolute():
        raise InvalidStorageKey("Storage key must be relative and cannot traverse directories")
    # A leading slash is rejected above, so only trailing slashes are stripped here.
    normalized = normalized.strip("/")
    if not normalized:
        if allow_empty:
            return ""
        raise InvalidStorageKey("Storage key cannot be empty")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise InvalidStorageKey("Storage key must be relative and cannot traverse directories")
    return path.as_posix()
