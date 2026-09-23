"""In-memory DataStore/AssetStore fakes for tests that must not touch Postgres."""

from __future__ import annotations

import hashlib

from lib.data_store import Entry, Revision, StorageNotFoundError


class MemoryDataStore:
    """Minimal DataStore: bytes under flat keys with digest revisions."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def read_bytes(self, key: str) -> tuple[bytes, Revision]:
        content = self._data.get(key)
        if content is None:
            raise StorageNotFoundError(key)
        return content, Revision(token=hashlib.sha256(content).hexdigest())

    def write_bytes(self, key: str, content: bytes, *, expected=None) -> Revision:
        self._data[key] = content
        return Revision(token=hashlib.sha256(content).hexdigest())

    def list(self, prefix: str = "", *, recursive: bool = False) -> list[Entry]:
        del recursive
        scope = f"{prefix.rstrip('/')}/" if prefix else ""
        return [
            Entry(key=key, is_dir=False, size=len(content))
            for key, content in sorted(self._data.items())
            if not prefix or key == prefix or key.startswith(scope)
        ]

    def exists(self, key: str) -> bool:
        return key in self._data

    def mkdir(self, key: str) -> None:
        self._data.setdefault(key, b"")

    def delete(self, key: str, *, recursive: bool = False) -> None:
        if recursive:
            for existing in [existing for existing in self._data if existing == key or existing.startswith(f"{key}/")]:
                del self._data[existing]
        else:
            self._data.pop(key, None)

    def move(self, source: str, destination: str) -> None:
        self._data[destination] = self._data.pop(source)


class _Asset:
    def __init__(self, logical_path: str, content: bytes | None) -> None:
        self.logical_path = logical_path
        self.content = content


class MemoryAssetStore:
    """Minimal AssetStore: only what SandboxService touches (list/read/write/delete)."""

    def __init__(self) -> None:
        self._assets: dict[str, bytes] = {}

    def list(self, prefix: str = "", *, kind: str | None = None) -> list[_Asset]:
        return [_Asset(k, None) for k in self._assets if k.startswith(prefix)]

    def read(self, logical_path: str) -> _Asset:
        content = self._assets.get(logical_path)
        if content is None:
            raise StorageNotFoundError(logical_path)
        return _Asset(logical_path, content)

    def write(self, kind: str, logical_path: str, content: bytes, *, mime: str | None = None) -> _Asset:
        self._assets[logical_path] = content
        return _Asset(logical_path, content)

    def delete(self, logical_path: str) -> None:
        self._assets.pop(logical_path, None)