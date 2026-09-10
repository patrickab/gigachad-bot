"""Backend-neutral storage for application-owned data.

Keys are always relative to the configured Documents root.  Domain stores use
this module instead of local filesystem paths, so changing the backing store
does not leak into chat, project, prompt, memory, or graph logic.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import fnmatch
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


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
    """Opaque content revision; an HTTP ETag for WebDAV and a digest locally."""

    token: str


@dataclass(frozen=True)
class Entry:
    key: str
    is_dir: bool
    size: int | None = None
    revision: Revision | None = None


class DataStore(Protocol):
    def read_bytes(self, key: str) -> tuple[bytes, Revision]: ...
    def write_bytes(self, key: str, content: bytes, *, expected: Revision | None = None) -> Revision: ...
    def list(self, prefix: str = "", *, recursive: bool = False) -> list[Entry]: ...
    def exists(self, key: str) -> bool: ...
    def mkdir(self, key: str) -> None: ...
    def delete(self, key: str, *, recursive: bool = False) -> None: ...
    def move(self, source: str, destination: str) -> None: ...
    def copy(self, source: str, destination: str) -> None: ...


class DataStorePath:
    """Small ``Path`` compatibility layer for legacy domain-store algorithms.

    It deliberately exposes only operations already used by the stores.  The
    backing data still goes through ``DataStore``; new code should use logical
    keys and the adapter directly.
    """

    def __init__(self, store: DataStore, key: str = "") -> None:
        self.store = store
        self.key = validate_key(key, allow_empty=True)

    def __str__(self) -> str:
        return self.key

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DataStorePath) and self.store is other.store and self.key == other.key

    def __hash__(self) -> int:
        return hash((id(self.store), self.key))

    def __fspath__(self) -> str:
        return self.key

    def __truediv__(self, child: str) -> "DataStorePath":
        return DataStorePath(self.store, f"{self.key}/{child}" if self.key else child)

    @property
    def name(self) -> str:
        return PurePosixPath(self.key).name

    @property
    def stem(self) -> str:
        return PurePosixPath(self.key).stem

    @property
    def suffix(self) -> str:
        return PurePosixPath(self.key).suffix

    @property
    def parent(self) -> "DataStorePath":
        parent = PurePosixPath(self.key).parent.as_posix()
        return DataStorePath(self.store, "" if parent == "." else parent)

    def resolve(self) -> "DataStorePath":
        return self

    def relative_to(self, other: "DataStorePath") -> PurePosixPath:
        if self.store is not other.store:
            raise ValueError("Paths belong to different data stores")
        return PurePosixPath(self.key).relative_to(PurePosixPath(other.key))

    def exists(self) -> bool:
        return bool(self.key) and self.store.exists(self.key)

    def is_file(self) -> bool:
        return self.exists() and not any(entry.key == self.key and entry.is_dir for entry in self.store.list(self.parent.key))

    def is_dir(self) -> bool:
        return self.exists() and not self.is_file()

    def _children(self, recursive: bool) -> list["DataStorePath"]:
        return [DataStorePath(self.store, entry.key) for entry in self.store.list(self.key, recursive=recursive)]

    def iterdir(self):
        return iter(self._children(False))

    def glob(self, pattern: str):
        return (child for child in self._children(False) if fnmatch.fnmatch(child.name, pattern))

    def rglob(self, pattern: str):
        return (child for child in self._children(True) if fnmatch.fnmatch(child.name, pattern))

    def read_bytes(self) -> bytes:
        return self.store.read_bytes(self.key)[0]

    def read_text(self, *, encoding: str = "utf-8") -> str:
        return self.read_bytes().decode(encoding)

    def write_bytes(self, content: bytes) -> None:
        try:
            _, revision = self.store.read_bytes(self.key)
        except StorageNotFoundError:
            revision = None
        self.store.write_bytes(self.key, content, expected=revision)

    def write_text(self, content: str, *, encoding: str = "utf-8") -> None:
        self.write_bytes(content.encode(encoding))

    def mkdir(self, *, parents: bool = False, exist_ok: bool = False) -> None:
        del exist_ok
        if parents:
            parent = self.parent
            if parent.key and not parent.exists():
                parent.mkdir(parents=True)
        self.store.mkdir(self.key)

    def unlink(self, *, missing_ok: bool = False) -> None:
        if not missing_ok and not self.exists():
            raise FileNotFoundError(self.key)
        self.store.delete(self.key)

    def replace(self, destination: "DataStorePath") -> "DataStorePath":
        self.store.move(self.key, destination.key)
        return destination

    def with_name(self, name: str) -> "DataStorePath":
        return self.parent / name


def read_text(store: DataStore, key: str) -> tuple[str, Revision]:
    content, revision = store.read_bytes(key)
    return content.decode("utf-8"), revision


def write_text(store: DataStore, key: str, content: str, *, expected: Revision | None = None) -> Revision:
    return store.write_bytes(key, content.encode("utf-8"), expected=expected)


def validate_key(key: str, *, allow_empty: bool = False) -> str:
    """Normalize a logical key and reject paths outside the storage root."""
    if not isinstance(key, str):
        raise InvalidStorageKey("Storage key must be a string")
    normalized = key.replace("\\", "/").strip("/")
    if not normalized:
        if allow_empty:
            return ""
        raise InvalidStorageKey("Storage key cannot be empty")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise InvalidStorageKey("Storage key must be relative and cannot traverse directories")
    return path.as_posix()


class LocalDataStore:
    """Atomic local adapter rooted at one Documents directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    def _path(self, key: str, *, allow_empty: bool = False) -> Path:
        normalized = validate_key(key, allow_empty=allow_empty)
        return self.root if not normalized else self.root.joinpath(*normalized.split("/"))

    @staticmethod
    def _revision(content: bytes) -> Revision:
        return Revision(hashlib.sha256(content).hexdigest())

    def read_bytes(self, key: str) -> tuple[bytes, Revision]:
        path = self._path(key)
        try:
            content = path.read_bytes()
        except FileNotFoundError as exc:
            raise StorageNotFoundError(key) from exc
        return content, self._revision(content)

    def write_bytes(self, key: str, content: bytes, *, expected: Revision | None = None) -> Revision:
        path = self._path(key)
        if expected is not None:
            try:
                _, current = self.read_bytes(key)
            except StorageNotFoundError:
                raise StorageConflictError(f"{key} was deleted before it could be saved") from None
            if current != expected:
                raise StorageConflictError(f"{key} changed on another device; reload before saving")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(content)
            Path(temporary).replace(path)
        except OSError:
            Path(temporary).unlink(missing_ok=True)
            raise
        return self._revision(content)

    def list(self, prefix: str = "", *, recursive: bool = False) -> list[Entry]:
        path = self._path(prefix, allow_empty=True)
        if not path.exists():
            return []
        iterator = path.rglob("*") if recursive else path.iterdir()
        entries: list[Entry] = []
        for item in iterator:
            relative = item.relative_to(self.root).as_posix()
            if item.is_dir():
                entries.append(Entry(relative, is_dir=True))
            else:
                content = item.read_bytes()
                entries.append(Entry(relative, is_dir=False, size=len(content), revision=self._revision(content)))
        return entries

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def mkdir(self, key: str) -> None:
        self._path(key).mkdir(parents=True, exist_ok=True)

    def delete(self, key: str, *, recursive: bool = False) -> None:
        path = self._path(key)
        if path.is_dir():
            if recursive:
                shutil.rmtree(path)
            else:
                path.rmdir()
        else:
            path.unlink(missing_ok=True)

    def move(self, source: str, destination: str) -> None:
        src, dst = self._path(source), self._path(destination)
        if not src.exists():
            raise StorageNotFoundError(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)

    def copy(self, source: str, destination: str) -> None:
        src, dst = self._path(source), self._path(destination)
        if not src.exists():
            raise StorageNotFoundError(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


class WebDavDataStore:
    """WebDAV adapter rooted at Nextcloud's Documents collection."""

    def __init__(self, base_url: str, user: str, password: str) -> None:
        self._base_url = base_url.rstrip("/")
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        self._authorization = f"Basic {token}"

    @classmethod
    def from_environment(cls) -> "WebDavDataStore":
        url = os.getenv("GIGACHAD_WEBDAV_URL")
        user = os.getenv("GIGACHAD_WEBDAV_USER")
        password = os.getenv("GIGACHAD_WEBDAV_PASSWORD")
        if not all((url, user, password)):
            raise StorageError("Set GIGACHAD_WEBDAV_URL, _USER, and _PASSWORD for WebDAV storage.")
        return cls(url, user, password)

    def _url(self, key: str, *, allow_empty: bool = False) -> str:
        normalized = validate_key(key, allow_empty=allow_empty)
        return self._base_url if not normalized else f"{self._base_url}/{quote(normalized, safe='/')}"

    def _request(self, method: str, key: str, *, body: bytes | None = None, headers: dict[str, str] | None = None, allow_empty: bool = False) -> tuple[bytes, str | None]:
        request = Request(
            self._url(key, allow_empty=allow_empty), data=body,
            headers={"Authorization": self._authorization, **(headers or {})}, method=method,
        )
        try:
            with urlopen(request, timeout=20) as response:  # noqa: S310 -- endpoint is explicit user config.
                return response.read(), response.headers.get("ETag", response.headers.get("OC-Etag"))
        except HTTPError as exc:
            if exc.code == 404:
                raise StorageNotFoundError(key) from exc
            if exc.code == 412:
                raise StorageConflictError(f"{key} changed on another device; reload before saving") from exc
            raise StorageError(f"WebDAV {method} failed for {key}: HTTP {exc.code}") from exc
        except URLError as exc:
            raise StorageError(f"WebDAV {method} failed for {key}: {exc.reason}") from exc

    def read_bytes(self, key: str) -> tuple[bytes, Revision]:
        body, etag = self._request("GET", key)
        if not etag:
            raise StorageError(f"WebDAV GET for {key} did not return an ETag")
        return body, Revision(etag)

    def write_bytes(self, key: str, content: bytes, *, expected: Revision | None = None) -> Revision:
        # A caller that has read a file supplies its ETag.  Blind writes are retained
        # for immutable/new binary imports; editable API routes always pass a revision.
        headers = {"If-Match": expected.token} if expected else {}
        _, etag = self._request("PUT", key, body=content, headers=headers)
        if etag:
            return Revision(etag)
        _, revision = self.read_bytes(key)
        return revision

    def list(self, prefix: str = "", *, recursive: bool = False) -> list[Entry]:
        body, _ = self._request("PROPFIND", prefix, headers={"Depth": "infinity" if recursive else "1"}, allow_empty=True)
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            raise StorageError("WebDAV PROPFIND returned invalid XML") from exc
        namespace = "{DAV:}"
        base = self._url("", allow_empty=True).rstrip("/")
        entries: list[Entry] = []
        for response in root.findall(f"{namespace}response"):
            href = response.findtext(f"{namespace}href", "").rstrip("/")
            if href.rstrip("/") == base:
                continue
            key = href.split("/", 3)[-1] if href.startswith("/") else href
            # Nextcloud returns a URL path; keep only its suffix beneath configured base.
            configured_path = self._base_url.split("://", 1)[-1].split("/", 1)[-1].rstrip("/")
            if configured_path and configured_path in key:
                key = key.split(configured_path, 1)[1].strip("/")
            if not key:
                continue
            prop = response.find(f"{namespace}propstat/{namespace}prop")
            is_dir = prop is not None and prop.find(f"{namespace}resourcetype/{namespace}collection") is not None
            length = prop.findtext(f"{namespace}getcontentlength") if prop is not None else None
            etag = prop.findtext(f"{namespace}getetag") if prop is not None else None
            entries.append(Entry(key, is_dir=is_dir, size=int(length) if length and length.isdigit() else None, revision=Revision(etag) if etag else None))
        return entries

    def exists(self, key: str) -> bool:
        try:
            self._request("PROPFIND", key, headers={"Depth": "0"})
        except StorageNotFoundError:
            return False
        return True

    def mkdir(self, key: str) -> None:
        try:
            self._request("MKCOL", key)
        except StorageError as exc:
            # Nextcloud reports an existing collection as 405; a successful exists check makes this idempotent.
            if not self.exists(key):
                raise exc

    def delete(self, key: str, *, recursive: bool = False) -> None:
        del recursive  # WebDAV DELETE handles collections recursively.
        try:
            self._request("DELETE", key)
        except StorageNotFoundError:
            return

    def _copy_or_move(self, method: str, source: str, destination: str) -> None:
        self._request(method, source, headers={"Destination": self._url(destination), "Overwrite": "F"})

    def move(self, source: str, destination: str) -> None:
        self._copy_or_move("MOVE", source, destination)

    def copy(self, source: str, destination: str) -> None:
        self._copy_or_move("COPY", source, destination)
