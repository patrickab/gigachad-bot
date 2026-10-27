"""Where a stored document lives (the notes collection or one project's documents).

Also the writes, moves and renames that keep project references in step.
"""

from dataclasses import dataclass
import logging
from pathlib import Path, PurePosixPath

from config import DOCUMENTS
from lib.asset_store import AssetStore
from lib.data_store import DataStore, StorageConflictError, StorageNotFoundError, validate_key, write_text
from lib.project_store import ProjectStore
from lib.storage_namespace import GRAPH, NOTE, PDFS, project_attachment_prefix, project_document, project_document_prefix

log = logging.getLogger(__name__)


class DocumentNotFound(Exception):
    """The document is unknown, outside the scope it was named in, or not stored."""


class DocumentNameConflict(Exception):
    """Another document already has the destination name."""


class ProjectNotFound(Exception):
    """A project scope names a project the catalog does not have."""


class InvalidDocumentName(ValueError):
    """A rename target without a usable file name."""


def under(key: str, prefix: str) -> bool:
    """Whether *key* is *prefix* itself or lies below it."""
    return key == prefix or key.startswith(f"{prefix}/")


@dataclass(frozen=True)
class DocumentScope:
    """The notes collection (empty slug) or one project's documents: where its files are stored and how they are referenced.

    Notes are referenced by their location alone, a project additionally lists its files in its catalog entry.
    """

    slug: str

    @property
    def prefix(self) -> str:
        return project_document_prefix(self.slug) if self.slug else NOTE

    def key_for(self, name: str) -> str:
        return f"{self.prefix}/{name}"

    def require(self, store: ProjectStore) -> None:
        """Raise ProjectNotFound("Project not found") when the project is unknown."""
        if self.slug:
            try:
                store.require(self.slug)
            except FileNotFoundError as exc:
                raise ProjectNotFound("Project not found") from exc

    def owns(self, store: ProjectStore, key: str) -> bool:
        """Whether *key* belongs here: under the notes prefix, or referenced by the project.

        An unknown project raises ProjectNotFound naming the slug.
        """
        if not self.slug:
            return under(key, NOTE)
        try:
            return key in set(store.list_files(self.slug))
        except FileNotFoundError as exc:
            raise ProjectNotFound(str(exc)) from exc

    def link(self, store: ProjectStore, key: str) -> None:
        if self.slug:
            store.add_file(self.slug, key)

    def unlink(self, store: ProjectStore, key: str) -> None:
        if self.slug:
            store.remove_file(self.slug, key)

    def relink(self, store: ProjectStore, old: str, new: str) -> None:
        if self.slug:
            store.replace_file(self.slug, old, new)

    def delete_stored(self, docs: DataStore, path: str) -> None:
        """Delete *path* only when it is a valid key of a file stored under this scope."""
        try:
            key = validate_key(path)
        except ValueError:
            return
        if under(key, self.prefix) and docs.is_file(key):
            docs.delete(key)


def is_known(store: ProjectStore, key: str) -> bool:
    """Whether *key* is a library PDF, an architecture source or referenced by any project.

    Anything else must look absent, so another user's documents stay undiscoverable.
    """
    return under(key, PDFS) or under(key, GRAPH) or key in set(store.list_all_files())


def is_stored(docs: DataStore, assets: AssetStore, key: str) -> bool:
    """Whether *key* is a stored document file or holds an asset."""
    return docs.is_file(key) or bool(assets.list(key))


def write(store: ProjectStore, docs: DataStore, assets: AssetStore, scope: DocumentScope, name: str, content: str) -> str:
    """Create or overwrite the text document *name* in *scope*, returning its key.

    A project document name is validated as one key segment, and its chat attachment copies with that file name are refreshed.
    """
    if not scope.slug:
        key = scope.key_for(name)
        write_text(docs, key, content)
        return key
    scope.require(store)
    key = project_document(scope.slug, name)
    write_text(docs, key, content)
    scope.link(store, key)
    encoded = content.encode("utf-8")
    for asset in assets.list(project_attachment_prefix(scope.slug), kind="upload"):
        if PurePosixPath(asset.logical_path).name == name:
            assets.write("upload", asset.logical_path, encoded)
    return key


def move(store: ProjectStore, docs: DataStore, key: str, source: DocumentScope, target: DocumentScope) -> str:
    """Move the stored document *key* from *source* to *target*, keeping its name, returning its new key."""
    if not source.owns(store, key) or not docs.is_file(key):
        raise DocumentNotFound(key)
    target.require(store)
    destination = target.key_for(PurePosixPath(key).name)
    if destination != key:
        if docs.exists(destination):
            raise DocumentNameConflict(destination)
        docs.move(key, destination)
    source.unlink(store, key)
    target.link(store, destination)
    return destination


def rename(store: ProjectStore, docs: DataStore, assets: AssetStore, scope: DocumentScope, key: str, name: str) -> str:
    """Rename *key* (a stored document of *scope*, else a library PDF) to *name*, keeping its suffixes, returning its new key."""
    if not scope.owns(store, key):
        raise DocumentNotFound(key)
    name = Path(name).name.strip()
    if not name:
        raise InvalidDocumentName("Document name is required")
    suffix = "".join(Path(PurePosixPath(key).name).suffixes)
    if suffix and not name.endswith(suffix):
        name += suffix
    if docs.is_file(key):
        return _rename_stored(store, docs, scope, key, name)
    return _rename_pdf(store, assets, key, name)


def _rename_stored(store: ProjectStore, docs: DataStore, scope: DocumentScope, key: str, name: str) -> str:
    """Rename a stored document inside its scope, repointing the project's reference."""
    if not under(key, scope.prefix):
        raise DocumentNotFound(key)
    destination = scope.key_for(name)
    if destination == key:
        return key
    if docs.exists(destination):
        raise DocumentNameConflict(destination)
    docs.move(key, destination)
    scope.relink(store, key, destination)
    return destination


def _rename_pdf(store: ProjectStore, assets: AssetStore, key: str, name: str) -> str:
    """Rename a library PDF, repointing every project that references it."""
    if not under(key, PDFS):
        raise DocumentNotFound(key)
    try:
        asset = assets.read(key)
    except StorageNotFoundError as exc:
        raise DocumentNotFound(key) from exc
    if asset.kind != "pdf":
        raise DocumentNotFound(key)

    destination = f"{PDFS}/{name}"
    if destination == key:
        return key
    try:
        renamed = assets.move(key, destination)
    except StorageConflictError as exc:
        raise DocumentNameConflict(destination) from exc

    # Keep Nextcloud's convenience mirror aligned with the database authority.
    Path(DOCUMENTS, *key.split("/")).unlink(missing_ok=True)
    try:
        assets.mirror(renamed, DOCUMENTS)
    except OSError:
        log.exception("Failed to mirror renamed PDF %s into Nextcloud", destination)
    store.replace_file_everywhere(key, destination)
    return destination
