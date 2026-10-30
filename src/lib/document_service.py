"""Where a stored document lives (the notes collection or one project's documents).

Also the writes, moves and renames that keep project references in step.
"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from lib.asset_store import AssetStore
from lib.data_store import DataStore, validate_key, write_text
from lib.project_store import ProjectStore
from lib.references import Released, is_project_document
from lib.storage_namespace import GRAPH, NOTE, PDFS, project_attachment_prefix, project_document, project_document_prefix

Relocate = Callable[[str, str], None]
"""Point every reference to a stored key at a new key or disk path, then rekey or delete it (`PostgresKeyRelocator.relocate`)."""


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

    def link(self, store: ProjectStore, key: str, name: str | None = None) -> None:
        """Reference *key* from the project, shown as *name* when given."""
        if self.slug:
            store.add_file(self.slug, key)
            if name is not None:
                store.set_file_name(self.slug, key, name)

    def unlink(self, store: ProjectStore, key: str) -> None:
        if self.slug:
            store.remove_file(self.slug, key)

    def shown_name(self, store: ProjectStore, key: str) -> str:
        """The name *key* is shown under here: the project's display name, else its file name."""
        if self.slug:
            return dict(store.list_named_files(self.slug)).get(key, PurePosixPath(key).name)
        return PurePosixPath(key).name

    def removal(self, path: str) -> Released:
        """What removing *path* from here releases: a note is deleted outright, a project file once nothing else references it."""
        if self.slug:
            return Released(refs={path})
        try:
            key = validate_key(path)
        except ValueError:
            return Released()
        return Released(owned={key}) if under(key, NOTE) else Released()


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


def move(store: ProjectStore, docs: DataStore, relocate: Relocate, key: str, source: DocumentScope, target: DocumentScope) -> str:
    """Move the stored document *key* from *source* to *target*, keeping the name it is shown under, returning its new key.

    - A project keeps the file name in the key and the shown name as display name, a note is stored under the shown name
    - Chats, canvases and other projects referencing *key* follow it to the new key
    """
    if not source.owns(store, key) or not docs.is_file(key):
        raise DocumentNotFound(key)
    target.require(store)
    shown = source.shown_name(store, key)
    destination = target.key_for(PurePosixPath(key).name if target.slug else shown)
    if destination != key:
        if docs.exists(destination):
            raise DocumentNameConflict(destination)
        relocate(key, destination)
    source.unlink(store, destination)
    target.link(store, destination, shown)
    return destination


def rename(store: ProjectStore, docs: DataStore, scope: DocumentScope, key: str, name: str) -> tuple[str, str]:
    """Rename *key* of *scope* to *name*, keeping its suffixes, returning its key and the name it is shown under.

    A project file keeps its key and only takes *name* as display name, a note moves to the key *name* gives it.
    """
    if not scope.owns(store, key):
        raise DocumentNotFound(key)
    name = Path(name).name.strip()
    if not name:
        raise InvalidDocumentName("Document name is required")
    suffix = "".join(Path(PurePosixPath(key).name).suffixes)
    if suffix and not name.endswith(suffix):
        name += suffix
    if scope.slug:
        store.set_file_name(scope.slug, key, name)
        return key, name
    if not docs.is_file(key):
        raise DocumentNotFound(key)
    destination = scope.key_for(name)
    if destination != key:
        if docs.exists(destination):
            raise DocumentNameConflict(destination)
        docs.move(key, destination)
    return destination, name


def move_to_folder(store: ProjectStore, docs: DataStore, relocate: Relocate, key: str, folder: Path) -> Path:
    """Write the stored project file *key* into *folder* and relocate every reference to it there, returning the new file.

    - From then on it is a mounted file: projects, chats and canvases read it from disk, and it is never deleted
    - The file takes the name the document is shown under, an existing file of that name raises DocumentNameConflict
    - Canvases live in the app, so they raise DocumentNotFound like any key that is not a stored project file
    """
    names = dict(store.list_all_named_files())
    if key not in names or not is_project_document(key) or key.endswith(".canvas") or not docs.is_file(key):
        raise DocumentNotFound(key)
    target = folder / Path(names[key]).name
    content, _ = docs.read_bytes(key)
    try:
        with target.open("xb") as file:
            file.write(content)
    except FileExistsError as exc:
        raise DocumentNameConflict(str(target)) from exc
    try:
        relocate(key, str(target))
    except Exception:
        target.unlink()
        raise
    return target
