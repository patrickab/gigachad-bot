"""Where a stored document lives (the notes collection or one project's documents).

Also the writes, moves and renames that keep project references in step.
"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from lib import latex
from lib.asset_store import AssetStore
from lib.attachment_materialize import sync_library_pdf
from lib.data_store import DataStore, InvalidStorageKey, validate_key, write_text
from lib.file_roots import FileRoots
from lib.project_store import ProjectStore
from lib.references import References, Released, is_project_document
from lib.storage_namespace import (
    GRAPH, NOTE, PDFS, library_pdf, project_attachment_prefix, project_document, project_document_prefix,
)

Relocate = Callable[[str, str], None]
"""Point every reference to a stored key or disk path at its new key or path, then rekey or delete the stored one (`PostgresKeyRelocator.relocate`)."""


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


def add_extraction(store: ProjectStore, docs: DataStore, slug: str, stem: str, markdown: str) -> None:
    """List the MinerU *markdown* of a PDF in project *slug*'s files as `<stem>.md`, keeping a document already stored under that name."""
    key = project_document(slug, f"{stem}.md")
    if not docs.exists(key):
        write_text(docs, key, markdown)
    store.add_file(slug, key)


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


def renamed(file_name: str, name: str) -> str:
    """*name* as the new name of the file *file_name*, which keeps its suffixes unless *name* ends in its last one.

    Raises InvalidDocumentName for an empty name, a path separator or a control character.
    """
    name = name.strip()
    if not name:
        raise InvalidDocumentName("Document name is required")
    if "/" in name or "\\" in name or name in {".", ".."} or any(ord(character) < 32 for character in name):
        raise InvalidDocumentName("A file name cannot hold a path separator")
    suffix = Path(file_name).suffix
    if suffix and not name.lower().endswith(suffix.lower()):
        name += "".join(Path(file_name).suffixes)
    return name


def is_name_taken(store: ProjectStore, scope: DocumentScope, path: str, name: str) -> bool:
    """Whether another file of *scope*'s project is shown as *name*, ignoring case."""
    if not scope.slug:
        return False
    return any(other != path and shown.lower() == name.lower() for other, shown in store.list_named_files(scope.slug))


def rename(store: ProjectStore, docs: DataStore, relocate: Relocate, scope: DocumentScope, key: str, name: str) -> tuple[str, str]:
    """Rename the stored document *key* of *scope* to *name* (see `renamed`), returning its new key and file name.

    - The key moves, so its file name is the shown name, and a display name any project gave it is dropped
    - Chats, canvases and projects referencing the key follow it, an existing document of that name raises DocumentNameConflict
    - Only notes and project documents can be renamed: anything else raises InvalidDocumentName
    """
    if not scope.owns(store, key):
        raise DocumentNotFound(key)
    if not (under(key, NOTE) or is_project_document(key)):
        raise InvalidDocumentName("This file cannot be renamed")
    if not docs.is_file(key):
        raise DocumentNotFound(key)
    destination = str(PurePosixPath(key).with_name(renamed(PurePosixPath(key).name, name)))
    try:
        validate_key(destination)
    except InvalidStorageKey as exc:
        raise InvalidDocumentName("Invalid file name") from exc
    if destination != key:
        if docs.exists(destination) or is_name_taken(store, scope, key, PurePosixPath(destination).name):
            raise DocumentNameConflict(destination)
        relocate(key, destination)
    store.forget_file_names(destination)
    return destination, PurePosixPath(destination).name


def rename_on_disk(store: ProjectStore, relocate: Relocate, roots: FileRoots, scope: DocumentScope, path: str, name: str) -> Path:
    """Rename the file at the absolute *path* inside its folder (see `renamed`), then repoint every reference to it, returning the new path.

    - Allowed where saving is: *roots* decides, and a project scope must reference the file
    - An existing file of that name raises DocumentNameConflict. The move is undone when the references cannot follow
    - References: project file lists, `workbench/sessions.json`, chats, canvases and their mutation log, notes
    - A `.tex` file leaves its build outputs behind, so they are discarded and the next build starts clean
    """
    scope.require(store)
    if scope.slug and path not in set(store.list_files(scope.slug)):
        raise DocumentNotFound(path)
    source = Path(path)
    target = source.with_name(renamed(source.name, name))
    if target == source:
        return source
    if is_name_taken(store, scope, path, target.name):
        raise DocumentNameConflict(str(target))
    try:
        roots.rename(path, target.name)
    except FileExistsError as exc:
        raise DocumentNameConflict(str(target)) from exc
    try:
        relocate(path, str(target))
    except BaseException:
        target.rename(source)
        raise
    store.forget_file_names(str(target))
    if source.suffix == ".tex":
        latex.discard_build(source)
        latex.discard_build(target)
    return target


def follow_library_pdf(assets: AssetStore, references: References, old_name: str, new_name: str, content: Callable[[], bytes]) -> None:
    """Give the renamed PDF its library copy under *new_name* and let go of the one under *old_name* once nothing uses it.

    A PDF that was never synced into the library gets no copy.
    """
    if old_name == new_name or not new_name.lower().endswith(".pdf") or assets.stat(library_pdf(old_name)) is None:
        return
    sync_library_pdf(assets, new_name, content())
    references.release(Released(refs={library_pdf(old_name)}))


def file_name(store: ProjectStore, docs: DataStore, key: str) -> str:
    """The file name the stored project file *key* takes on disk: the name it is shown under.

    - Canvases live in the app, so they raise DocumentNotFound like any key that is not a stored project file
    """
    names = dict(store.list_all_named_files())
    if key not in names or not is_project_document(key) or key.endswith(".canvas") or not docs.is_file(key):
        raise DocumentNotFound(key)
    return Path(names[key]).name


def move_to_folder(store: ProjectStore, docs: DataStore, relocate: Relocate, key: str, folder: Path) -> Path:
    """Write the stored project file *key* into *folder* and relocate every reference to it there, returning the new file.

    - From then on it is a mounted file: projects, chats and canvases read it from disk, and it is never deleted
    - The file takes the name the document is shown under (`file_name`), an existing file of that name raises DocumentNameConflict
    """
    target = folder / file_name(store, docs, key)
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
