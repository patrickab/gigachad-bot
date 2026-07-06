"""File-vault routes — the API owns every filesystem access.

The UI never touches a vault directly: it lists files (`/files`), previews one
(`/file`), and *attaches* a chosen file (`/attach`). Attach is a **live
reference** — nothing is copied; the returned `path` is stored on the
Attachment and content is read from (and written back to) the actual file.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.routes.deps import get_file_vault
from lib import document_library as lib_docs
from lib.attachment_materialize import materialize
from lib.file_vault import FileVault

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/filevaults", tags=["filevaults"])


class VaultFile(BaseModel):
    path: str
    name: str


class VaultListResponse(BaseModel):
    enabled: bool
    files: list[VaultFile]


class VaultNode(BaseModel):
    name: str
    path: str
    type: str
    project: str | None = None
    children: list["VaultNode"] | None = None


class VaultTreeResponse(BaseModel):
    enabled: bool
    tree: list[VaultNode]


class RootBody(BaseModel):
    path: str
    project: str | None = None


class WriteBody(BaseModel):
    path: str
    content: str


class VaultFileContent(BaseModel):
    path: str
    content: str


class AttachResult(BaseModel):
    name: str
    mime: str
    path: str
    content: str | None = None
    parsedMd: str | None = None


@router.get("/files", response_model=VaultListResponse)
async def list_files(vault: FileVault = Depends(get_file_vault)) -> VaultListResponse:
    return VaultListResponse(enabled=vault.enabled, files=vault.list_files())


class ProjectDocumentOut(BaseModel):
    path: str
    name: str
    mime: str


class ProjectDocumentsResponse(BaseModel):
    documents: list[ProjectDocumentOut]


@router.get("/project-documents", response_model=ProjectDocumentsResponse)
async def project_documents(slug: str = Query(...), vault: FileVault = Depends(get_file_vault)) -> ProjectDocumentsResponse:
    """Files from vaults mounted to *slug*, shaped like project documents.

    Surfaced so the chat sidebar can list mounted-vault files alongside library
    documents. They attach as live references (no copy). A vault PDF whose
    cloud-library copy already exists is reported at the *library* path — the
    parsed cloud copy takes precedence over the vault original, so the sidebar
    never lists the same PDF twice (vault row + library row).
    """
    docs: list[ProjectDocumentOut] = []
    for f in vault.list_files_for_project(slug):
        p = Path(f["path"])
        if p.suffix.lower() == ".pdf":
            library_pdf = lib_docs.LIBRARY_DIR / p.name
            if library_pdf.is_file():
                p = library_pdf
        docs.append(ProjectDocumentOut(path=str(p), name=p.name, mime=lib_docs.mime_for(p)))
    return ProjectDocumentsResponse(documents=docs)


@router.get("/tree", response_model=VaultTreeResponse)
async def list_tree(vault: FileVault = Depends(get_file_vault)) -> VaultTreeResponse:
    return VaultTreeResponse(enabled=vault.enabled, tree=vault.tree())


@router.post("/roots", response_model=VaultTreeResponse)
async def add_root(body: RootBody, vault: FileVault = Depends(get_file_vault)) -> VaultTreeResponse:
    try:
        vault.add_root(body.path, project=body.project)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return VaultTreeResponse(enabled=vault.enabled, tree=vault.tree())


@router.delete("/roots", response_model=VaultTreeResponse)
async def remove_root(path: str = Query(...), vault: FileVault = Depends(get_file_vault)) -> VaultTreeResponse:
    vault.remove_root(path)
    return VaultTreeResponse(enabled=vault.enabled, tree=vault.tree())


class MountpointBody(BaseModel):
    path: str


@router.post("/mountpoints", response_model=VaultTreeResponse)
async def add_mountpoint(
    body: MountpointBody,
    vault_path: str = Query(..., alias="vault"),
    vault: FileVault = Depends(get_file_vault),
) -> VaultTreeResponse:
    try:
        vault.add_mountpoint(vault_path, body.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return VaultTreeResponse(enabled=vault.enabled, tree=vault.tree())


@router.delete("/mountpoints", response_model=VaultTreeResponse)
async def remove_mountpoint(
    vault_path: str = Query(..., alias="vault"),
    path: str = Query(...),
    vault: FileVault = Depends(get_file_vault),
) -> VaultTreeResponse:
    vault.remove_mountpoint(vault_path, path)
    return VaultTreeResponse(enabled=vault.enabled, tree=vault.tree())


@router.get("/file", response_model=VaultFileContent)
async def read_file(path: str = Query(...), vault: FileVault = Depends(get_file_vault)) -> VaultFileContent:
    try:
        return VaultFileContent(path=path, content=vault.read(path))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/file")
async def write_file(body: WriteBody, vault: FileVault = Depends(get_file_vault)) -> dict[str, bool]:
    try:
        vault.write(body.path, body.content)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@router.get("/rendered", response_model=VaultFileContent)
async def read_rendered(path: str = Query(...), vault: FileVault = Depends(get_file_vault)) -> VaultFileContent:
    try:
        raw = vault.read(path)
        return VaultFileContent(path=path, content=vault.resolve_wiki_content(raw, path))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/attach", response_model=AttachResult)
async def attach_file(
    path: str = Query(...),
    vault: FileVault = Depends(get_file_vault),
) -> AttachResult:
    """Attach as a live reference: validate the path, return current content.

    Idempotent and copy-free — the frontend calls this again at send time to
    refresh content (and pick up a finished PDF extraction).

    PDFs are the one exception to the copy-free rule: a vault PDF is *promoted*
    into the cloud library (``DIRECTORY_OUTPUT_PDF``) on first attach and the
    library copy takes precedence thereafter. This dedupes against any
    same-named PDF already parsed by MinerU, so a vault never shadows the
    canonical library copy. A library PDF path is also accepted here so the
    send-time refresh (which re-calls this endpoint with the canonical path)
    keeps working for promoted PDFs. Non-PDF vault files stay pure live
    references.
    """
    resolved = Path(path).expanduser().resolve()
    library = lib_docs.LIBRARY_DIR.resolve()
    in_library = resolved.is_relative_to(library) and resolved.is_file()
    mime = lib_docs.mime_for(resolved)

    if resolved.suffix.lower() == ".pdf":
        if in_library:
            canonical = resolved
        else:
            try:
                vault_resolved = vault.resolve(path)
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            library_pdf = lib_docs.LIBRARY_DIR / vault_resolved.name
            # Cloud copy takes precedence if it already exists (don't clobber a
            # possibly-parsed library PDF with a same-named vault file). Otherwise
            # promote the vault PDF into the library — filename is identity, so
            # this is an overwrite-by-name organize, never a `<name> (n).pdf` dupe.
            if library_pdf.is_file():
                canonical = library_pdf
            else:
                try:
                    canonical = lib_docs.organize_file(vault_resolved)
                except Exception:
                    log.exception("Failed to promote vault PDF %s into library", vault_resolved)
                    canonical = vault_resolved  # fall back to the vault path
        parsed = materialize(canonical).parsed_md
        return AttachResult(name=canonical.name, mime=mime, path=str(canonical), parsedMd=parsed)

    try:
        vault_resolved = vault.resolve(path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    result = AttachResult(name=vault_resolved.name, mime=mime, path=str(vault_resolved))
    try:
        result.content = vault.read(path)
    except UnicodeDecodeError:
        result.content = None
    return result
