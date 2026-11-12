// Project document routes under /documents.
import { toQuery } from "../urls"
import { del, fileForm, post, request } from "./http"
import type { ProjectDocument } from "@/lib/domain/projects"

type DocumentList = { documents: ProjectDocument[] }
const documentsOf = (data: DocumentList) => data.documents

/** Add file-root files to a project's documents: a folder holding a root `.tex` stays one folder entry, other folders expand to their valid files. */
export async function addRootFilesToProject(slug: string, paths: string[]): Promise<ProjectDocument[]> {
  return post<DocumentList>("/fileroots/add-to-project", { slug, paths }).then(documentsOf)
}

export async function listProjectDocuments(slug: string, signal?: AbortSignal): Promise<ProjectDocument[]> {
  return request<DocumentList>(`/documents${toQuery({ slug })}`, { signal }).then(documentsOf)
}

export async function listAllDocuments(): Promise<ProjectDocument[]> {
  return request<DocumentList>("/documents/all").then(documentsOf)
}

export async function listNotes(signal?: AbortSignal): Promise<ProjectDocument[]> {
  return request<DocumentList>("/documents/notes", { signal }).then(documentsOf)
}

export async function addDocument(slug: string, path: string): Promise<ProjectDocument> {
  return post<ProjectDocument>(`/documents/add${toQuery({ slug })}`, { path })
}

/** Delete `path` from project `slug`; resolves with the keys and absolute paths that are gone (a mounted file only unlinked stays out). */
export async function removeDocument(slug: string, path: string): Promise<string[]> {
  return (await del<{ removed: string[] }>(`/documents${toQuery({ slug, path })}`)).removed
}

export async function moveDocument(path: string, fromSlug: string, toSlug: string): Promise<ProjectDocument> {
  return post<ProjectDocument>("/documents/move", { path, from_slug: fromSlug, to_slug: toSlug })
}

/** Move the stored project file `path` into the file-root `folder`, every project then references the file on disk. */
export async function moveDocumentToFolder(path: string, folder: string): Promise<ProjectDocument> {
  return post<ProjectDocument>("/documents/move-to-folder", { path, folder })
}

/** The renamed document, and the absolute paths of other files the rename rewrote to follow it (references to it). */
export type RenamedDocument = ProjectDocument & { changed: string[] }

export async function renameDocument(slug: string, path: string, name: string): Promise<RenamedDocument> {
  return post<RenamedDocument>("/documents/rename", { slug, path, name })
}

export async function writeDocument(slug: string, name: string, content: string = ""): Promise<ProjectDocument> {
  return post<ProjectDocument>("/documents/write", { slug, name, content })
}

export async function writeBinaryDocument(slug: string, filename: string, blob: Blob): Promise<ProjectDocument> {
  return request<ProjectDocument>(`/documents/write-binary${toQuery({ slug })}`, {
    method: "POST",
    body: fileForm(blob, filename),
  })
}

/** Store a rendered canvas (.jpg) as a database asset. */
export async function storeDrawing(filename: string, blob: Blob): Promise<void> {
  await request("/documents/store-drawing", { method: "POST", body: fileForm(blob, filename) })
}
