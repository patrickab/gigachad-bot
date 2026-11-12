// Architecture model routes under /architecture.
import { toQuery } from "../urls"
import { del, post, put, request } from "./http"
import type { ArchitectureOperation, ArchitectureWorkspace } from "@/lib/domain/architecture"

// Architecture writes carry no revision. Structured operations are revalidated
// against the server's current model: one that no longer fits is rejected with a
// 400 and a reason. A source save replaces the file with the supplied text,
// whatever it held meanwhile, provided the model then parses without errors.
const architecturePath = (slug: string) => `/architecture/${encodeURIComponent(slug)}`

/** The whole model: every view, with saved positions applied; empty for a project without an architecture. */
export function readArchitecture(slug: string): Promise<ArchitectureWorkspace> {
  return request<ArchitectureWorkspace>(architecturePath(slug))
}

/** `file` is the editing window's source, where top-level elements drawn there are written; null for ops that name their own file, e.g. creating a system. */
export function applyArchitectureOperations(slug: string, ops: ArchitectureOperation[], file: string | null): Promise<ArchitectureWorkspace> {
  return post<ArchitectureWorkspace>(`${architecturePath(slug)}/ops`, { ops, file })
}

/** Replaces `path` with `content`; rejected unless the model then parses without errors. */
export function writeArchitectureSource(slug: string, path: string, content: string): Promise<ArchitectureWorkspace> {
  return put<ArchitectureWorkspace>(`${architecturePath(slug)}/source`, { path, content })
}

/** Adds a source file exactly as named, empty: for writing `.c4` by hand. */
export function createArchitectureSource(slug: string, path: string): Promise<ArchitectureWorkspace> {
  return post<ArchitectureWorkspace>(`${architecturePath(slug)}/source`, { path })
}

/** Deletes the files together, rejected while the remaining files still use what they declare. */
export function deleteArchitectureSources(slug: string, paths: string[]): Promise<ArchitectureWorkspace> {
  return del<ArchitectureWorkspace>(`${architecturePath(slug)}/source${toQuery({ path: paths })}`)
}
