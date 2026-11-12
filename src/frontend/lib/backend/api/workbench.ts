// Workbench preparation, file session and image routes.
import { toQuery } from "../urls"
import { fileForm, post, put, request } from "./http"
import type { StoredImage } from "@/lib/domain/latex"

/** What `prepareWorkbench` resolves: the file's absolute disk path and the directory its omp sessions run in. The response's `kind` is not read, the client derives it from the path. */
export interface PreparedWorkbench {
  path: string
  cwd: string
}

/** Get the workbench file *path* ready: a stored document moves to disk first, its new path comes back. */
export function prepareWorkbench(path: string): Promise<PreparedWorkbench> {
  return post<PreparedWorkbench>("/workbench/prepare", { path })
}

/**
 * Create workbench document *name* (`.tex` or `.md`) of project *slug* on disk, listed in its files. Resolves to its path.
 *
 * - *template* names a LaTeX template set: the document becomes a presentation folder `<stem>/main.tex` with the theme beside it
 */
export async function createWorkbenchDocument(slug: string, name: string, template?: string): Promise<string> {
  return (await post<{ path: string }>("/workbench/create", { slug, name, ...(template ? { template } : {}) })).path
}

/**
 * The omp sessions of one file's chat: `active` is null while a new session starts, `sessionIds` lists every session the file had, first linked first.
 *
 * - `moved` are sessions the file moved away from (a rename), each has to be told its new path once
 */
export interface FileSessions {
  active: string | null
  sessionIds: string[]
  moved: string[]
}

interface LinkedSessions {
  session_id: string | null
  session_ids: string[]
  moved: string[]
}

function toFileSessions(linked: LinkedSessions): FileSessions {
  return { active: linked.session_id, sessionIds: linked.session_ids, moved: linked.moved }
}

/** The omp sessions linked to the file at absolute *path*. */
export async function loadFileSessions(path: string): Promise<FileSessions> {
  return toFileSessions(await request<LinkedSessions>(`/workbench/session${toQuery({ path })}`))
}

/** Make *sessionId* the active session of the file at absolute *path*, null starts a new one. Resolves to the file's sessions. */
export async function saveFileSession(path: string, sessionId: string | null): Promise<FileSessions> {
  return toFileSessions(await put<LinkedSessions>("/workbench/session", { path, session_id: sessionId }))
}

/** *sessionId* was told where the file at absolute *path* is now. Resolves to the file's sessions. */
export async function informFileSession(path: string, sessionId: string): Promise<FileSessions> {
  return toFileSessions(await post<LinkedSessions>("/workbench/session/informed", { path, session_id: sessionId }))
}

/** Store a pasted PNG in the image folder next to the workbench file at absolute *path*, resolving to it, its placeholder path and the line to insert. */
export function storeWorkbenchImage(path: string, image: Blob): Promise<StoredImage> {
  return request<StoredImage>(`/workbench/image${toQuery({ path })}`, { method: "POST", body: fileForm(image, "image.png") })
}
