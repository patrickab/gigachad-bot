// Workbench preparation, Pi chat and image routes.
import { toQuery } from "../urls"
import { fileForm, post, request } from "./http"
import type { PiSessionInfo, PiSessionOptions } from "./pi"
import type { StoredImage } from "@/lib/domain/latex"

/** What `prepareWorkbench` resolves: the file's absolute disk path. The response's `kind` is not read, the client derives it from the path. */
export interface PreparedWorkbench {
  path: string
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

/** Store a pasted PNG in the image folder next to the workbench file at absolute *path*, resolving to it, its placeholder path and the line to insert. */
export function storeWorkbenchImage(path: string, image: Blob): Promise<StoredImage> {
  return request<StoredImage>(`/workbench/image${toQuery({ path })}`, { method: "POST", body: fileForm(image, "image.png") })
}

/** What a workbench chat is about, from its path: the system prompt and folder follow it. */
export type WorkbenchChatKind = "latex" | "markdown" | "pdf" | "canvas" | "notebook"

/**
 * The Pi sessions of one file's chat (`workbench/chats.json`, G4): `active` is null before the first one, `sessionIds` lists every one, first linked first.
 *
 * - `moved` are sessions the file moved away from (a rename): each is told the new path once, then `markWorkbenchChatInformed`
 */
export interface WorkbenchChatLink {
  active: string | null
  sessionIds: string[]
  moved: string[]
}

/** A PDF chat's extracted text: `ready` at an absolute path its first message names (`Text: <path>`, G3), or `pending` while it is extracted. */
export type WorkbenchPdfText = { state: "ready"; path: string } | { state: "pending" }

/** An opened workbench chat: its live Pi `document` session, the file's link, its kind, folder and, for a PDF, its text. */
export interface OpenedWorkbenchChat {
  session: PiSessionInfo
  link: WorkbenchChatLink
  kind: WorkbenchChatKind
  cwd: string
  pdfText: WorkbenchPdfText | null
}

interface ChatLinkResponse {
  session_id: string | null
  session_ids: string[]
  moved: string[]
}

function toChatLink(linked: ChatLinkResponse): WorkbenchChatLink {
  return { active: linked.session_id, sessionIds: linked.session_ids, moved: linked.moved }
}

/**
 * Open the Pi chat of *path* (an absolute file, LaTeX project folder or notebook, or a stored canvas key) and make it the file's active one.
 *
 * - *sessionId* resumes that session, *new* starts one, otherwise the active session resumes (a new one without)
 * - *model* and *thinking* apply to a new session only, a resumed one keeps its own
 */
export async function openWorkbenchChat({ path, sessionId, new: fresh = false, model, thinking }: { path: string; sessionId?: string; new?: boolean } & PiSessionOptions): Promise<OpenedWorkbenchChat> {
  const opened = await post<{ session: PiSessionInfo; link: ChatLinkResponse; kind: WorkbenchChatKind; cwd: string; pdf_text: { state: "ready" | "pending"; path: string | null } | null }>(
    "/workbench/chat",
    { path, session_id: sessionId ?? null, new: fresh, model: model ?? null, thinking: thinking ?? null },
  )
  const pdf = opened.pdf_text
  const pdfText: WorkbenchPdfText | null = !pdf ? null : pdf.state === "ready" && pdf.path ? { state: "ready", path: pdf.path } : { state: "pending" }
  return { session: opened.session, link: toChatLink(opened.link), kind: opened.kind, cwd: opened.cwd, pdfText }
}

/** The file's chat link and the linked Pi sessions pi-host still has, in link order. */
export async function fetchWorkbenchChats(path: string): Promise<{ link: WorkbenchChatLink; sessions: PiSessionInfo[] }> {
  const listed = await request<{ link: ChatLinkResponse; sessions: PiSessionInfo[] }>(`/workbench/chats${toQuery({ path })}`)
  return { link: toChatLink(listed.link), sessions: listed.sessions }
}

/** Session *sessionId* was told where the file at *path* is now, so it leaves the link's `moved`. Resolves to the link. */
export async function markWorkbenchChatInformed(path: string, sessionId: string): Promise<WorkbenchChatLink> {
  return toChatLink(await post<ChatLinkResponse>("/workbench/chat/informed", { path, session_id: sessionId }))
}
