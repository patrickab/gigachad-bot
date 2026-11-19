// Pi agent routes: repositories and pi-host sessions (`backend/routes/pi.py`).
import { post, request } from "./http"

export interface PiRepository {
  name: string
  path: string
}

/** Live child of a session: whether a run streams, waits on a dialog, and how many clients watch. */
export interface PiLiveStatus {
  running: boolean
  waiting: boolean
  attached: number
}

/** One Pi session file as pi-host reports it, ISO timestamps. */
export interface PiSessionInfo {
  sessionId: string
  sessionPath: string
  cwd: string
  machine: "local"
  createdAt: string
  modifiedAt: string
  firstMessage: string
  messageCount: number
  parentSession?: string
  live: PiLiveStatus | null
  pinned: boolean
  archived: boolean
}

/** How a session opens: a fresh one, the repo's newest, a given session file, or a fork of a file at an entry (D11). */
export type PiResume = null | "continue" | { sessionPath: string } | { forkOf: { sessionPath: string; entryId: string } }

/** Model and thinking level a new session starts with, Pi's defaults when omitted. */
export interface PiSessionOptions {
  model?: { provider: string; id: string }
  thinking?: string
}

/** A session matching a search, with the matched text and the entry it sits in (pi-host `GET /search`, M24). */
export interface PiSearchHit {
  session: PiSessionInfo
  snippet: string
  entryId: string
}

/** Git checkouts directly under `~/git` a Pi session may start in. */
export async function fetchPiRepositories(): Promise<PiRepository[]> {
  return request<PiRepository[]>("/pi/repos")
}

/** Sessions newest first: of `repo`, of every repo when omitted, the archived ones with `archived`. */
export async function fetchPiSessions(repo?: string, options: { archived?: boolean } = {}): Promise<PiSessionInfo[]> {
  const query = new URLSearchParams()
  if (repo !== undefined) query.set("repo", repo)
  if (options.archived) query.set("archived", "true")
  const search = query.toString()
  const { sessions } = await request<{ sessions: PiSessionInfo[] }>(`/pi/sessions${search ? `?${search}` : ""}`)
  return sessions
}

/** Sessions of every repo whose entries match `query`, best first. */
export async function searchPiSessions(query: string, signal?: AbortSignal): Promise<PiSearchHit[]> {
  const { hits } = await request<{ hits: PiSearchHit[] }>(`/pi/search?q=${encodeURIComponent(query)}`, { signal })
  return hits
}

/** Up to 500 files of `repo`, unranked (the UI ranks them, K6), for `@` completion. */
export async function fetchPiFiles(repo: string): Promise<string[]> {
  const { files } = await request<{ files: { path: string }[] }>(`/pi/files?repo=${encodeURIComponent(repo)}&limit=500`)
  return files.map((file) => file.path)
}

/** Spawn a session in `repo`, or get the already-live one. Rejects with status 409 while another process holds the file. */
export async function openPiSession(repo: string, resume: PiResume, options: PiSessionOptions = {}): Promise<PiSessionInfo> {
  const { session } = await post<{ session: PiSessionInfo }>("/pi/sessions", { repo, resume, ...options })
  return session
}
