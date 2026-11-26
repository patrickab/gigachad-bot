// Pi agent routes: repositories and pi-host sessions (`backend/routes/pi.py`).
import { toQuery } from "../urls"
import { ApiError, fileForm, jsonInit, post, request, requestVoid } from "./http"

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
  /** Context size after the last answer, 0 before one. */
  tokens: number
  parentSession?: string
  live: PiLiveStatus | null
  pinned: boolean
  archived: boolean
  /** Session profile (G5): coding tabs, or a workbench `document` chat. Sessions written before profiles are `coding`. */
  profile?: PiProfile
  /** Until when the provider likely keeps the prompt cached (ISO), absent when the last answer used no cache. */
  cachedUntil?: string
}

/** What a session runs as: a coding agent in a repo, or a workbench chat about one document. */
export type PiProfile = "coding" | "document"

/** Where `@` completion finds files: a checkout under `~/git` by name, or an absolute folder (a workbench chat's cwd). */
export type PiFileScope = { repo: string } | { cwd: string }

/** How a session opens: a fresh one, the repo's newest, a given session file, or a fork of a file at an entry (D11). */
export type PiResume = null | "continue" | { sessionPath: string } | { forkOf: { sessionPath: string; entryId: string } }

/** Model and thinking level a new session starts with, Pi's defaults when omitted. */
export interface PiSessionOptions {
  model?: { provider: string; id: string }
  thinking?: string
}

/** One hit in a session: the `ordinal`th match (text order) in the user or assistant text of entry `entryId`, with the text around it. */
export interface PiSessionMatch {
  entryId: string
  ordinal: number
  quality: number
  before: string
  match: string
  after: string
}

/** A session matching a search, with the matched text and the entry it sits in, and its best few hits (pi-host `GET /search`, M24). */
export interface PiSearchHit {
  session: PiSessionInfo
  snippet: string
  entryId: string
  matches: PiSessionMatch[]
}

/** A session's ranked hits for a query, `total` counting those beyond the `limit` asked. */
export interface PiSessionMatches {
  matches: PiSessionMatch[]
  total: number
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

/** The best `limit` hits of one session's transcript for `query`, best first. */
export async function fetchPiSessionMatches(sessionId: string, query: string, limit = 100, signal?: AbortSignal): Promise<PiSessionMatches> {
  return request(`/pi/sessions/${encodeURIComponent(sessionId)}/matches?q=${encodeURIComponent(query)}&limit=${limit}`, { signal })
}

/** Up to 500 files of `scope`, unranked (the UI ranks them, K6), for `@` completion. */
export async function fetchPiFiles(scope: PiFileScope): Promise<string[]> {
  const where = "repo" in scope ? `repo=${encodeURIComponent(scope.repo)}` : `cwd=${encodeURIComponent(scope.cwd)}`
  const { files } = await request<{ files: { path: string }[] }>(`/pi/files?${where}&limit=500`)
  return files.map((file) => file.path)
}

/** Store `video` in `.pi/pastes/` of `scope`'s folder, resolves to its path relative to that folder (what follows `@`). */
export async function pastePiVideo(video: File, scope: PiFileScope): Promise<string> {
  const { mention } = await request<{ mention: string }>(`/pi/pastes${toQuery(scope)}`, { method: "POST", body: fileForm(video) })
  return mention
}

/**
 * Spawn a session in `repo`, or get the already-live one. Rejects with status 409 while another process holds the file.
 * Resuming a session file is safe to repeat, so it waits out a restarting backend. A new session or a fork is sent once.
 */
export async function openPiSession(repo: string, resume: PiResume, options: PiSessionOptions = {}): Promise<PiSessionInfo> {
  const retryOffline = resume !== null && typeof resume === "object" && "sessionPath" in resume
  const { session } = await post<{ session: PiSessionInfo }>("/pi/sessions", { repo, resume, ...options }, { retryOffline })
  return session
}

/** Delete a session file for good. */
export function deletePiSession(sessionId: string): Promise<void> {
  return requestVoid(`/pi/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" })
}

export type PiTurnRefusal = "running" | "compacted" | "locked" | "changed"

/**
 * Delete the turn that `entryId` (a user prompt) starts for good: `through: "turn"` that turn, `"end"` it and every later one.
 * A 409 rejects with an ApiError whose `code` and `message` are the reason (`PiTurnRefusal`).
 */
export async function deletePiTurn(sessionId: string, entryId: string, through: "turn" | "end"): Promise<{ removedEntries: number }> {
  try {
    const answer = await post<{ removedEntries: number }>(
      `/pi/sessions/${encodeURIComponent(sessionId)}/turns/${encodeURIComponent(entryId)}/delete`,
      { through },
    )
    return { removedEntries: answer.removedEntries }
  } catch (error) {
    if (error instanceof ApiError && error.status === 409) throw new ApiError(error.message, 409, error.message)
    throw error
  }
}

/** Move a session file into the archive (M9). */
export async function archivePiSession(sessionId: string): Promise<PiSessionInfo> {
  const { session } = await post<{ session: PiSessionInfo }>(`/pi/sessions/${encodeURIComponent(sessionId)}/archive`, {})
  return session
}

/** Move an archived session file back (M9). */
export async function restorePiSession(sessionId: string): Promise<PiSessionInfo> {
  const { session } = await post<{ session: PiSessionInfo }>(`/pi/sessions/${encodeURIComponent(sessionId)}/restore`, {})
  return session
}

/** Pin or unpin a session: pinned ones are never evicted or archived automatically (M20). */
export async function pinPiSession(sessionId: string, pinned: boolean): Promise<PiSessionInfo> {
  const { session } = await request<{ session: PiSessionInfo }>(`/pi/sessions/${encodeURIComponent(sessionId)}/pin`, { method: "PUT", body: JSON.stringify({ pinned }) })
  return session
}

/** A persisted session entry (pi docs `session-format.md`), its type-specific fields untyped. */
export interface PiSessionEntry {
  type: string
  id: string
  parentId: string | null
  timestamp: string
  [field: string]: unknown
}

/** Every entry of a session in append order, all branches, and the current leaf: pi's `get_entries` shape. */
export interface PiSessionEntries {
  entries: PiSessionEntry[]
  leafId: string | null
}

/** The persisted entries of a session, read from its file: no child attaches. `turns` keeps only the active branch's last ones. */
export async function fetchPiSessionEntries(sessionId: string, options: { turns?: number } = {}): Promise<PiSessionEntries> {
  const query = options.turns === undefined ? "" : `?turns=${options.turns}`
  return request(`/pi/sessions/${encodeURIComponent(sessionId)}/entries${query}`)
}

/** pi-host's settings: reopen the runs a pi-host crash or reboot interrupted. */
export interface PiSettings {
  autoResume: boolean
}

export async function fetchPiSettings(signal?: AbortSignal): Promise<PiSettings> {
  return request<PiSettings>("/pi/settings", { signal })
}

export async function savePiSettings(settings: PiSettings): Promise<void> {
  await request<PiSettings>("/pi/settings", await jsonInit("PUT", settings))
}
