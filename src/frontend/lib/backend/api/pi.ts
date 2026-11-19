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

/** How a session opens: a fresh one, the repo's newest, or a given session file. */
export type PiResume = null | "continue" | { sessionPath: string }

/** Git checkouts directly under `~/git` a Pi session may start in. */
export async function fetchPiRepositories(): Promise<PiRepository[]> {
  return request<PiRepository[]>("/pi/repos")
}

/** The repo's sessions, newest first. */
export async function fetchPiSessions(repo: string): Promise<PiSessionInfo[]> {
  const { sessions } = await request<{ sessions: PiSessionInfo[] }>(`/pi/sessions?repo=${encodeURIComponent(repo)}`)
  return sessions
}

/** Spawn a session in `repo`, or get the already-live one. Rejects with status 409 while another process holds the file. */
export async function openPiSession(repo: string, resume: PiResume): Promise<PiSessionInfo> {
  const { session } = await post<{ session: PiSessionInfo }>("/pi/sessions", { repo, resume })
  return session
}
