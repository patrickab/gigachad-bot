// omp session routes.
import { post, request, requestVoid } from "./http"
import type { Message, OmpArchivedSession, OmpOverview, OmpSystemPrompt } from "@/lib/domain/types"

export async function fetchOmpOverview(signal?: AbortSignal): Promise<OmpOverview> {
  return request<OmpOverview>("/omp/overview", { signal })
}

export async function fetchOmpArchive(): Promise<OmpArchivedSession[]> {
  return request<OmpArchivedSession[]>("/omp/archive")
}

/** Restore an archived omp session, resolves to its session id. */
export async function restoreOmpSession(key: string): Promise<string> {
  const data = await post<{ session_id: string }>("/omp/archive/restore", { key })
  return data.session_id
}

/** Archive an omp session. omp-web refuses (409) while forks of it exist. */
export async function archiveOmpSession(sessionId: string): Promise<void> {
  await requestVoid(`/omp/sessions/${encodeURIComponent(sessionId)}/archive`, { method: "POST" })
}

/** Delete an omp session with its file and artifacts. Its forks are re-parented, not deleted. */
export async function deleteOmpSession(sessionId: string): Promise<void> {
  await requestVoid(`/omp/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" })
}

export async function fetchOmpSystemPrompt(sessionId: string): Promise<OmpSystemPrompt> {
  return request<OmpSystemPrompt>(`/omp/sessions/${encodeURIComponent(sessionId)}/system-prompt`)
}

/** An omp session's active branch as a chat history: user and assistant text, no thinking or tool calls. */
export async function fetchOmpSessionHistory(sessionId: string): Promise<Pick<Message, "role" | "content">[]> {
  return request<Pick<Message, "role" | "content">[]>(`/omp/sessions/${encodeURIComponent(sessionId)}/history`)
}
