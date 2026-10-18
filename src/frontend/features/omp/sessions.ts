import type { OmpOverview, OmpSession, OmpView } from "@/lib/domain/types"

/** Sidebar and tab title of a session: its name, else its first prompt. */
export function sessionTitle(session: OmpSession): string {
  return session.name || session.first_message.trim() || "Untitled session"
}

/** Tab title for what an omp tab shows. */
export function viewTitle(view: OmpView, overview: OmpOverview | null): string {
  if (view.kind === "settings") return "Settings"
  if (view.kind === "new") return "New session"
  const session = overview?.sessions.find((candidate) => candidate.id === view.sessionId)
  return session ? sessionTitle(session) : "Session"
}
