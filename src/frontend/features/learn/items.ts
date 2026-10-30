import type { WorkspaceItem } from "@/contexts/TabContext"
import type { LearnSession, LearnView } from "@/lib/domain/types"

/** Workspace item that shows `view` in the learn frame, opened from project `projectSlug` (null outside projects). */
export function learnItem(view: LearnView, projectSlug: string | null = null): WorkspaceItem {
  return { kind: "learn", path: "", projectSlug, learn: view }
}

/** Tab title of a learn tab showing `view`: a session's own title once `sessions` holds it. */
export function learnTitle(view: LearnView, sessions: readonly LearnSession[] = []): string {
  if (view.kind === "session") return sessions.find((session) => session.id === view.sessionId)?.title ?? "Learn session"
  if (view.kind === "new") return view.sessionKind === "practice" ? "New practice session" : "New learn session"
  return "Learn"
}

/** Opens a learn item in the current tab, or a new one when `inNewTab`. */
export type OpenItem = (item: WorkspaceItem, inNewTab?: boolean) => void
