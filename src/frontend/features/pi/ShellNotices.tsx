"use client"

import { registerMessageRenderer } from "./renderers"
import type { AgentMessage, CustomMessage } from "./types"

const detailsOf = (message: CustomMessage): Record<string, unknown> => (typeof message.details === "object" && message.details !== null ? (message.details as Record<string, unknown>) : {})

/** A background session that ended well: meant for the agent, the transcript leaves it out. */
export function isSilentShellNotice(message: AgentMessage): boolean {
  if (message.role !== "custom" || (message as CustomMessage).customType !== "interactive-shell-transfer") return false
  const details = detailsOf(message as CustomMessage)
  return typeof details.sessionId === "string" && details.exitCode === 0
}

/**
 * The notice pi-interactive-shell injects when a background session ends (`interactive-shell-transfer`). One that ended
 * well is not shown at all (`isSilentShellNotice`); a failed one is one line (name, exit code, duration) in place of its
 * text. Read leniently: a notice without a session name keeps its text.
 */
registerMessageRenderer("interactive-shell-transfer", (message, next) => {
  const details = detailsOf(message)
  if (typeof details.sessionId !== "string" || !details.sessionId) return next()
  const failed = typeof details.exitCode === "number" && details.exitCode !== 0
  const outcome = failed ? `failed (exit ${details.exitCode})` : "finished"
  const duration = typeof details.duration === "string" && details.duration ? ` · ${details.duration}` : ""
  return (
    <p className={`my-1 truncate text-xs ${failed ? "text-danger" : "text-ink-muted"}`}>
      {details.sessionId} {outcome}{duration}
    </p>
  )
})
