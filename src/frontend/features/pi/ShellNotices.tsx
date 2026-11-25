"use client"

import { registerMessageRenderer } from "./renderers"

/**
 * The notice pi-interactive-shell injects when a background session ends (`interactive-shell-transfer`): one quiet line
 * (name, outcome, duration) in place of its text, which repeats the exit code and a log path meant for the agent.
 * Read leniently: a notice without a session name keeps its text.
 */
registerMessageRenderer("interactive-shell-transfer", (message, next) => {
  const details = typeof message.details === "object" && message.details !== null ? (message.details as Record<string, unknown>) : {}
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
