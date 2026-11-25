"use client"

import { Disclosure } from "@/components/ui/Disclosure"
import { registerMessageRenderer } from "./renderers"

/** Stages of a notice's first line that need the user's eye. */
const ATTENTION = /rejected|blocked|interrupted|asks|waits|never start/

/**
 * The notice the agent factory sends into the session that declared a task or wave (`factory-notice`). Its text is
 * meant for the agent: the transcript shows its first line, the rest opens on demand as plain text.
 */
registerMessageRenderer("factory-notice", (message, next) => {
  if (typeof message.content !== "string" || !message.content.trim()) return next()
  const [first, ...rest] = message.content.trim().split("\n")
  const title = first.replace(/^factory:\s*/, "")
  const ink = ATTENTION.test(title) ? "text-warning" : undefined
  if (rest.length === 0) return <p className={`my-1 truncate text-xs ${ink ?? "text-ink-muted"}`}>{title}</p>
  return (
    <Disclosure className="my-1" summary={<span className="truncate">{title}</span>} summaryClassName={ink}>
      <pre className="whitespace-pre-wrap break-words pl-5 font-sans text-xs text-ink-muted">{rest.join("\n")}</pre>
    </Disclosure>
  )
})
