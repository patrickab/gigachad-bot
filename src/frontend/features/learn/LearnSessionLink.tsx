"use client"

import { useEffect } from "react"
import { learnItem } from "./items"
import { useLearn } from "./LearnProvider"

/** A link that shows learn session `sessionId` in a new tab, as a session of project `projectSlug` (null: none). */
export function LearnSessionLink({ sessionId, title, projectSlug }: { sessionId: string; title: string; projectSlug: string | null }) {
  const { learnUrl, sessions, openItem } = useLearn()
  const known = sessions.sessions.some((session) => session.id === sessionId)
  const { lookUp } = sessions

  // A session started from the chat joins the sidebar now, not on the next poll.
  useEffect(() => {
    if (learnUrl && !known) void lookUp(sessionId)
  }, [learnUrl, known, lookUp, sessionId])

  if (!known && sessions.missing.has(sessionId)) {
    return <span className="text-sm text-ink-faint">{title} (deleted)</span>
  }
  return (
    <button
      type="button"
      onClick={() => openItem(learnItem({ kind: "session", sessionId }, projectSlug), true)}
      className="text-left text-sm text-ink underline decoration-divider-strong underline-offset-2 hover:decoration-ink"
    >
      {title}
    </button>
  )
}
