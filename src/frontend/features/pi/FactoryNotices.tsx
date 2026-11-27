"use client"

import { Disclosure } from "@/components/ui/Disclosure"
import { isRecord } from "@/lib/utils"
import { factoryCard } from "./factoryCard"
import { FactoryEvent } from "./factoryView"
import { registerMessageRenderer } from "./renderers"
import { textOf } from "./transcript"

/**
 * The notice the agent factory sends into the session that declared a task or wave (`factory-notice`). With a card
 * (`details.card`) it shows the card, keyed by `details.key` for the decision that answers it. Without one its text,
 * meant for the agent (only old transcripts): the first line, the rest opens on demand as plain text.
 */
registerMessageRenderer("factory-notice", (message, next) => {
  const card = factoryCard(message.details)
  if (card) {
    const key = isRecord(message.details) && typeof message.details.key === "string" ? message.details.key : undefined
    return <FactoryEvent card={card} text={textOf(message.content)} noticeKey={key} />
  }
  if (typeof message.content !== "string" || !message.content.trim()) return next()
  const [first, ...rest] = message.content.trim().split("\n")
  const title = first.replace(/^factory:\s*/, "")
  if (rest.length === 0) return <p className="my-1 truncate text-xs text-ink-muted">{title}</p>
  return (
    <Disclosure className="my-1" summary={<span className="truncate">{title}</span>}>
      <pre className="whitespace-pre-wrap break-words pl-5 font-sans text-xs text-ink-muted">{rest.join("\n")}</pre>
    </Disclosure>
  )
})
