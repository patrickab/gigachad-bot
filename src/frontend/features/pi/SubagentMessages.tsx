"use client"

import { ChevronRight } from "lucide-react"
import type { ReactNode } from "react"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { isRecord } from "@/lib/utils"
import { registerMessageRenderer } from "./renderers"
import { str } from "./subagents"
import { textOf } from "./transcript"
import type { CustomMessage } from "./types"

/**
 * Cards for the messages the user's pi-subagents and pi-intercom extensions inject into the transcript:
 * `intercom_message` (a child or peer session writes to this one) and `subagent-notify` (a background run settled).
 * A quiet one-line summary, the full text folds open on demand. Both read their `details` leniently.
 */
function Card({ source, summary, body }: { source: string; summary: string; body: string }) {
  const folds = body.trim() !== "" && body.trim() !== summary.trim()
  const head: ReactNode = (
    <>
      <span className="shrink-0 font-medium text-ink">{source}</span>
      <span className="min-w-0 flex-1 truncate">{summary}</span>
    </>
  )
  return (
    <div className="my-1.5 rounded-lg bg-surface-elevated px-3 py-1.5 text-xs text-ink-muted">
      {folds ? (
        <details className="group">
          <summary className="flex cursor-pointer list-none items-center gap-2">
            <ChevronRight size={12} aria-hidden="true" className="shrink-0 transition-transform group-open:rotate-90" />
            {head}
          </summary>
          <div className="mt-1.5 text-sm text-ink"><LaTeXMarkdown content={body} /></div>
        </details>
      ) : (
        <div className="flex items-center gap-2">{head}</div>
      )}
    </div>
  )
}

/** The message text without the `**From …**` header and delivery metadata `pi-intercom` prepends to it. */
function intercomBody(message: CustomMessage): { source: string; body: string } | null {
  const details = isRecord(message.details) ? message.details : undefined
  const from = details && isRecord(details.from) ? details.from : undefined
  const inner = details && isRecord(details.message) ? details.message : undefined
  const content = inner && isRecord(inner.content) ? inner.content : undefined
  const body = str(details?.bodyText) ?? str(content?.text)
  const source = str(from?.name) ?? /^\*\*From (.+?)\*\*/.exec(textOf(message.content))?.[1]
  return body && source ? { source, body } : null
}

const firstLine = (text: string) => text.split("\n").find((line) => line.trim() !== "")?.trim() ?? ""

registerMessageRenderer("intercom_message", (message, next) => {
  const parsed = intercomBody(message)
  return parsed ? <Card source={parsed.source} summary={firstLine(parsed.body)} body={parsed.body} /> : next()
})

registerMessageRenderer("subagent-notify", (message, next) => {
  const details = isRecord(message.details) ? message.details : undefined
  const runs = Array.isArray(details?.runs) ? details.runs.filter(isRecord) : []
  if (runs.length === 0) return next()
  const agents = runs.map((run) => str(run.agent) ?? "subagent").join(", ")
  const statuses = [...new Set(runs.map((run) => str(run.status) ?? "done"))].join(", ")
  return <Card source={agents} summary={statuses} body={textOf(message.content)} />
})
