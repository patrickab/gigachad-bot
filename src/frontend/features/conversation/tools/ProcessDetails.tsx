// Ported from kahme247/ompweb components/ChatWindow-transcript-group-row.tsx (MIT, Copyright (c) 2026 agegr).
"use client"

import { useId, useState, type JSX, type ReactNode } from "react"
import { ChevronDown, Layers } from "lucide-react"
import "./tools.css"

/** Folded "Process details · N messages · M tool calls" toggle; `children` renders only while expanded. */
export function ProcessDetails({ messageCount, toolCallCount, children }: { messageCount: number; toolCallCount: number; children: () => ReactNode }): JSX.Element {
  const [expanded, setExpanded] = useState(false)
  const contentId = useId()
  const parts = ["Process details", `${messageCount} message${messageCount === 1 ? "" : "s"}`]
  if (toolCallCount > 0) parts.push(`${toolCallCount} tool call${toolCallCount === 1 ? "" : "s"}`)

  return (
    <div className="pi-process-details">
      <button
        type="button"
        aria-expanded={expanded}
        aria-controls={contentId}
        onClick={() => setExpanded((value) => !value)}
        className="pi-process-details-toggle"
      >
        <Layers size={12} strokeWidth={1.8} aria-hidden className="pi-process-details-icon" />
        <ChevronDown size={12} strokeWidth={1.8} aria-hidden className="pi-process-details-chevron" data-expanded={expanded || undefined} />
        <span className="pi-process-details-label">{parts.join(" · ")}</span>
      </button>
      {expanded && <div id={contentId} className="pi-process-details-content">{children()}</div>}
    </div>
  )
}
