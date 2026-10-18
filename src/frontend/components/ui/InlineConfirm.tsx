"use client"

import { useEffect, useRef } from "react"

const ANSWER_CLASS = "rounded px-1 text-ink-muted hover:bg-hover hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted disabled:opacity-50"

/** One-line yes/no question: Enter answers yes, Escape no, both answers are also buttons. */
export function InlineConfirm({ question, busy, onYes, onNo }: {
  question: string
  busy?: boolean
  onYes: () => void
  onNo: () => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => { ref.current?.focus() }, [question])
  return <div
    ref={ref} tabIndex={-1} role="group" aria-label={question}
    onKeyDown={(event) => {
      if (busy) return
      if (event.key === "Enter") { event.preventDefault(); onYes() }
      if (event.key === "Escape") { event.stopPropagation(); onNo() }
    }}
    className="flex items-center gap-1 py-0.5 text-[11px] text-ink-subtle outline-none"
  >
    <span className="min-w-0 flex-1">{question}</span>
    <button disabled={busy} onClick={onYes} className={ANSWER_CLASS}>Yes</button>
    <button disabled={busy} onClick={onNo} className={ANSWER_CLASS}>No</button>
  </div>
}
