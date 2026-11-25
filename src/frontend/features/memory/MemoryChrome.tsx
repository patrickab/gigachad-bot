"use client"

import { ArrowLeft, Loader2 } from "lucide-react"

/** Centered loading indicator of a memory view. */
export function Spinner({ label }: { label: string }) {
  return (
    <div className="flex items-center justify-center py-16">
      <Loader2 className="h-4 w-4 animate-spin text-ink-muted" aria-label={label} />
    </div>
  )
}

/** Breadcrumb back link: the parent level to return to, then the current one. */
export function PageHeader({ parent, current, onBack }: { parent: string; current: string; onBack: () => void }) {
  return (
    <div className="flex min-w-0 items-center gap-2 text-sm">
      <button onClick={onBack} aria-label={`Back to ${parent}`}
        className="flex shrink-0 items-center gap-2 rounded-md px-1 py-0.5 text-ink-muted transition-colors hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
        <ArrowLeft className="h-4 w-4" />{parent}
      </button>
      <span className="truncate text-ink-subtle">/ {current}</span>
    </div>
  )
}
