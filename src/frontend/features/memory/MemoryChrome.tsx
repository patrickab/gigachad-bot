"use client"

import { ArrowLeft, Loader2 } from "lucide-react"
import type { ScopeMemory } from "@/features/memory/scopeEdits"

/** Centered loading indicator of a memory view. */
export function Spinner({ label }: { label: string }) {
  return (
    <div className="flex items-center justify-center py-16">
      <Loader2 className="h-4 w-4 animate-spin text-ink-muted" aria-label={label} />
    </div>
  )
}

/** Back link to the memory overview, naming the project of a project scope. */
export function PageHeader({ scope, onBack }: { scope: ScopeMemory; onBack: () => void }) {
  return (
    <div className="flex shrink-0 items-center gap-2 px-5 pb-2 pt-5 text-sm">
      <button onClick={onBack}
        className="flex items-center gap-2 rounded-md px-1 py-0.5 text-ink-muted transition-colors hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
        <ArrowLeft className="h-4 w-4" />Memory
      </button>
      {scope.scope === "project" && <span className="truncate text-ink-subtle">/ {scope.label}</span>}
    </div>
  )
}
