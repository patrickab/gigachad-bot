"use client"

import { createPortal } from "react-dom"
import { Trash2 } from "lucide-react"
import type { ArchitectureElement } from "@/lib/domain/architecture"

/** Asks whether elements just removed from a saved view should leave the model too; Enter and Escape come from `useRemoveFromViewOffer`. */
export function RemoveReferenceDialog({ elements, onAnswer }: {
  elements: ArchitectureElement[]
  onAnswer: (remove: boolean) => void
}) {
  return createPortal(
    // Portaled: the window sits inside the canvas's transform, which would confine a fixed overlay.
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-backdrop backdrop-blur-[2px]" onClick={() => onAnswer(false)}>
      <div role="alertdialog" aria-label="Remove reference" onClick={(event) => event.stopPropagation()}
        className="mx-4 flex flex-col gap-3 rounded-xl border border-divider bg-paper px-5 py-4 text-sm text-ink shadow-[var(--shadow-xl)]">
        <span className="flex items-center gap-2">
          <Trash2 className="h-4 w-4" />
          Remove reference from {[...new Set(elements.map((element) => element.file ?? "the model"))].join(", ")}?
        </span>
        <span className="flex justify-end gap-2 text-xs">
          <button type="button" onClick={() => onAnswer(true)} className="rounded-lg border border-divider bg-surface px-3 py-1.5 text-ink hover:border-ink-muted">Yes (Enter)</button>
          <button type="button" onClick={() => onAnswer(false)} className="rounded-lg border border-divider px-3 py-1.5 text-ink-muted hover:border-ink-muted hover:text-ink">No (Esc)</button>
        </span>
      </div>
    </div>,
    document.body,
  )
}
