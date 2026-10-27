"use client"

import { createPortal } from "react-dom"

/** Speech bubble button whose tip touches the lower-left corner of a text selection's `bounds`. Pressing it keeps the selection. */
export function SelectionBubble({ bounds, label, onActivate }: { bounds: DOMRect; label: string; onActivate: () => void }) {
  return createPortal(
    <button
      type="button"
      aria-label={label}
      onPointerDown={(event) => event.preventDefault()}
      onClick={onActivate}
      // The SVG's tip sits at (22, 1)
      style={{ left: bounds.left - 22, top: bounds.bottom - 1 }}
      className="fixed z-50 h-8 w-8 rounded text-ink-muted hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
    >
      <svg aria-hidden="true" viewBox="0 0 32 32" className="h-8 w-8 fill-paper stroke-current" strokeWidth="2.25" strokeLinejoin="round">
        <path d="M22 1v6q0 3 3 3h1q5 0 5 6v8q0 6-6 6H7q-6 0-6-6v-8q0-6 6-6h5q2 0 4-2z" />
      </svg>
    </button>,
    document.body,
  )
}
