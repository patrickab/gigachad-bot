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
      // The SVG's tip sits at (20, 1)
      style={{ left: bounds.left - 20, top: bounds.bottom - 1 }}
      className="fixed z-50 h-7 w-7 rounded-lg text-ink-muted transition-colors hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
    >
      <svg aria-hidden="true" viewBox="0 0 28 28" className="h-7 w-7 fill-paper stroke-current drop-shadow-sm" strokeWidth="1.5" strokeLinejoin="round">
        <path d="M20 1L14 6H7a6 6 0 0 0-6 6v9a6 6 0 0 0 6 6h14a6 6 0 0 0 6-6v-9a6 6 0 0 0-6-6h-1z" />
        <g className="fill-current stroke-none">
          <circle cx="9" cy="16.5" r="1.4" />
          <circle cx="14" cy="16.5" r="1.4" />
          <circle cx="19" cy="16.5" r="1.4" />
        </g>
      </svg>
    </button>,
    document.body,
  )
}
