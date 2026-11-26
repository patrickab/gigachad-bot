"use client"

import { useEffect, useRef, useState } from "react"
import { TextQuote, X } from "lucide-react"
import { useAttachmentHoverCard } from "@/features/chat/preview/AttachmentHoverCard"
import { CHIP_CLASS, CHIP_REMOVE_CLASS } from "./AttachmentChip"
import type { ComposerChip } from "./composerChips"

/**
 * One staged text chip above the composer (`composerChips`), drawn like an attachment chip.
 *
 * - Hovering the label peeks at the text like an attachment
 * - Clicking it opens an editor above the chip, Esc or clicking outside closes it
 */
export function TextChip({ chip, onChange, onRemove }: { chip: ComposerChip; onChange: (text: string) => void; onRemove: () => void }) {
  const [editing, setEditing] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const hover = useAttachmentHoverCard({ name: chip.label, mime: "text/plain", url: "", active: true }, chip.text)

  useEffect(() => {
    if (!editing) return
    const close = (event: MouseEvent) => { if (!rootRef.current?.contains(event.target as Node)) setEditing(false) }
    document.addEventListener("mousedown", close)
    return () => document.removeEventListener("mousedown", close)
  }, [editing])

  return (
    <div ref={rootRef} className={`${CHIP_CLASS} relative`}>
      <TextQuote className="h-3.5 w-3.5 shrink-0 text-ink-muted" aria-hidden="true" />
      <button
        type="button"
        aria-label={`Edit ${chip.label}`}
        onClick={() => setEditing((open) => !open)}
        {...hover.anchorProps}
        className="min-w-0 truncate rounded text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
      >
        {chip.label}
      </button>
      <button type="button" aria-label={`Remove ${chip.label}`} onClick={onRemove} className={CHIP_REMOVE_CLASS}>
        <X className="h-3 w-3" />
      </button>
      {!editing && hover.card}
      {editing && (
        <div className="absolute bottom-full left-0 z-[80] pb-2">
          <textarea
            autoFocus
            aria-label={`Text of ${chip.label}`}
            value={chip.text}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={(event) => { if (event.key === "Escape") { event.stopPropagation(); setEditing(false) } }}
            className="block h-64 w-[min(36rem,80vw)] resize-none rounded-xl border border-divider bg-paper p-2 font-mono text-[11px] text-ink shadow-[var(--shadow-xl)] focus:outline-none"
          />
        </div>
      )}
    </div>
  )
}
