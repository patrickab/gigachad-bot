"use client"

import { TextQuote, X } from "lucide-react"
import { CHIP_CLASS, CHIP_REMOVE_CLASS } from "./AttachmentChip"
import type { ComposerChip } from "./composerChips"

/** One staged text chip above the composer (`composerChips`), drawn like an attachment chip. */
export function TextChip({ chip, onRemove }: { chip: ComposerChip; onRemove: () => void }) {
  return (
    <div className={CHIP_CLASS}>
      <TextQuote className="h-3.5 w-3.5 shrink-0 text-ink-muted" aria-hidden="true" />
      <span className="min-w-0 truncate text-ink">{chip.label}</span>
      <button type="button" aria-label={`Remove ${chip.label}`} onClick={onRemove} className={CHIP_REMOVE_CLASS}>
        <X className="h-3 w-3" />
      </button>
    </div>
  )
}
