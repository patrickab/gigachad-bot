"use client"

import { ImageIcon } from "lucide-react"
import { useEventListener } from "@/hooks/useEventListener"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import type { CanvasContextMenuAt } from "@/features/editor/canvas/ink/useInkSurface"

/** The long-press menu, mounted only while open: any press elsewhere or Escape closes it. */
export function CanvasContextMenu({ at, canPaste, onClose, onPasteImage }: {
  at: CanvasContextMenuAt
  canPaste: boolean
  onClose: () => void
  onPasteImage: (canvasX: number, canvasY: number) => void
}) {
  useEventListener("document", "pointerdown", onClose)
  useEscapeLayer(onClose)
  return (
    <div
      className="absolute z-20 rounded-lg border border-divider bg-paper shadow-[var(--shadow-lg)] py-1 min-w-[100px]"
      style={{ left: at.left, top: at.top }}
      onPointerDown={(e) => e.stopPropagation()}
    >
      <button
        onClick={() => {
          onClose()
          onPasteImage(at.canvasX, at.canvasY)
        }}
        disabled={!canPaste}
        className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[11px] text-ink-muted hover:text-ink hover:bg-hover transition-colors disabled:opacity-30"
      >
        <ImageIcon className="h-3 w-3 shrink-0 text-ink-faint" />
        Paste image
      </button>
    </div>
  )
}
