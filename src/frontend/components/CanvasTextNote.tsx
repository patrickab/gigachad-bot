"use client"

import type { CanvasText } from "@/lib/canvasDocument"
import { inkColor } from "@/lib/canvasInk"
import { LaTeXMarkdown } from "./LaTeXMarkdown"

interface CanvasTextNoteProps {
  note: CanvasText
  /** Effective canvas zoom and pan: the note is positioned HTML over the SVG. */
  scale: number
  offset: { x: number; y: number }
  isDark: boolean
  editing: boolean
  autoFocus: boolean
  onStartEditing: () => void
  onEdit: (text: string) => void
  onFocusNote: () => void
  onBlurNote: () => void
  onDragStart: (mode: "move" | "resize", clientX: number, clientY: number) => void
}

const DRAG_BAND_PX = 6

/** A text note: Markdown while unfocused, a source textarea once clicked. Rasterized as source text in canvas exports. */
export function CanvasTextNote({ note: t, scale, offset, isDark, editing, autoFocus, onStartEditing, onEdit, onFocusNote, onBlurNote, onDragStart }: CanvasTextNoteProps) {
  return (
    <div className="absolute group/text" style={{ left: t.x * scale + offset.x, top: t.y * scale + offset.y, width: t.width * scale, height: t.height * scale }}>
      {editing ? (
        <textarea
          autoFocus={autoFocus}
          value={t.text}
          onChange={(e) => onEdit(e.target.value)}
          onFocus={onFocusNote}
          onBlur={onBlurNote}
          onKeyDown={(e) => {
            e.stopPropagation()
            if (e.key === "Enter" && !(e.ctrlKey || e.metaKey)) {
              e.preventDefault()
              ;(e.target as HTMLTextAreaElement).blur()
            } else if (e.key === "Escape") {
              e.preventDefault()
              ;(e.target as HTMLTextAreaElement).blur()
            }
          }}
          onPointerDown={(e) => e.stopPropagation()}
          className="absolute inset-0 resize-none bg-transparent outline-none border border-transparent focus:border-dashed focus:border-divider-strong rounded px-1"
          style={{
            fontSize: t.size * scale,
            lineHeight: 1.2,
            color: inkColor(t.color, isDark),
          }}
        />
      ) : (
        <div
          className="absolute inset-0 cursor-text overflow-auto rounded px-1 py-0.5"
          onPointerDown={(e) => e.stopPropagation()}
          onClick={(e) => {
            if ((e.target as HTMLElement).closest("a")) return
            onStartEditing()
          }}
        >
          <LaTeXMarkdown content={t.text} style={{ fontSize: t.size * scale, color: inkColor(t.color, isDark) }} />
        </div>
      )}
      {/* top drag-band (move) — thin sliver above the textarea, dim until hover */}
      <div
        className="absolute inset-x-0 top-0 opacity-0 group-hover/text:opacity-60 hover:!opacity-100 transition-opacity bg-[var(--ink-faint)] rounded-t"
        style={{ height: DRAG_BAND_PX, cursor: "grab" }}
        onPointerDown={(e) => { e.stopPropagation(); onDragStart("move", e.clientX, e.clientY) }}
      />
      {/* resize handle (bottom-right) — broadens/elongates the box */}
      <div
        className="absolute bottom-0 right-0 w-3 h-3 opacity-0 group-hover/text:opacity-100 transition-opacity cursor-nwse-resize"
        onPointerDown={(e) => { e.stopPropagation(); onDragStart("resize", e.clientX, e.clientY) }}
      >
        <svg className="w-full h-full text-ink-faint" viewBox="0 0 16 16">
          <path d="M14 2L2 14M14 6L6 14M14 10L10 14" stroke="currentColor" strokeWidth="1.5" fill="none" />
        </svg>
      </div>
    </div>
  )
}
