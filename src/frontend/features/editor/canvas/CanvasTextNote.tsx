"use client"

import { useRef, useState } from "react"
import type { CanvasText } from "@/lib/domain/canvasDocument"
import type { Point } from "@/lib/domain/edgeRouting"
import { CanvasResizeHandle } from "@/features/editor/canvas/CanvasResizeHandle"
import { inkColor } from "@/features/editor/canvas/ink/canvasInk"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { isModKey } from "@/lib/utils"

interface CanvasTextNoteProps {
  note: CanvasText
  /** Effective canvas zoom and pan: the note is positioned HTML over the SVG. */
  scale: number
  offset: Point
  editing: boolean
  autoFocus: boolean
  onStartEditing: () => void
  onEdit: (text: string) => void
  onFocusNote: () => void
  onBlurNote: () => void
  onDragStart: (mode: "move" | "resize", clientX: number, clientY: number) => void
}

const DRAG_BAND_PX = 6

type NoteSourceEditorProps = Pick<CanvasTextNoteProps, "scale" | "autoFocus" | "onEdit" | "onFocusNote" | "onBlurNote"> & { note: CanvasText }

/** The note's source textarea: Enter or Escape blurs it, and while focused it owns Escape so an enclosing overlay stays open. */
function NoteSourceEditor({ note: t, scale, autoFocus, onEdit, onFocusNote, onBlurNote }: NoteSourceEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const [focused, setFocused] = useState(false)
  useEscapeLayer(() => textareaRef.current?.blur(), focused)
  return (
    <textarea
      ref={textareaRef}
      autoFocus={autoFocus}
      value={t.text}
      onChange={(e) => onEdit(e.target.value)}
      onFocus={() => { setFocused(true); onFocusNote() }}
      onBlur={() => { setFocused(false); onBlurNote() }}
      onKeyDown={(e) => {
        e.stopPropagation()
        if (e.key === "Enter" && !isModKey(e)) {
          e.preventDefault()
          e.currentTarget.blur()
        }
      }}
      onPointerDown={(e) => e.stopPropagation()}
      className="absolute inset-0 resize-none bg-transparent outline-none border border-transparent focus:border-dashed focus:border-divider-strong rounded px-1"
      style={{
        fontSize: t.size * scale,
        lineHeight: 1.2,
        color: inkColor(t.color),
      }}
    />
  )
}

/** A text note: Markdown while unfocused, a source textarea once clicked. Rasterized as source text in canvas exports. */
export function CanvasTextNote({ note: t, scale, offset, editing, autoFocus, onStartEditing, onEdit, onFocusNote, onBlurNote, onDragStart }: CanvasTextNoteProps) {
  return (
    <div className="absolute group/text" style={{ left: t.x * scale + offset.x, top: t.y * scale + offset.y, width: t.width * scale, height: t.height * scale }}>
      {editing ? (
        <NoteSourceEditor note={t} scale={scale} autoFocus={autoFocus} onEdit={onEdit} onFocusNote={onFocusNote} onBlurNote={onBlurNote} />
      ) : (
        <div
          className="absolute inset-0 cursor-text overflow-auto rounded px-1 py-0.5"
          onPointerDown={(e) => e.stopPropagation()}
          onClick={(e) => {
            if ((e.target as HTMLElement).closest("a")) return
            onStartEditing()
          }}
        >
          <LaTeXMarkdown content={t.text} style={{ fontSize: t.size * scale, color: inkColor(t.color) }} />
        </div>
      )}
      {/* top drag-band (move) — thin sliver above the textarea, dim until hover */}
      <div
        className="absolute inset-x-0 top-0 opacity-0 group-hover/text:opacity-60 hover:!opacity-100 transition-opacity bg-[var(--ink-faint)] rounded-t"
        style={{ height: DRAG_BAND_PX, cursor: "grab" }}
        onPointerDown={(e) => { e.stopPropagation(); onDragStart("move", e.clientX, e.clientY) }}
      />
      {/* resize handle (bottom-right) — broadens/elongates the box */}
      <CanvasResizeHandle className="w-3 h-3 opacity-0 group-hover/text:opacity-100 transition-opacity" onDragStart={onDragStart} />
    </div>
  )
}
